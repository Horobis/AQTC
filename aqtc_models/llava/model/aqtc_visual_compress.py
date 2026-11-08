"""AQTC visual-token compression helpers for LLaVA-Video.

Reuses the shared novelty / scoring / four-quadrant planner under ``aqtc_models``.
Saliency is last-layer SigLIP received-attention (pooled to match visual tokens);
query uses mean text embedding cosine similarity.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Profiling mode: set AQTC_PROFILING=1 to enable per-stage synchronization
_AQTC_PROFILING = os.environ.get("AQTC_PROFILING", "0") == "1"


def _aqtc_cuda_sync() -> None:
    """Synchronize CUDA device if profiling mode is enabled."""
    if _AQTC_PROFILING and torch.cuda.is_available():
        torch.cuda.synchronize()

# aqtc_models/llava/model -> aqtc_models
_AQTC_MODELS = Path(__file__).resolve().parents[2]
if str(_AQTC_MODELS) not in sys.path:
    sys.path.insert(0, str(_AQTC_MODELS))

from novelty import NoveltyEstimator
from scoring import ImportanceFusion
from quadrant_compress import plan_four_quadrant_groups, warn_if_retention_off_target


def _strategy_pool_visual(
    visual_embeds: torch.Tensor,
    importance: torch.Tensor,
    groups: list[list[int]],
    strategies: list[str],
) -> torch.Tensor:
    """Pool visual features: argmax uses importance scores; mean uses visual."""
    if not groups:
        return visual_embeds.new_empty((0, visual_embeds.shape[-1]))
    device = visual_embeds.device
    hidden = visual_embeds.shape[-1]
    result_parts: list[torch.Tensor] = []

    for group, strat in zip(groups, strategies):
        if not group:
            continue
        idx_t = torch.tensor(group, device=device, dtype=torch.long)
        if strat == "argmax":
            # Pick token with max importance in group
            scores = importance[idx_t]
            best = int(torch.argmax(scores).item())
            result_parts.append(visual_embeds[idx_t[best]].unsqueeze(0))
        else:
            result_parts.append(visual_embeds[idx_t].mean(dim=0, keepdim=True))

    return torch.cat(result_parts, dim=0) if result_parts else visual_embeds.new_empty((0, hidden))


def _mean_pool_groups(embeds: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    """Mean-pool variable-size index groups in one index_add."""
    if not groups:
        return embeds.new_empty((0, embeds.shape[-1]))
    device = embeds.device
    lengths = torch.tensor([len(g) for g in groups], device=device, dtype=embeds.dtype)
    flat = torch.tensor([idx for group in groups for idx in group], device=device, dtype=torch.long)
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=device, dtype=torch.long),
        torch.tensor([len(g) for g in groups], device=device, dtype=torch.long),
    )
    pooled = embeds.new_zeros(len(groups), embeds.shape[-1])
    pooled.index_add_(0, group_ids, embeds[flat])
    return pooled / lengths.clamp(min=1).unsqueeze(-1)


def build_video_token_coords(num_frames: int, height: int, width: int) -> list[tuple[int, int, int]]:
    """Row-major (t, h, w) coords matching flatten order of (T, H*W, D)."""
    t_grid, h_grid, w_grid = torch.meshgrid(
        torch.arange(num_frames, dtype=torch.long),
        torch.arange(height, dtype=torch.long),
        torch.arange(width, dtype=torch.long),
        indexing='ij'
    )
    t_flat = t_grid.flatten().tolist()
    h_flat = h_grid.flatten().tolist()
    w_flat = w_grid.flatten().tolist()
    return list(zip(t_flat, h_flat, w_flat))


def video_segment_sizes(
    num_frames: int,
    spatial_tokens: int,
    novelty_frames_per_segment: int,
) -> list[int]:
    """Token counts per novelty segment (frame-contiguous)."""
    return [
        (end - start) * spatial_tokens
        for start, end in video_segment_frame_ranges(
            num_frames, novelty_frames_per_segment
        )
    ]


def video_segment_frame_ranges(
    num_frames: int,
    novelty_frames_per_segment: int,
) -> list[tuple[int, int]]:
    """Return half-open frame ranges using actual frames per segment."""
    if novelty_frames_per_segment < 1:
        raise ValueError("novelty_frames_per_segment must be at least 1")
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    return [
        (start, min(start + novelty_frames_per_segment, num_frames))
        for start in range(0, num_frames, novelty_frames_per_segment)
    ]


def compress_pooled_video_tokens(
    video_feat: torch.Tensor,
    query_embeds: torch.Tensor,
    novelty_estimator: NoveltyEstimator,
    importance_fusion: ImportanceFusion,
    keep_ratio: float,
    saliency_scores: torch.Tensor,
    novelty_frames_per_segment: int = 4,
    # Dynamic segment budget parameters (simplified)
    dynamic_segment_budget: bool = False,
    dynamic_budget_strength: float = 0.5,
    budget_ceiling: float = 0.45,
    temporal_aware: bool = False,
    temporal_weight: float = 0.2,
    temporal_protect_ratio: float = 0.15,
) -> tuple[list[torch.Tensor], dict, dict]:
    """Compress pooled video tokens ``(T, H*W, D)`` -> per-frame token list.

    Returns:
        (frame_tokens, stats, timing_dict)

        ``frame_tokens`` is a list of ``num_frames`` tensors, each of shape
        ``(K_i, D)``.  The caller is responsible for inserting ``image_newline``
        separators between adjacent frames.
    """
    timing = {
        "query_s": 0.0,
        "fusion_s": 0.0,
        "novelty_s": 0.0,
        "plan_s": 0.0,
        "pool_s": 0.0,
        "update_s": 0.0,
    }

    if video_feat.ndim != 3:
        raise ValueError(f"Expected pooled video (T, HW, D), got shape {tuple(video_feat.shape)}")

    num_frames, spatial_tokens, hidden = video_feat.shape
    grid = int(round(math.sqrt(spatial_tokens)))
    if grid * grid != spatial_tokens:
        raise ValueError(
            f"Pooled spatial tokens must form a square grid, got HW={spatial_tokens}"
        )

    visual_embeds = video_feat.reshape(num_frames * spatial_tokens, hidden)
    num_tokens = visual_embeds.shape[0]
    coords = build_video_token_coords(num_frames, grid, grid)

    if saliency_scores is None:
        raise ValueError("saliency_scores are required (last-layer received-attention)")
    saliency_scores = saliency_scores.reshape(-1)
    if saliency_scores.shape[0] != num_tokens:
        raise ValueError(
            "Saliency scores must match visual token count: "
            f"got {tuple(saliency_scores.shape)}, expected {(num_tokens,)}"
        )
    saliency_scores = saliency_scores.to(device=visual_embeds.device, dtype=visual_embeds.dtype)

    # Query scoring
    _aqtc_cuda_sync()
    _t = time.perf_counter()
    if query_embeds is None or query_embeds.numel() == 0:
        query_scores = visual_embeds.new_zeros(num_tokens)
    else:
        query = query_embeds.float().mean(dim=0, keepdim=True)
        query_scores = F.cosine_similarity(visual_embeds.float(), query, dim=-1).to(visual_embeds.dtype)
    _aqtc_cuda_sync()
    timing["query_s"] = time.perf_counter() - _t

    # Importance fusion
    _aqtc_cuda_sync()
    _t = time.perf_counter()
    importance = importance_fusion(query_scores, saliency_scores)
    _aqtc_cuda_sync()
    timing["fusion_s"] = time.perf_counter() - _t

    segment_sizes = video_segment_sizes(num_frames, spatial_tokens, novelty_frames_per_segment)
    if sum(segment_sizes) != num_tokens:
        raise ValueError(
            f"Visual segment mismatch: segments={sum(segment_sizes)}, features={num_tokens}"
        )

    # Temporal scores are computed before allocation so that the global budget
    # can move from redundant frames to frames with genuine visual change.
    num_segments = len(segment_sizes)
    global_budget = max(1, int(round(num_tokens * keep_ratio)))
    budget_floor = keep_ratio * 0.5  # auto-derived soft floor
    precomputed_novelty: list[torch.Tensor] = []
    temporal_budget_enabled = bool(temporal_aware and dynamic_segment_budget)
    if temporal_aware:
        _aqtc_cuda_sync()
        _t_temporal = time.perf_counter()
        pre_offset = 0
        for ss in segment_sizes:
            pre_segment = visual_embeds[pre_offset : pre_offset + ss]
            pre_coords = coords[pre_offset : pre_offset + ss]
            pre_novelty = novelty_estimator.score(pre_segment, pre_coords)
            precomputed_novelty.append(pre_novelty)
            # Build a bounded provisional causal memory. Only representatives
            # that could plausibly survive the target budget are written.
            provisional_n = max(1, int(round(ss * keep_ratio)))
            provisional_idx = torch.topk(
                pre_novelty.float(), k=min(provisional_n, ss), sorted=False
            ).indices
            novelty_estimator.update_kept(
                pre_segment.index_select(0, provisional_idx),
                [pre_coords[int(i)] for i in provisional_idx.detach().cpu().tolist()],
            )
            pre_offset += ss
        _aqtc_cuda_sync()
        timing["novelty_s"] += time.perf_counter() - _t_temporal
        # The actual compression pass must update memory from final kept
        # representatives, rather than from provisional budget probes.
        novelty_estimator.reset()

    # ---- Dynamic segment budget allocation ----
    if dynamic_segment_budget and num_segments > 1:
        # Per-segment scoring. The legacy path remains importance-only.
        importance_cpu = importance.detach().cpu().float()
        seg_imp_means: list[float] = []
        offset_pre = 0
        for ss in segment_sizes:
            seg_imp = importance_cpu[offset_pre : offset_pre + ss]
            seg_imp_means.append(float(seg_imp.mean()))
            offset_pre += ss

        # Normalize to [0, 1]
        imp_min, imp_max = min(seg_imp_means), max(seg_imp_means)
        imp_range = max(imp_max - imp_min, 1e-8)
        imp_norm = [(v - imp_min) / imp_range for v in seg_imp_means]

        if temporal_budget_enabled and precomputed_novelty:
            nov_means = [float(n.float().mean().item()) for n in precomputed_novelty]
            nov_min, nov_max = min(nov_means), max(nov_means)
            nov_range = max(nov_max - nov_min, 1e-8)
            nov_norm = [(v - nov_min) / nov_range for v in nov_means]
            budget_scores = [
                imp_norm[i] + max(0.0, float(temporal_weight)) * nov_norm[i]
                for i in range(num_segments)
            ]
            score_min, score_max = min(budget_scores), max(budget_scores)
            score_range = max(score_max - score_min, 1e-8)
            budget_norm = [(v - score_min) / score_range for v in budget_scores]
        else:
            nov_means = [0.0] * num_segments
            nov_norm = [0.0] * num_segments
            budget_norm = imp_norm

        # Compute per-segment keep_ratios: score-driven interpolation
        #   scored_kr = budget_floor + score * (budget_ceiling - budget_floor)
        #   seg_kr = keep_ratio + strength * (scored_kr - keep_ratio)
        #   clamped to [budget_floor, budget_ceiling]
        seg_keep_ratios: list[float] = []
        for s in budget_norm:
            scored_kr = budget_floor + s * (budget_ceiling - budget_floor)
            seg_kr = keep_ratio + dynamic_budget_strength * (scored_kr - keep_ratio)
            seg_kr = max(budget_floor, min(budget_ceiling, seg_kr))
            seg_keep_ratios.append(seg_kr)

        # Allocate budgets with soft floors
        seg_allocated: list[int] = []
        total_alloc = 0
        for i, ss in enumerate(segment_sizes):
            n = int(round(ss * seg_keep_ratios[i]))
            n = max(1, min(ss, n))
            seg_allocated.append(n)
            total_alloc += n

        # Adjust to match global budget
        diff = global_budget - total_alloc
        ordered = sorted(range(num_segments), key=lambda i: budget_norm[i], reverse=True)
        if diff > 0:
            # Add to high-score segments (capped at ceiling)
            for idx in ordered:
                if diff <= 0:
                    break
                ciel = int(segment_sizes[idx] * budget_ceiling)
                add = min(diff, max(0, ciel - seg_allocated[idx]))
                if add > 0:
                    seg_allocated[idx] += add
                    diff -= add
        elif diff < 0:
            # Remove from low-score segments (capped at floor)
            diff = -diff
            for idx in reversed(ordered):
                if diff <= 0:
                    break
                floor = max(1, int(segment_sizes[idx] * budget_floor))
                rem = min(diff, max(0, seg_allocated[idx] - floor))
                if rem > 0:
                    seg_allocated[idx] -= rem
                    diff -= rem

        # Integer rounding plus floor/ceiling caps can leave a residual. Use
        # any remaining feasible capacity as a final exact-budget correction.
        residual = global_budget - sum(seg_allocated)
        if residual > 0:
            for idx in ordered:
                if residual <= 0:
                    break
                add = min(residual, segment_sizes[idx] - seg_allocated[idx])
                seg_allocated[idx] += add
                residual -= add
        elif residual < 0:
            residual = -residual
            for idx in reversed(ordered):
                if residual <= 0:
                    break
                rem = min(residual, max(0, seg_allocated[idx] - 1))
                seg_allocated[idx] -= rem
                residual -= rem

        # Build per-segment keep_ratios from allocated budgets
        seg_keep_ratios = [float(seg_allocated[i]) / float(segment_sizes[i]) for i in range(num_segments)]

        # Budget summary
        # actual_total = sum(seg_allocated)
        # all_in_range = all(budget_floor <= seg_keep_ratios[i] <= budget_ceiling for i in range(num_segments))
        # print(f"[AQTC budget] global_budget={global_budget} actual_total={actual_total} "
        #       f"base_ratio={keep_ratio:.3f} floor={budget_floor:.3f} ceiling={budget_ceiling:.3f} "
        #       f"strength={dynamic_budget_strength:.1f} bounds_ok={all_in_range}")
        # for i in range(num_segments):
        #     print(f"  seg_{i:02d}: score={imp_norm[i]:.3f}  "
        #           f"ratio={seg_keep_ratios[i]:.3f}  "
        #           f"allocated={seg_allocated[i]:4d}/{segment_sizes[i]}")
    else:
        seg_keep_ratios = [keep_ratio] * num_segments
        nov_means = [
            float(n.float().mean().item()) for n in precomputed_novelty
        ] if precomputed_novelty else [0.0] * num_segments
        nov_norm = [0.0] * num_segments
        budget_norm = [0.0] * num_segments

    # Per-frame accumulators: list of (Ki, D) tensors or list-of-tensors to be concatenated
    frame_pooled: list[list[torch.Tensor]] = [[] for _ in range(num_frames)]
    # Per-frame quadrant breakdown: {frame_idx: {"must_keep": N, "merge_cand": N, "explore": N, "pool_cand": N}}
    frame_quadrant_stats: list[dict[str, int]] = [
        {"must_keep": 0, "merge_cand": 0, "explore": 0, "pool_cand": 0}
        for _ in range(num_frames)
    ]
    # Per-segment stats: {seg_idx: {"original": N, "compressed": N}}
    segment_stats: list[dict[str, int]] = []

    novelty_parts = []
    offset = 0
    seg_frame_start = 0  # global frame index of the first frame in this segment
    for seg_idx, segment_size in enumerate(segment_sizes):
        seg_slice = slice(offset, offset + segment_size)
        segment = visual_embeds[seg_slice]
        seg_coords = coords[offset : offset + segment_size]
        seg_importance = importance[seg_slice]

        # Novelty scoring. Temporal mode reuses the pre-allocation scores;
        # legacy mode preserves the original causal memory behavior.
        if temporal_aware:
            seg_novelty = precomputed_novelty[seg_idx]
        else:
            _aqtc_cuda_sync()
            _t = time.perf_counter()
            seg_novelty = novelty_estimator.score(segment, seg_coords)
            _aqtc_cuda_sync()
            timing["novelty_s"] += time.perf_counter() - _t
        novelty_parts.append(seg_novelty)

        # Quadrant planning with per-segment keep_ratio
        seg_kr = seg_keep_ratios[seg_idx]
        _t = time.perf_counter()
        local_groups, _, debug_info, strategies = plan_four_quadrant_groups(
            seg_importance,
            seg_novelty,
            keep_ratio=seg_kr,
            frame_size=spatial_tokens,
            frame_tolerance=0.03,
        )

        # Preserve high-change representatives instead of averaging them away.
        # This is deliberately local and lightweight; it does not reproduce a
        # full temporal forest and leaves the planner unchanged by default.
        if temporal_aware and temporal_protect_ratio > 0 and local_groups:
            nov_threshold = torch.quantile(
                seg_novelty.float(),
                max(0.0, min(1.0, 1.0 - float(temporal_protect_ratio))),
            )
            protected = set(
                int(i) for i in torch.nonzero(seg_novelty >= nov_threshold, as_tuple=False).flatten().tolist()
            )
            for gi, group in enumerate(local_groups):
                if protected.intersection(group):
                    strategies[gi] = "argmax"
        timing["plan_s"] += time.perf_counter() - _t

        # ---- Accumulate per-frame quadrant classification ----
        seg_frames = segment_size // spatial_tokens
        mk_cpu = debug_info["must_keep"].cpu()
        mc_cpu = debug_info["merge_cand"].cpu()
        ex_cpu = debug_info["explore"].cpu()
        pc_cpu = debug_info["pool_cand"].cpu()
        for f in range(seg_frames):
            global_f = seg_frame_start + f
            f_slice = slice(f * spatial_tokens, (f + 1) * spatial_tokens)
            frame_quadrant_stats[global_f]["must_keep"] += int(mk_cpu[f_slice].sum().item())
            frame_quadrant_stats[global_f]["merge_cand"] += int(mc_cpu[f_slice].sum().item())
            frame_quadrant_stats[global_f]["explore"] += int(ex_cpu[f_slice].sum().item())
            frame_quadrant_stats[global_f]["pool_cand"] += int(pc_cpu[f_slice].sum().item())

        # Segment-level stats (only after local_groups processing is complete)
        seg_compressed = len(local_groups)
        seg_original = segment_size
        segment_stats.append({"original": seg_original, "compressed": seg_compressed, "ratio": seg_kr})

        if local_groups:
            _aqtc_cuda_sync()
            _t = time.perf_counter()
            pooled_seg = _strategy_pool_visual(
                segment, seg_importance, local_groups, strategies
            )
            _aqtc_cuda_sync()
            timing["pool_s"] += time.perf_counter() - _t

            # Distribute pooled tokens to per-frame accumulators
            for g_idx, local_group in enumerate(local_groups):
                seg_frame = local_group[0] // spatial_tokens  # 0..3
                global_frame = seg_frame_start + seg_frame
                frame_pooled[global_frame].append(pooled_seg[g_idx : g_idx + 1])

            # ---- Update novelty estimator (mean pool, consistent with stored memory) ----
            _t = time.perf_counter()
            novelty_pooled = _mean_pool_groups(segment, local_groups)
            novelty_estimator.update_kept(
                novelty_pooled,
                [seg_coords[group[0]] for group in local_groups],
            )
            timing["update_s"] += time.perf_counter() - _t

        offset += segment_size
        seg_frame_start += segment_size // spatial_tokens

    # Concatenate per-frame pooled tensors
    frame_tensors: list[torch.Tensor] = []
    total_compressed = 0
    for ft_list in frame_pooled:
        if ft_list:
            ft = torch.cat(ft_list, dim=0)
        else:
            ft = visual_embeds.new_empty((0, hidden))
        frame_tensors.append(ft)
        total_compressed += ft.shape[0]

    # ---- Print compression statistics summary ----
    # print(f"[AQTC stats] total={num_tokens} compressed={total_compressed} "
    #       f"ratio={total_compressed/num_tokens:.3f} frames={num_frames} segments={len(segment_stats)}")
    # Per-segment summary
    # for si, ss in enumerate(segment_stats):
    #     print(f"  seg_{si:02d}: {ss['original']:4d} -> {ss['compressed']:4d} "
    #           f"({ss['compressed']/ss['original']:.3f})")
    # Per-frame quadrant breakdown
    # print(f"  frame quadrant breakdown (must_keep/merge_cand/explore/pool_cand):")
    # for fi in range(num_frames):
    #     fs = frame_quadrant_stats[fi]
    #     ft_compressed = frame_tensors[fi].shape[0] if fi < len(frame_tensors) else 0
        # print(f"    f_{fi:02d}: total={spatial_tokens:3d} -> {ft_compressed:3d}  "
        #       f"mk={fs['must_keep']:3d} mc={fs['merge_cand']:3d} "
        #       f"ex={fs['explore']:3d} pc={fs['pool_cand']:3d}")

    novelty_scores = (
        torch.cat(novelty_parts, dim=0) if novelty_parts else visual_embeds.new_empty((0,))
    )

    warn_if_retention_off_target(
        retained=total_compressed,
        total=num_tokens,
        keep_ratio=keep_ratio,
        tol=0.03,
        context="LLaVA-AQTC video compress",
    )
    stats = {
        "original_visual": num_tokens,
        "compressed_visual": total_compressed,
        "num_frames": num_frames,
        "grid": grid,
        "query": query_scores.detach(),
        "saliency": saliency_scores.detach(),
        "importance": importance.detach(),
        "novelty": novelty_scores.detach(),
        "segment_stats": segment_stats,
        "temporal_aware": bool(temporal_aware),
        "budget_scores": budget_norm,
    }
    return frame_tensors, stats, timing
