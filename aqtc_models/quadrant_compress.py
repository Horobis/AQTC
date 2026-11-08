"""Four-quadrant visual token compression with keep_ratio as a retention target."""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch


def _percentile_threshold(scores: torch.Tensor, keep_top_fraction: float = 0.4) -> torch.Tensor:
    """Tokens with score >= threshold are treated as 'high' (top fraction)."""
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    if scores.numel() == 1:
        return scores[0]
    q = max(0.0, min(1.0, 1.0 - keep_top_fraction))
    return torch.quantile(scores.float(), q).to(scores.dtype)


def _contiguous_runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    indices = sorted(indices)
    runs = [[indices[0]]]
    for idx in indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def _is_contiguous(group: list[int]) -> bool:
    if not group:
        return True
    g = sorted(group)
    return g[-1] - g[0] + 1 == len(g) and g == list(range(g[0], g[-1] + 1))


def _force_contiguous_groups(
    groups: list[list[int]],
    priorities: list[int],
) -> tuple[list[list[int]], list[int]]:
    """Split any non-contiguous group into contiguous runs (same priority)."""
    out_groups: list[list[int]] = []
    out_prios: list[int] = []
    for group, prio in zip(groups, priorities):
        for run in _contiguous_runs(group):
            out_groups.append(run)
            out_prios.append(prio)
    items = sorted(
        [(g[0], g, p) for g, p in zip(out_groups, out_prios)],
        key=lambda x: x[0],
    )
    return [g for _, g, _ in items], [p for _, _, p in items]


def _group_mean_importance(importance: np.ndarray, group: list[int]) -> float:
    if not group:
        return 0.0
    return float(np.mean(importance[group]))


def _as_numpy_1d(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=np.float64)
    return values.detach().float().cpu().numpy().astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
#  Spatial helpers for low-I + high-N explore merge (grid-aware adjacency)
# ---------------------------------------------------------------------------

def _spatial_4neighbors(flat_idx: int, grid: int, frame_size: int) -> list[int]:
    """Return list of valid 4-neighbor flat indices (up/down/left/right)
    within the same frame for a 2-D ``grid × grid`` spatial layout.

    ``flat_idx`` is a segment-local index; ``frame_size = grid * grid``.
    """
    local = flat_idx % frame_size
    row, col = divmod(local, grid)
    neighbors: list[int] = []
    base = flat_idx - local
    if row > 0:
        neighbors.append(base + (row - 1) * grid + col)
    if row < grid - 1:
        neighbors.append(base + (row + 1) * grid + col)
    if col > 0:
        neighbors.append(base + row * grid + (col - 1))
    if col < grid - 1:
        neighbors.append(base + row * grid + (col + 1))
    return neighbors


def _compute_novelty_threshold(novelty_np: np.ndarray) -> float:
    """Adaptive novelty-similarity threshold:
    threshold = 0.10 * (P90 - P10), with a floor epsilon.
    """
    if novelty_np.size < 2:
        return 1e-8
    p10 = float(np.percentile(novelty_np, 10))
    p90 = float(np.percentile(novelty_np, 90))
    rng = p90 - p10
    thr = 0.10 * rng
    return max(thr, 1e-8)


def _merge_explore_spatially(
    explore_indices: list[int],
    importance_np: np.ndarray,
    novelty_np: np.ndarray,
    grid: int,
    frame_size: int,
    max_window: int = 2,
) -> tuple[list[list[int]], list[float], list[float]]:
    """Greedy spatial-adjacency merge for low-I + high-N tokens.

    Within the same frame and 4-neighbor adjacency, two tokens are merged
    when ``abs(novelty[i] - novelty[j]) <= threshold`` (adaptive).
    Groups are capped at ``max_window`` tokens; mean pooling is used.

    Returns:
        (groups, group_importance, group_novelty)
    """
    if not explore_indices:
        return [], [], []

    threshold = _compute_novelty_threshold(novelty_np)
    ungrouped = set(explore_indices)
    # Build lookup: index -> set of ungrouped neighbors
    neighbor_of: dict[int, set[int]] = {}
    for idx in explore_indices:
        nb = {n for n in _spatial_4neighbors(idx, grid, frame_size) if n in ungrouped}
        if nb:
            neighbor_of[idx] = nb

    groups: list[list[int]] = []
    group_imp: list[float] = []
    group_nov: list[float] = []

    # Process in descending novelty order (more novel tokens drive the merge)
    sorted_explore = sorted(explore_indices, key=lambda i: float(novelty_np[i]), reverse=True)
    for anchor in sorted_explore:
        if anchor not in ungrouped:
            continue
        # Find best neighbor: novel-close AND highest importance
        candidates = neighbor_of.get(anchor, set()) & ungrouped
        valid = [n for n in candidates if abs(float(novelty_np[anchor]) - float(novelty_np[n])) <= threshold]
        if not valid or max_window < 2:
            # Singleton
            groups.append([anchor])
            group_imp.append(float(importance_np[anchor]))
            group_nov.append(float(novelty_np[anchor]))
            ungrouped.discard(anchor)
            continue

        # Pick the valid neighbor with highest importance (tie-break: smaller index)
        best_nb = max(valid, key=lambda n: (float(importance_np[n]), -n))
        # Ensure group is sorted by index (important for downstream merge logic)
        group = sorted([anchor, best_nb])
        groups.append(group)
        group_imp.append(float(np.mean(importance_np[group])))
        group_nov.append(float(np.mean(novelty_np[group])))
        ungrouped.discard(anchor)
        ungrouped.discard(best_nb)

    # Sort groups by first index
    ordered = sorted(zip(groups, group_imp, group_nov), key=lambda x: x[0][0])
    return (
        [g for g, _, _ in ordered],
        [gi for _, gi, _ in ordered],
        [gn for _, _, gn in ordered],
    )


def _adjust_groups_to_target(
    groups: list[list[int]],
    priorities: list[int],
    importance: torch.Tensor,
    target: int,
) -> tuple[list[list[int]], list[int]]:
    """Shrink/expand group count toward ``target`` without dropping tokens.

    Shrink: merge two temporally-adjacent lower-priority groups.
    Expand: peel one **end** token from a multi-token lower-priority group
    (never from the middle) so every group stays index-contiguous.
    Priority 0 (must-keep) is protected unless no alternative exists.
    """
    if target < 1:
        target = 1

    importance_np = _as_numpy_1d(importance)
    groups, priorities = _force_contiguous_groups(groups, priorities)

    # Filter empty groups
    items = [(g[0], list(g), int(p)) for g, p in zip(groups, priorities) if g]
    if not items:
        return [], []
    items.sort(key=lambda x: x[0])

    # ---- Build numpy arrays for merge ----
    N = len(items)
    starts = np.array([it[0] for it in items], dtype=np.int32)
    ends = np.array([it[1][-1] for it in items], dtype=np.int32)
    prios = np.array([it[2] for it in items], dtype=np.int8)

    # ---- Shrink: numpy batch merge ----
    while N > target:
        if N < 2:
            break
        left_idx = np.arange(N - 1)
        right_idx = left_idx + 1
        adjacent = ends[left_idx] + 1 == starts[right_idx]
        if not adjacent.any():
            break

        li = left_idx[adjacent]
        ri = right_idx[adjacent]
        lp = prios[li].astype(np.int32)
        rp = prios[ri].astype(np.int32)

        both_zero = (lp == 0) & (rp == 0)
        one_zero = ((lp == 0) | (rp == 0)) & ~both_zero
        penalty = np.where(both_zero, 1000.0, np.where(one_zero, 100.0, 0.0))

        # Compute merged importance via direct numpy slicing
        len_li = ends[li] - starts[li] + 1
        len_ri = ends[ri] - starts[ri] + 1
        merged_len = (len_li + len_ri).astype(np.float64)
        sum_li = np.array([float(np.sum(importance_np[starts[li[j]] : ends[li[j]] + 1]))
                           for j in range(len(li))], dtype=np.float64)
        sum_ri = np.array([float(np.sum(importance_np[starts[ri[j]] : ends[ri[j]] + 1]))
                           for j in range(len(ri))], dtype=np.float64)
        merged_imp = (sum_li + sum_ri) / merged_len
        scores = penalty + merged_imp + 0.01 * np.maximum(lp, rp).astype(np.float64)

        best = int(np.argmin(scores))
        li_best, ri_best = int(li[best]), int(ri[best])

        # Merge ri_best into li_best
        ends[li_best] = ends[ri_best]
        prios[li_best] = max(prios[li_best], prios[ri_best])
        # Delete ri_best from arrays
        mask = np.ones(N, dtype=bool)
        mask[ri_best] = False
        starts = starts[mask]
        ends = ends[mask]
        prios = prios[mask]
        N -= 1

    # ---- Build records from arrays ----
    records = [
        {
            "group": list(range(int(starts[i]), int(ends[i]) + 1)),
            "priority": int(prios[i]),
        }
        for i in range(N)
    ]

    # ---- Expand: peel from ends ----
    while len(records) < target:
        records.sort(key=lambda r: r["group"][0])

        splittable = [(i, r) for i, r in enumerate(records) if len(r["group"]) > 1]
        if not splittable:
            break

        # Try non-prio-0 first
        candidates = [(i, r) for i, r in splittable if r["priority"] != 0]
        penalty_extra = 0.0
        if not candidates:
            candidates = [(i, r) for i, r in splittable]
            penalty_extra = 10.0

        best_val = None  # (score, i, token)
        for i, r in candidates:
            g = r["group"]
            left_t, right_t = g[0], g[-1]
            end_token = right_t if float(importance_np[right_t]) > float(importance_np[left_t]) else left_t
            split_score = -float(importance_np[end_token]) + 0.1 * r["priority"] + penalty_extra
            if best_val is None or split_score < best_val[0]:
                best_val = (split_score, i, end_token)

        if best_val is None:
            break
        _, i, token = best_val
        r = records[i]
        group = r["group"]
        prio = r["priority"]
        if token == group[0]:
            remain = group[1:]
        elif token == group[-1]:
            remain = group[:-1]
        else:
            remain = [t for t in group if t != token]

        peeled = {"group": [token], "priority": 2}
        remaining = {"group": remain, "priority": prio}
        replacement = [peeled, remaining] if token == group[0] else [remaining, peeled]
        records = records[:i] + replacement + records[i + 1:]

    # ---- Final contiguity check ----
    groups_out = [r["group"] for r in records]
    prios_out = [r["priority"] for r in records]
    groups_out, prios_out = _force_contiguous_groups(groups_out, prios_out)

    # ---- Defensive shrink if contiguity inflated count ----
    while len(groups_out) > target:
        items2 = [(g[0], g, p) for g, p in zip(groups_out, prios_out) if g]
        items2.sort(key=lambda x: x[0])
        N2 = len(items2)
        if N2 < 2:
            break

        starts2 = np.array([it[0] for it in items2], dtype=np.int32)
        ends2 = np.array([it[1][-1] for it in items2], dtype=np.int32)
        prios2 = np.array([it[2] for it in items2], dtype=np.int8)

        left_idx = np.arange(N2 - 1)
        right_idx = left_idx + 1
        adjacent = ends2[left_idx] + 1 == starts2[right_idx]
        if not adjacent.any():
            break

        li = left_idx[adjacent]
        ri = right_idx[adjacent]
        lp = prios2[li].astype(np.int32)
        rp = prios2[ri].astype(np.int32)

        both_zero = (lp == 0) & (rp == 0)
        one_zero = ((lp == 0) | (rp == 0)) & ~both_zero
        penalty = np.where(both_zero, 1000.0, np.where(one_zero, 100.0, 0.0))

        # Compute merged importance inline with prefix sums
        len_li = ends2[li] - starts2[li] + 1
        len_ri = ends2[ri] - starts2[ri] + 1
        merged_len = (len_li + len_ri).astype(np.float64)
        sum_li = np.array([float(np.sum(importance_np[starts2[li[j]] : ends2[li[j]] + 1]))
                           for j in range(len(li))], dtype=np.float64)
        sum_ri = np.array([float(np.sum(importance_np[starts2[ri[j]] : ends2[ri[j]] + 1]))
                           for j in range(len(ri))], dtype=np.float64)
        merged_imp = (sum_li + sum_ri) / merged_len
        scores = penalty + merged_imp + 0.01 * np.maximum(lp, rp).astype(np.float64)

        best = int(np.argmin(scores))
        li_best, ri_best = int(li[best]), int(ri[best])

        merged = items2[li_best][1] + items2[ri_best][1]
        merged_prio = max(items2[li_best][2], items2[ri_best][2])
        new_items = items2[:li_best] + [(merged[0], merged, merged_prio)] + items2[ri_best + 1:]
        groups_out = [g for _, g, _ in new_items]
        prios_out = [p for _, _, p in new_items]

    items_final = [(g[0], g, p) for g, p in zip(groups_out, prios_out) if g]
    items_final.sort(key=lambda x: x[0])
    groups_out = [g for _, g, _ in items_final]
    prios_out = [p for _, _, p in items_final]
    for group in groups_out:
        if not _is_contiguous(group):
            raise RuntimeError(f"Non-contiguous compression group after adjust: {group}")
    return groups_out, prios_out


def _frame_local_runs(indices: list[int], frame_size: int | None) -> list[list[int]]:
    """Return contiguous runs, optionally split at flattened frame boundaries."""
    runs = _contiguous_runs(indices)
    if frame_size is None:
        return runs
    if frame_size < 1:
        raise ValueError("frame_size must be positive")

    out: list[list[int]] = []
    for run in runs:
        start = 0
        while start < len(run):
            frame_end = ((run[start] // frame_size) + 1) * frame_size
            end = start
            while end < len(run) and run[end] < frame_end:
                end += 1
            out.append(run[start:end])
            start = end
    return out


def _frame_local_runs_with_scores(
    indices: list[int],
    priority: int,
    importance: np.ndarray,
    novelty: np.ndarray,
    frame_size: int,
) -> tuple[list[list[int]], list[int], list[float], list[float]]:
    """Split parent runs at frame boundaries while inheriting parent scores."""
    out_groups: list[list[int]] = []
    out_priorities: list[int] = []
    out_importance: list[float] = []
    out_novelty: list[float] = []
    for parent in _contiguous_runs(indices):
        parent_importance = _group_mean_importance(importance, parent)
        parent_novelty = _group_mean_importance(novelty, parent)
        for child in _frame_local_runs(parent, frame_size):
            out_groups.append(child)
            out_priorities.append(priority)
            out_importance.append(parent_importance)
            out_novelty.append(parent_novelty)
    return out_groups, out_priorities, out_importance, out_novelty


def _adjust_frame_local_groups_to_target(
    groups: list[list[int]],
    priorities: list[int],
    importance: torch.Tensor,
    novelty: torch.Tensor,
    target: int,
    frame_size: int,
    group_importance: list[float] | None = None,
    group_novelty: list[float] | None = None,
) -> tuple[list[list[int]], list[int], list[float], list[float]]:
    """Adjust frame-local groups while retaining split/merge ranking metadata.

    Optimized: numpy batch merge replaces the original per-pair heap, which
    was the dominant bottleneck.  Expand & prune rewritten in the same style.
    """
    if frame_size < 1:
        raise ValueError("frame_size must be positive")
    target = max(1, target)
    importance_np = _as_numpy_1d(importance)
    novelty_np = _as_numpy_1d(novelty)

    imp_prefix = np.concatenate(([0.0], np.cumsum(importance_np, dtype=np.float64)))
    nov_prefix = np.concatenate(([0.0], np.cumsum(novelty_np, dtype=np.float64)))

    def _prefix_sum(prefix: np.ndarray, start: int, end: int) -> float:
        return float(prefix[end + 1] - prefix[start])

    if group_importance is None:
        group_importance = [_group_mean_importance(importance_np, group) for group in groups]
    if group_novelty is None:
        group_novelty = [_group_mean_importance(novelty_np, group) for group in groups]
    if len(group_importance) != len(groups) or len(group_novelty) != len(groups):
        raise ValueError("group score metadata length mismatch")

    # Filter & sort non-empty groups
    filtered = [(list(g), int(p), float(gi), float(gn))
                for g, p, gi, gn in zip(groups, priorities, group_importance, group_novelty)
                if g]
    if not filtered:
        return [], [], [], []
    filtered.sort(key=lambda x: x[0][0])

    # ---- Build numpy arrays ----
    N = len(filtered)
    starts = np.array([x[0][0] for x in filtered], dtype=np.int32)
    ends = np.array([x[0][-1] for x in filtered], dtype=np.int32)
    prios = np.array([x[1] for x in filtered], dtype=np.int8)
    # Pre-compute imp_sum / nov_sum via prefix arrays (vectorised)
    imp_sums = np.array([_prefix_sum(imp_prefix, int(s), int(e)) for s, e in zip(starts, ends)], dtype=np.float64)
    nov_sums = np.array([_prefix_sum(nov_prefix, int(s), int(e)) for s, e in zip(starts, ends)], dtype=np.float64)

    alive = np.ones(N, dtype=bool)
    active = N

    _t_merge_total = 0.0
    _merge_iterations = 0

    # ================================================================
    #  Phase 1: Shrink --- numpy batch merge (was per-pair heap)
    # ================================================================
    while active > target:
        _t_iter = time.perf_counter()
        alive_idx = np.where(alive)[0]
        if len(alive_idx) < 2:
            break

        left_idx = alive_idx[:-1]
        right_idx = alive_idx[1:]

        # Valid pairs: index-adjacent AND same frame
        adjacent = ends[left_idx] + 1 == starts[right_idx]
        same_frame = starts[left_idx] // frame_size == starts[right_idx] // frame_size
        valid = adjacent & same_frame

        if not valid.any():
            break

        li = left_idx[valid]
        ri = right_idx[valid]
        
        # Filter out pairs with invalid ranges (defensive)
        len_li_arr = ends[li] - starts[li] + 1
        len_ri_arr = ends[ri] - starts[ri] + 1
        valid_range = (len_li_arr > 0) & (len_ri_arr > 0)
        if not valid_range.any():
            break
        li = li[valid_range]
        ri = ri[valid_range]
        
        lp = prios[li].astype(np.int32)
        rp = prios[ri].astype(np.int32)

        # protect_penalty: (0,0)=1000  (0,x)=100  otherwise=0
        both_zero = (lp == 0) & (rp == 0)
        one_zero = ((lp == 0) | (rp == 0)) & ~both_zero
        penalty = np.where(both_zero, 1000.0, np.where(one_zero, 100.0, 0.0))

        # Merged importance = (sum_li + sum_ri) / (len_li + len_ri)
        len_li = ends[li] - starts[li] + 1
        len_ri = ends[ri] - starts[ri] + 1
        merged_len = (len_li + len_ri).astype(np.float64)
        merged_imp = (imp_sums[li] + imp_sums[ri]) / merged_len

        scores = penalty + merged_imp + 0.01 * np.maximum(lp, rp).astype(np.float64)

        # Tie-breaking: np.argmin → smallest idx = smallest start (matches old heap)
        best = int(np.argmin(scores))
        li_b, ri_b = int(li[best]), int(ri[best])

        # Merge ri_b into li_b
        ends[li_b] = ends[ri_b]
        prios[li_b] = max(prios[li_b], prios[ri_b])
        imp_sums[li_b] += imp_sums[ri_b]
        nov_sums[li_b] += nov_sums[ri_b]
        alive[ri_b] = False
        active -= 1
        _t_merge_total += time.perf_counter() - _t_iter
        _merge_iterations += 1

    # ---- Build dict records from alive nodes ----
    alive_idx = np.where(alive)[0]
    records: list[dict] = []
    for i in alive_idx:
        s, e = int(starts[i]), int(ends[i])
        if e < s:
            # Skip corrupted nodes (should not happen, but defensive)
            continue
        length = e - s + 1
        records.append({
            "group": list(range(s, e + 1)),
            "priority": int(prios[i]),
            "importance": float(imp_sums[i]) / float(length),
            "novelty": float(nov_sums[i]) / float(length),
        })
    records.sort(key=lambda r: r["group"][0])

    # ================================================================
    #  Phase 2: Expand --- numpy batch scoring + Python insertion
    # ================================================================
    _t_expand_total = 0.0
    _expand_iterations = 0
    while len(records) < target:
        _t_exp_iter = time.perf_counter()
        records.sort(key=lambda r: r["group"][0])
        n_rec = len(records)

        rec_starts = np.array([r["group"][0] for r in records], dtype=np.int32)
        rec_ends = np.array([r["group"][-1] for r in records], dtype=np.int32)
        rec_prios = np.array([r["priority"] for r in records], dtype=np.int8)
        rec_lens = rec_ends - rec_starts + 1

        can_split = np.where(rec_lens > 1)[0]
        if len(can_split) == 0:
            break

        # Exclude priority 0 on first attempt
        non_zero = rec_prios[can_split] != 0
        if non_zero.any():
            split_idx = can_split[non_zero]
            penalty_extra = 0.0
        else:
            split_idx = can_split
            penalty_extra = 10.0

        left_imp = importance_np[rec_starts[split_idx]]
        right_imp = importance_np[rec_ends[split_idx]]
        use_right = right_imp > left_imp
        end_imp = np.where(use_right, right_imp, left_imp)
        split_scores = -end_imp + 0.1 * rec_prios[split_idx].astype(np.float64) + penalty_extra

        best = int(np.argmin(split_scores))
        best_rec_i = int(split_idx[best])
        rec = records[best_rec_i]
        group = rec["group"]
        token = int(rec_ends[best_rec_i]) if use_right[best] else int(rec_starts[best_rec_i])

        if token == group[0]:
            remain = group[1:]
        elif token == group[-1]:
            remain = group[:-1]
        else:
            remain = [t for t in group if t != token]

        peeled = {
            "group": [token],
            "priority": rec["priority"],
            "importance": rec["importance"],
            "novelty": rec["novelty"],
        }
        remaining = {
            "group": remain,
            "priority": rec["priority"],
            "importance": rec["importance"],
            "novelty": rec["novelty"],
        }
        replacement = [peeled, remaining] if token == group[0] else [remaining, peeled]
        records = records[:best_rec_i] + replacement + records[best_rec_i + 1:]
        _t_expand_total += time.perf_counter() - _t_exp_iter
        _expand_iterations += 1

    records.sort(key=lambda r: r["group"][0])
    # print(
    #     f"[AQTC adjust detail] N={N}  target={target}  "
    #     f"merge={_t_merge_total:.4f}s({_merge_iterations}ops)  "
    #     f"expand={_t_expand_total:.4f}s({_expand_iterations}ops)  "
    #     f"final={len(records)}groups"
    # )
    return (
        [r["group"] for r in records],
        [r["priority"] for r in records],
        [r["importance"] for r in records],
        [r["novelty"] for r in records],
    )


def _prune_frame_local_groups(
    groups: list[list[int]],
    priorities: list[int],
    group_importance: list[float],
    group_novelty: list[float],
    target: int,
    frame_size: int,
    tolerance: float,
) -> tuple[list[list[int]], list[int]]:
    """Drop low-priority groups only when frame-local splitting overshoots budget.

    Optimized: pre-sort by drop priority once, then two-pass selection
    (unprotected → fallback) instead of the original per-iteration scan.
    """
    upper = target + int(math.floor(target * tolerance))
    if len(groups) <= upper:
        return groups, priorities

    n = len(groups)
    # Build records: (group, priority, importance, novelty, frame)
    records = [(list(g), int(p), float(gi), float(gn), g[0] // frame_size)
               for g, p, gi, gn in zip(groups, priorities, group_importance, group_novelty)]

    # ---- Drop-key: identical to original min() key ----
    # Priority 3→2→1→0; within quadrant: the signal that defines it
    def _drop_key_tuple(rec):
        g, p, gi, gn, _frame = rec
        if p == 0 or p == 3:
            secondary = gi + gn
        elif p == 1:
            secondary = gi
        else:  # p == 2
            secondary = gn
        return (-p, secondary, gi, gn, g[0])

    # Sort indices by drop priority (ascending → worst first)
    order = sorted(range(n), key=lambda i: _drop_key_tuple(records[i]))

    # ---- Frame counts (track live groups per frame) ----
    frame_counts: dict[int, int] = {}
    for rec in records:
        fc = rec[4]
        frame_counts[fc] = frame_counts.get(fc, 0) + 1

    keep = [True] * n
    drops_needed = n - upper

    # ---- Pass 1: drop unprotected groups (never drop must_keep, priority=0) ----
    for idx in order:
        if drops_needed <= 0:
            break
        g, p, _gi, _gn, frame = records[idx]
        # Hard protection: never drop must_keep groups
        if p == 0:
            continue
        # Protected when the frame has exactly 1 group AND we can cover all frames
        if len(frame_counts) <= upper and frame_counts.get(frame, 0) == 1:
            continue
        keep[idx] = False
        frame_counts[frame] = frame_counts.get(frame, 0) - 1
        if frame_counts[frame] == 0:
            del frame_counts[frame]
        drops_needed -= 1

    # ---- Pass 2: fallback – drop must_keep only after all others exhausted ----
    if drops_needed > 0:
        for idx in order:
            if drops_needed <= 0:
                break
            if not keep[idx]:
                continue
            if records[idx][1] == 0:
                continue  # still protect must_keep in pass 2
            keep[idx] = False
            drops_needed -= 1

    # ---- Reconstruct sorted by position ----
    result = [(records[i][0], records[i][1]) for i in range(n) if keep[i]]
    result.sort(key=lambda x: x[0][0])
    return [r[0] for r in result], [r[1] for r in result]


def merge_adjacent_groups_to_target(
    groups: list[list[int]],
    target: int,
    is_visual_group: list[bool] | None = None,
) -> list[list[int]]:
    """Merge adjacent visual groups until ``len(groups) <= target`` (text untouched).

    Used as writeback fallback when orphan recovery inflates group count.
    """
    if target < 1:
        target = 1
    if len(groups) <= target:
        return groups
    if is_visual_group is None:
        is_visual_group = [True] * len(groups)

    groups = [list(g) for g in groups]
    flags = list(is_visual_group)
    while len(groups) > target:
        best = None  # (size, i)
        for i in range(len(groups) - 1):
            if not flags[i] or not flags[i + 1]:
                continue
            if groups[i][-1] + 1 != groups[i + 1][0]:
                continue
            size = len(groups[i]) + len(groups[i + 1])
            if best is None or size < best[0]:
                best = (size, i)
        if best is None:
            break
        i = best[1]
        groups[i] = groups[i] + groups[i + 1]
        del groups[i + 1]
        del flags[i + 1]
    return groups


def warn_if_retention_off_target(
    retained: int,
    total: int,
    keep_ratio: float,
    tol: float = 0.05,
    context: str = "AQTC",
) -> None:
    """Warn when empirical retention drifts more than ``tol`` from ``keep_ratio``."""
    if total <= 0:
        return
    retain = retained / float(total)
    if abs(retain - keep_ratio) > tol:
        warnings.warn(
            f"{context}: visual retention {retain:.4f} ({retained}/{total}) "
            f"deviates from keep_ratio={keep_ratio:.4f} by >{tol:.2f}",
            stacklevel=2,
        )


def plan_four_quadrant_groups(
    importance: torch.Tensor,
    novelty: torch.Tensor,
    keep_ratio: float,
    high_fraction: float = 0.4,
    frame_size: int | None = None,
    frame_tolerance: float = 0.03,
) -> tuple[list[list[int]], torch.Tensor, dict[str, torch.Tensor], list[str]]:
    """Plan compression groups over one segment (local indices 0..N-1).

    Quadrants (segment-wise percentiles):
      high I + high n -> singleton keep
      high I + low  n -> contiguous mean-merge
      low  I + high n -> singleton keep (ranked by novelty when shrinking)
      low  I + low  n -> contiguous mean-pool (hard-delete only in frame mode
      when the frame-boundary constraint exceeds the retention tolerance)

    ``keep_ratio`` is a **target retention rate**: the number of output groups is
    adjusted to approximately ``round(N * keep_ratio)``. When ``frame_size`` is
    provided, groups cannot cross flattened frame boundaries. The planner first
    adjusts within frames and then drops low-priority groups if the boundary
    constraint leaves more than ``frame_tolerance`` groups above target.
    """
    num_tokens = importance.shape[0]
    if novelty.shape[0] != num_tokens:
        raise ValueError("importance/novelty length mismatch")
    if num_tokens == 0:
        return [], importance.new_zeros((0,), dtype=torch.bool), {}, []
    if frame_size is not None and frame_size < 1:
        raise ValueError("frame_size must be positive")
    if frame_tolerance < 0:
        raise ValueError("frame_tolerance must be non-negative")

    # ---- Phase A: GPU operations (transfer + quantile + classification) ----
    _t_a0 = time.perf_counter()
    importance_cpu = importance.detach().cpu()
    novelty_cpu = novelty.detach().cpu()

    target = max(1, int(round(num_tokens * keep_ratio)))
    target = min(target, num_tokens)

    i_thr = _percentile_threshold(importance, high_fraction)
    n_thr = _percentile_threshold(novelty, high_fraction)
    high_i = importance >= i_thr
    high_n = novelty >= n_thr

    must_keep = high_i & high_n
    merge_cand = high_i & ~high_n
    explore = ~high_i & high_n
    pool_cand = ~high_i & ~high_n

    imp_np = _as_numpy_1d(importance_cpu)
    nov_np = _as_numpy_1d(novelty_cpu)
    _t_a1 = time.perf_counter()

    # ---- Phase B: group building (Python loops) ----
    used = np.zeros(num_tokens, dtype=bool)
    groups: list[list[int]] = []
    priorities: list[int] = []
    initial_group_importance: list[float] = []
    initial_group_novelty: list[float] = []

    # 1) must-keep singletons
    for idx in must_keep.cpu().nonzero(as_tuple=False).flatten().tolist():
        groups.append([idx])
        priorities.append(0)
        if frame_size is not None:
            initial_group_importance.append(float(imp_np[idx]))
            initial_group_novelty.append(float(nov_np[idx]))
        used[idx] = True

    # 2) high-I low-n contiguous merges
    merge_indices = [
        idx for idx in merge_cand.cpu().nonzero(as_tuple=False).flatten().tolist() if not used[idx]
    ]
    if frame_size is None:
        merge_runs = _contiguous_runs(merge_indices)
        for run in merge_runs:
            groups.append(run)
            priorities.append(1)
            for idx in run:
                used[idx] = True
    else:
        split_groups, split_priorities, split_importance, split_novelty = _frame_local_runs_with_scores(
            merge_indices,
            1,
            imp_np,
            nov_np,
            frame_size,
        )
        groups.extend(split_groups)
        priorities.extend(split_priorities)
        initial_group_importance.extend(split_importance)
        initial_group_novelty.extend(split_novelty)
        for idx in merge_indices:
            used[idx] = True

    # 3) low-I high-n explore: spatial-adjacency + novelty-threshold merge
    explore_indices = [
        idx for idx in explore.cpu().nonzero(as_tuple=False).flatten().tolist() if not used[idx]
    ]
    if frame_size is not None:
        grid = int(round(math.sqrt(frame_size)))
        explore_groups, explore_imp, explore_nov = _merge_explore_spatially(
            explore_indices,
            imp_np,
            nov_np,
            grid=grid,
            frame_size=frame_size,
            max_window=2,
        )
        groups.extend(explore_groups)
        priorities.extend([2] * len(explore_groups))
        initial_group_importance.extend(explore_imp)
        initial_group_novelty.extend(explore_nov)
        for idx in explore_indices:
            used[idx] = True
    else:
        explore_indices.sort(key=lambda i: float(nov_np[i]), reverse=True)
        for idx in explore_indices:
            groups.append([idx])
            priorities.append(2)
            used[idx] = True

    # 4) remaining tokens (typically low-I low-n, plus any leftovers) -> contiguous pools
    leftover = [idx for idx in range(num_tokens) if not used[idx]]
    if frame_size is None:
        leftover_runs = _contiguous_runs(leftover)
        for run in leftover_runs:
            groups.append(run)
            priorities.append(3)
            for idx in run:
                used[idx] = True
    else:
        split_groups, split_priorities, split_importance, split_novelty = _frame_local_runs_with_scores(
            leftover,
            3,
            imp_np,
            nov_np,
            frame_size,
        )
        groups.extend(split_groups)
        priorities.extend(split_priorities)
        initial_group_importance.extend(split_importance)
        initial_group_novelty.extend(split_novelty)
        for idx in leftover:
            used[idx] = True

    if frame_size is None:
        groups, priorities = _adjust_groups_to_target(groups, priorities, imp_np, target)
    else:
        _t_b0 = time.perf_counter()
        _t_a = _t_a1 - _t_a0  # GPU transfer + quantile + classification
        _t_b = _t_b0 - _t_a1  # Group building (Python loops)
        _t_adjust0 = time.perf_counter()
        groups, priorities, group_importance, group_novelty = _adjust_frame_local_groups_to_target(
            groups,
            priorities,
            importance,
            novelty,
            target,
            frame_size,
            initial_group_importance,
            initial_group_novelty,
        )
        _t_adjust = time.perf_counter() - _t_adjust0
        _t_prune0 = time.perf_counter()
        groups, priorities = _prune_frame_local_groups(
            groups,
            priorities,
            group_importance,
            group_novelty,
            target,
            frame_size,
            frame_tolerance,
        )
        _t_prune = time.perf_counter() - _t_prune0
        # Print per-segment phase breakdown
        # print(
        #     f"[AQTC plan detail] tokens={num_tokens}  target={target}  "
        #     f"GPU={_t_a:.4f}s  build={_t_b:.4f}s  "
        #     f"adjust={_t_adjust:.4f}s  prune={_t_prune:.4f}s"
        # )

    # Vectorised: build flat index list on CPU, single GPU write
    flat_indices = [idx for group in groups for idx in group]
    covered = torch.zeros(num_tokens, dtype=torch.bool, device=importance.device)
    if flat_indices:
        covered[torch.tensor(flat_indices, dtype=torch.long, device=importance.device)] = True
    if frame_size is None and int(covered.sum().item()) != num_tokens:
        missing = (~covered).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Four-quadrant planner lost tokens: {missing}")

    warn_if_retention_off_target(
        retained=len(groups),
        total=num_tokens,
        keep_ratio=keep_ratio,
        tol=frame_tolerance if frame_size is not None else 0.05,
        context="AQTC plan_four_quadrant_groups",
    )

    debug = {
        "must_keep": must_keep.detach(),
        "merge_cand": merge_cand.detach(),
        "explore": explore.detach(),
        "pool_cand": pool_cand.detach(),
        "target_groups": torch.tensor(target),
        "actual_groups": torch.tensor(len(groups)),
    }
    # Pool strategies: priority 1 (merge_cand) uses argmax; all others use mean
    strategies = ["argmax" if p == 1 else "mean" for p in priorities]
    return groups, covered, debug, strategies
