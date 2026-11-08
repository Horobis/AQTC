import torch
import torch.nn as nn
import torch.nn.functional as F

from .similarity import cosine_similarity_to_memory

# Signed (t,h,w) packed into int64 for batched searchsorted matching.
_T_BIAS = 8192
_H_BIAS = 512
_W_BIAS = 512
_T_STRIDE = 1 << 20
_H_STRIDE = 1 << 10


def _encode_coords(coords: torch.Tensor) -> torch.Tensor:
    t, h, w = coords.unbind(-1)
    return (t + _T_BIAS) * _T_STRIDE + (h + _H_BIAS) * _H_STRIDE + (w + _W_BIAS)


def _decode_key(key: int) -> tuple[int, int, int]:
    w = int(key % _H_STRIDE) - _W_BIAS
    rest = int(key // _H_STRIDE)
    h = int(rest % (_T_STRIDE // _H_STRIDE)) - _H_BIAS
    t = int(rest // (_T_STRIDE // _H_STRIDE)) - _T_BIAS
    return t, h, w


def _coords_to_tensor(coords: list[tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(coords, dtype=torch.long, device=device)


def _neighbor_offsets(device: torch.device) -> torch.Tensor:
    offs = [(dt, dh, dw) for dt in (-1, 0, 1) for dh in (-1, 0, 1) for dw in (-1, 0, 1)]
    return torch.tensor(offs, dtype=torch.long, device=device)


class NoveltyEstimator(nn.Module):
    """Marginal novelty vs already-kept compact memory.

    Memory stores:
      - location slots keyed by ``(t, h, w)``
      - a small FIFO of global prototypes

    For token ``x`` at ``(t,h,w)``:
      s_loc  = max cosine to prototypes in a 3x3 spatial neighborhood
      s_glob = mean of top-k cosines to global prototypes
      s      = max(λ s_loc + (1-λ) s_glob, s_glob)
      n      = 1 - s

    First segment (empty memory) uses in-segment spatial density / NMS-style
    scores instead of all-ones.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_capacity: int = 256,
        lambda_loc: float = 0.7,
        global_topk: int = 3,
        global_capacity: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_loc = float(lambda_loc)
        self.global_topk = int(global_topk)
        self.global_capacity = max(1, min(int(global_capacity), int(memory_capacity)))
        self.register_buffer("global_tokens", torch.empty(0, hidden_size), persistent=False)
        self.register_buffer("loc_keys", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("loc_protos", torch.empty(0, hidden_size), persistent=False)

    @property
    def location_prototypes(self) -> dict[tuple[int, int, int], torch.Tensor]:
        """Debug/compat view of packed location memory."""
        if self.loc_keys.numel() == 0:
            return {}
        out: dict[tuple[int, int, int], torch.Tensor] = {}
        for key, proto in zip(self.loc_keys.tolist(), self.loc_protos.unbind(0)):
            out[_decode_key(int(key))] = proto
        return out

    def reset(self) -> None:
        self.global_tokens = self.global_tokens.new_empty((0, self.hidden_size))
        self.loc_keys = self.loc_keys.new_empty(0)
        self.loc_protos = self.loc_protos.new_empty((0, self.hidden_size))

    def _global_memory(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.global_tokens.to(device=device, dtype=dtype)

    def _move_loc_to(self, device: torch.device) -> None:
        if self.loc_keys.device != device:
            self.loc_keys = self.loc_keys.to(device=device)
            self.loc_protos = self.loc_protos.to(device=device)

    @staticmethod
    def _neighborhood_keys(t: int, h: int, w: int) -> list[tuple[int, int, int]]:
        keys = []
        for dt in (-1, 0, 1):
            for dh in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    keys.append((t + dt, h + dh, w + dw))
        return keys

    def _location_similarity_tensor(
        self,
        tokens: torch.Tensor,
        coords: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Max neighborhood cosine; NaN where no stored neighbor exists."""
        n = tokens.shape[0]
        device = tokens.device
        s_loc = tokens.new_full((n,), float("nan"), dtype=torch.float32)
        if self.loc_keys.numel() == 0 or n == 0:
            return s_loc

        self._move_loc_to(device)
        coord_t = _coords_to_tensor(coords, device)
        offsets = _neighbor_offsets(device)
        neighbor_coords = coord_t.unsqueeze(1) + offsets.unsqueeze(0)
        neighbor_keys = _encode_coords(neighbor_coords).reshape(-1)

        order = torch.argsort(self.loc_keys)
        sorted_keys = self.loc_keys[order]
        sorted_protos = self.loc_protos[order]
        pos = torch.searchsorted(sorted_keys, neighbor_keys).clamp(max=sorted_keys.numel() - 1)
        matched = sorted_keys[pos] == neighbor_keys
        valid = matched.view(n, 27)
        if not bool(valid.any()):
            return s_loc

        normed = F.normalize(tokens.float(), p=2, dim=-1)
        proto = F.normalize(sorted_protos[pos].float(), p=2, dim=-1).view(n, 27, -1)
        sims = (normed.unsqueeze(1) * proto).sum(dim=-1)
        sims = sims.masked_fill(~valid, float("-inf"))
        max_sim = sims.max(dim=-1).values
        has_nb = valid.any(dim=-1)
        return torch.where(has_nb, max_sim, s_loc)

    def _location_similarity_batch(
        self,
        tokens: torch.Tensor,
        coords: list[tuple[int, int, int]],
    ) -> list[float | None]:
        """Compat wrapper: per-token max cosine, None if no neighbor in memory."""
        s_loc = self._location_similarity_tensor(tokens, coords)
        out: list[float | None] = []
        for val in s_loc.detach().cpu().tolist():
            out.append(None if val != val else float(val))
        return out

    def _global_similarity_batch(self, tokens: torch.Tensor) -> torch.Tensor | None:
        memory = self._global_memory(tokens.device, tokens.dtype)
        if memory.numel() == 0:
            return None
        sims = cosine_similarity_to_memory(tokens, memory)
        k = min(self.global_topk, sims.shape[-1])
        return sims.topk(k, dim=-1).values.mean(dim=-1)

    def _combine_similarity(self, s_loc: float | None, s_glob: float | None) -> float:
        if s_loc is None and s_glob is None:
            return 0.0
        if s_loc is None:
            return float(s_glob)
        if s_glob is None:
            return float(s_loc)
        mixed = self.lambda_loc * s_loc + (1.0 - self.lambda_loc) * s_glob
        return max(mixed, s_glob)

    def _combine_similarity_tensor(
        self,
        s_loc: torch.Tensor,
        s_glob: torch.Tensor | None,
    ) -> torch.Tensor:
        loc_valid = torch.isfinite(s_loc)
        if s_glob is None:
            similarity = torch.where(loc_valid, s_loc, torch.zeros_like(s_loc))
            return similarity
        s_glob = s_glob.float()
        glob_valid = torch.isfinite(s_glob)
        mixed = self.lambda_loc * s_loc + (1.0 - self.lambda_loc) * s_glob
        both = torch.maximum(mixed, s_glob)
        similarity = torch.zeros_like(s_loc)
        similarity = torch.where(loc_valid & ~glob_valid, s_loc, similarity)
        similarity = torch.where(~loc_valid & glob_valid, s_glob, similarity)
        similarity = torch.where(loc_valid & glob_valid, both, similarity)
        return similarity

    @torch.no_grad()
    def bootstrap_novelty(
        self,
        tokens: torch.Tensor,
        coords: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """In-segment spatial density / NMS-style novelty when memory is empty."""
        num_tokens = tokens.shape[0]
        if num_tokens == 0:
            return tokens.new_empty((0,))
        if num_tokens == 1:
            return tokens.new_ones(1)

        device = tokens.device
        coord_t = _coords_to_tensor(coords, device)
        offsets = _neighbor_offsets(device)
        self_mask = (offsets.abs().sum(dim=-1) == 0)
        neighbor_coords = coord_t.unsqueeze(1) + offsets.unsqueeze(0)
        neighbor_keys = _encode_coords(neighbor_coords)
        token_keys = _encode_coords(coord_t)
        order = torch.argsort(token_keys)
        sorted_keys = token_keys[order]
        pos = torch.searchsorted(sorted_keys, neighbor_keys.reshape(-1)).clamp(max=num_tokens - 1)
        matched = sorted_keys[pos] == neighbor_keys.reshape(-1)
        neighbor_idx = order[pos].view(num_tokens, 27)
        valid = matched.view(num_tokens, 27) & ~self_mask.unsqueeze(0)
        neighbor_idx = neighbor_idx.masked_fill(~valid, -1)

        normed = F.normalize(tokens.float(), p=2, dim=-1)
        mean_vec = normed.mean(dim=0)
        mean_sims = (normed * mean_vec).sum(dim=-1)
        scores = 1.0 - mean_sims

        has_neighbor = valid.any(dim=-1)
        if bool(has_neighbor.any()):
            safe_idx = neighbor_idx.clamp(min=0)
            neighbor_tokens = normed[safe_idx]
            sims = (normed.unsqueeze(1) * neighbor_tokens).sum(dim=-1)
            sims = sims.masked_fill(~valid, float("-inf"))
            max_sim = sims.max(dim=-1).values
            scores = torch.where(has_neighbor, 1.0 - max_sim, scores)
        return scores.to(dtype=tokens.dtype)

    @torch.no_grad()
    def score(
        self,
        tokens: torch.Tensor,
        coords: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        if len(coords) != tokens.shape[0]:
            raise ValueError(
                f"Coord/token mismatch: coords={len(coords)}, tokens={tokens.shape[0]}"
            )
        if self.loc_keys.numel() == 0 and self.global_tokens.numel() == 0:
            return self.bootstrap_novelty(tokens, coords)

        s_loc = self._location_similarity_tensor(tokens, coords)
        s_glob = self._global_similarity_batch(tokens)
        similarity = self._combine_similarity_tensor(s_loc, s_glob)
        return (1.0 - similarity).to(dtype=tokens.dtype)

    @torch.no_grad()
    def update_kept(
        self,
        tokens: torch.Tensor,
        coords: list[tuple[int, int, int]],
    ) -> None:
        """Write only kept representatives into location + global memory."""
        if tokens.numel() == 0:
            return
        tokens = tokens.detach()
        device = tokens.device
        self._move_loc_to(device)
        new_keys = _encode_coords(_coords_to_tensor(coords, device))
        new_protos = tokens.float()

        if self.loc_keys.numel() == 0:
            self.loc_keys = new_keys.detach()
            self.loc_protos = new_protos.detach()
        else:
            order = torch.argsort(self.loc_keys)
            sorted_keys = self.loc_keys[order]
            pos = torch.searchsorted(sorted_keys, new_keys).clamp(max=sorted_keys.numel() - 1)
            hit = sorted_keys[pos] == new_keys
            if hit.any():
                dest = order[pos[hit]]
                old = self.loc_protos[dest]
                self.loc_protos[dest] = (0.8 * old + 0.2 * new_protos[hit]).detach()
            miss = ~hit
            if miss.any():
                self.loc_keys = torch.cat((self.loc_keys, new_keys[miss]), dim=0)
                self.loc_protos = torch.cat((self.loc_protos, new_protos[miss]), dim=0)

        global_mem = self.global_tokens.to(device=device, dtype=tokens.dtype)
        global_mem = torch.cat((global_mem, tokens), dim=0)[-self.global_capacity :]
        self.global_tokens = global_mem.detach()

    # Backward-compatible aliases used by older call sites.
    def forward(self, current_tokens: torch.Tensor, coords: list[tuple[int, int, int]] | None = None):
        if coords is None:
            coords = [(0, 0, i) for i in range(current_tokens.shape[0])]
        return self.score(current_tokens, coords)

    def update(self, compressed_tokens: torch.Tensor) -> None:
        """Deprecated path: treat as global-only FIFO write without coords."""
        if compressed_tokens.numel() == 0:
            return
        tokens = compressed_tokens.detach().reshape(-1, self.hidden_size)
        global_mem = self.global_tokens.to(device=tokens.device, dtype=tokens.dtype)
        self.global_tokens = torch.cat((global_mem, tokens), dim=0)[-self.global_capacity :]
