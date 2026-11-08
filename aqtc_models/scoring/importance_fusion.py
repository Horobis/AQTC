import torch
import torch.nn as nn


class ImportanceFusion(nn.Module):
    """Fuse query relevance and saliency into one importance score."""

    def __init__(
        self,
        query_weight: float = 0.7,
        saliency_weight: float = 0.3,
    ):
        super().__init__()

        raw_weights = [float(query_weight), float(saliency_weight)]
        if any(weight < 0 for weight in raw_weights):
            raise ValueError("Importance weights must be non-negative")
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raise ValueError("Importance weights must sum to a positive value")
        self.weight_values = tuple(weight / weight_sum for weight in raw_weights)

    @staticmethod
    def _min_max_normalize(scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() < 2:
            return scores.new_ones(scores.shape)
        minimum = scores.min()
        span = scores.max() - minimum
        if span.abs() <= torch.finfo(scores.dtype).eps:
            return scores.new_ones(scores.shape)
        return (scores - minimum) / span

    def forward(
        self,
        query_scores: torch.Tensor,
        saliency_scores: torch.Tensor,
    ) -> torch.Tensor:
        components = torch.stack(
            [
                self._min_max_normalize(query_scores.float()),
                self._min_max_normalize(saliency_scores.float()),
            ],
            dim=-1,
        )
        weights = components.new_tensor(self.weight_values)
        return (components * weights).sum(dim=-1).to(query_scores.dtype)
