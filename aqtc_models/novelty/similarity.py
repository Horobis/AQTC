import torch
import torch.nn.functional as F


def cosine_similarity_to_memory(
    current_tokens: torch.Tensor,
    memory_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return the cosine-similarity matrix [current_tokens, memory_tokens]."""
    if memory_tokens.numel() == 0:
        return current_tokens.new_empty((current_tokens.shape[0], 0))
    current_tokens = F.normalize(current_tokens.float(), p=2, dim=-1)
    memory_tokens = F.normalize(memory_tokens.float(), p=2, dim=-1)
    return current_tokens @ memory_tokens.transpose(0, 1)
