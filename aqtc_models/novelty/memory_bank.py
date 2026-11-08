import torch
import torch.nn as nn


class VisualTokenMemoryBank(nn.Module):
    """Fixed-capacity FIFO memory for compressed visual tokens."""

    def __init__(self, hidden_size: int, capacity: int = 256):
        super().__init__()
        if capacity < 1:
            raise ValueError("Memory capacity must be at least 1")
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.register_buffer("tokens", torch.empty(0, hidden_size), persistent=False)

    def read(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.tokens.to(device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, tokens: torch.Tensor) -> None:
        if tokens.numel() == 0:
            return
        tokens = tokens.detach().reshape(-1, self.hidden_size)
        memory = self.tokens.to(device=tokens.device, dtype=tokens.dtype)
        self.tokens = torch.cat((memory, tokens), dim=0)[-self.capacity :]

    @torch.no_grad()
    def reset(self) -> None:
        self.tokens = self.tokens.new_empty((0, self.hidden_size))

    def __len__(self) -> int:
        return self.tokens.shape[0]
