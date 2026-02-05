import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the tensor for RoPE.
    
    Args:
        x: Input tensor [..., D]
        
    Returns:
        Rotated tensor with same shape
    """
    x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Applies rotary position embeddings to queries and keys
    for attention computation.
    """
    
    def __init__(self, head_dim: int, base: float = 10000.0):
        """Initialize RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            base: Base for frequency calculation
        """
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / \
            (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.
        
        Args:
            x: Input tensor [B, H, T, D]
            positions: Position indices [T]
            
        Returns:
            Embedded tensor with same shape as input
        """
        # x: [B, H, T, D]
        # positions: [T]
        freqs = torch.einsum("t,d->td", positions.float(),
                             self.inv_freq)  # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
        cos = emb.cos()[None, None, :, :]  # [1,1,T,D]
        sin = emb.sin()[None, None, :, :]
        return (x * cos) + (rotate_half(x) * sin)