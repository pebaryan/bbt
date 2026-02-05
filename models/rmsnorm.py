import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Simplified normalization without centering.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.
        
        Args:
            dim: Dimension to normalize over
            eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Normalized tensor with same shape
        """
        # x: [..., dim]
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight