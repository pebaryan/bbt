"""Mamba Block - combining SSM with MLP."""

import torch
import torch.nn as nn

from models.rmsnorm import RMSNorm
from models.mlp import SwiGLU, MLP
from .mamba_ssm import MambaSSM


class MambaBlock(nn.Module):
    """Transformer block using Mamba SSM instead of attention.
    
    Implements a Mamba-based transformer block with:
    - RMSNorm
    - Mamba SSM (replaces attention)
    - MLP with SwiGLU
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        act_quant: bool = True,
    ):
        """Initialize Mamba block.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            d_state: State dimension for SSM
            d_conv: Convolution kernel size for SSM
            expand: Expansion factor for SSM inner dimension
            act_quant: Whether to quantize activations
        """
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.ssm = MambaSSM(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand, act_quant=act_quant
        )
        self.n2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, act_quant=act_quant)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input [B, L, d_model]
            
        Returns:
            Output [B, L, d_model]
        """
        # Residual SSM
        x = x + self.ssm(self.n1(x))
        
        # Residual MLP
        x = x + self.mlp(self.n2(x))
        
        return x