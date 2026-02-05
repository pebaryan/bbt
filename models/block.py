import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .rmsnorm import RMSNorm
from .attention import CausalSelfAttention
from .mlp import MLP


class Block(nn.Module):
    """Transformer block with attention and MLP.
    
    Implements a single transformer block with:
    - RMSNorm
    - Causal self-attention
    - MLP with SwiGLU
    - Optional gradient checkpointing
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        ckpt: bool = True,
    ):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_head: Number of attention heads
            d_ff: Feed-forward dimension
            act_quant: Whether to quantize activations
            rope_base: Base for RoPE calculation
            use_sdpa: Whether to use SDPA
            ckpt: Whether to use gradient checkpointing
        """
        super().__init__()
        self.ckpt = ckpt
        self.n1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_head, rope_base=rope_base, act_quant=act_quant, use_sdpa=use_sdpa)
        self.n2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, act_quant=act_quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional checkpointing.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            Output tensor [B, T, C]
        """
        def attn_fn(x):
            return x + self.attn(self.n1(x))

        def mlp_fn(x):
            return x + self.mlp(self.n2(x))

        if self.ckpt and self.training:
            x = checkpoint(attn_fn, x, use_reentrant=False)
            x = checkpoint(mlp_fn, x, use_reentrant=False)
        else:
            x = attn_fn(x)
            x = mlp_fn(x)
        return x