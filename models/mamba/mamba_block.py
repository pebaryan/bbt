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
        use_checkpoint: bool = True,
        time_step_min: float = 1e-3,
        time_step_max: float = 1e-1,
        dt_init: str = "log_uniform",
        a_init: str = "uniform_0_16",
    ):
        """Initialize Mamba block.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            d_state: State dimension for SSM
            d_conv: Convolution kernel size for SSM
            expand: Expansion factor for SSM inner dimension
            act_quant: Whether to quantize activations
            use_checkpoint: Whether to use gradient checkpointing for SSM
            time_step_min: Minimum initial dt value for delta softplus bias
            time_step_max: Maximum initial dt value for delta softplus bias
            dt_init: Initialization for delta bias ("log_uniform" or "zeros")
            a_init: Initialization for A_log ("uniform_0_16" or "log_arange")
        """
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.ssm = MambaSSM(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            act_quant=act_quant,
            use_checkpoint=use_checkpoint,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            dt_init=dt_init,
            a_init=a_init,
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
