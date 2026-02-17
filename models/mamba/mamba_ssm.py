"""Mamba SSM module - a more complete implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from quantization.bitlinear import BitLinear


class MambaSSM(nn.Module):
    """Mamba State Space Model layer.

    Implements the core SSM of Mamba with:
    - Input expansion
    - Selective delta computation
    - Diagonal state space discretization
    - Output projection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        act_quant: bool = True,
        use_checkpoint: bool = True,
    ):
        """Initialize Mamba SSM.

        Args:
            d_model: Model dimension
            d_state: State dimension (N)
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            act_quant: Whether to quantize activations
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.use_checkpoint = use_checkpoint

        # Convolution layer for local processing
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Projection to inner dimension
        self.in_proj = BitLinear(
            d_model, 2 * self.d_inner, bias=False, act_quant=act_quant
        )

        # Selective delta projection (per-channel step size)
        self.delta_proj = BitLinear(
            d_model, self.d_inner, bias=True, act_quant=act_quant
        )

        # Input-dependent B/C projections into state space
        self.B_proj = BitLinear(d_model, self.d_state, bias=False, act_quant=act_quant)
        self.C_proj = BitLinear(d_model, self.d_state, bias=False, act_quant=act_quant)

        # Output projection
        self.out_proj = BitLinear(
            self.d_inner, d_model, bias=False, act_quant=act_quant
        )

        # Learnable A matrix: [d_inner, d_state]
        # Initialize with log-spaced values for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1)
        A = A.repeat(self.d_inner, 1)  # [d_inner, d_state]
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (direct connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    @staticmethod
    def _ssm_scan(
        x_input: torch.Tensor,
        delta: torch.Tensor,
        B_matrix: torch.Tensor,
        C_matrix: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """SSM scan operation - separable for checkpointing."""
        B, L, d_inner = x_input.shape
        d_state = A.shape[1]

        h = torch.zeros(B, d_inner, d_state, device=x_input.device, dtype=x_input.dtype)
        y_ssm = torch.zeros(B, L, d_inner, device=x_input.device, dtype=x_input.dtype)

        for t in range(L):
            A_bar_t = torch.exp(delta[:, t].unsqueeze(-1) * A)
            h = A_bar_t * h + delta[:, t].unsqueeze(-1) * B_matrix[:, t].unsqueeze(
                1
            ) * x_input[:, t].unsqueeze(-1)
            y_ssm[:, t, :] = torch.sum(C_matrix[:, t].unsqueeze(1) * h, dim=-1)

        return y_ssm + D.view(1, 1, -1) * x_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, d_model]

        Returns:
            Output [B, L, d_model]
        """
        B, L, _ = x.shape

        # Project to inner dimension
        x_projected = self.in_proj(x)  # [B, L, 2*d_inner]

        # Split for gate and input
        x_gate, x_input = x_projected.chunk(2, dim=-1)
        x_input = F.silu(x_gate) * x_input  # [B, L, d_inner]

        # Convolution (local processing)
        x_conv = self.conv1d(x_input.transpose(1, 2).contiguous())  # [B, d_inner, L]
        x_conv = x_conv[:, :, :L].contiguous()  # Trim padding
        x_input = x_conv.transpose(1, 2).contiguous()  # [B, L, d_inner]

        # Selective delta (per-channel step size)
        delta = F.softplus(self.delta_proj(x))  # [B, L, d_inner]

        # Input-dependent B/C in state space (from original x)
        B_matrix = self.B_proj(x.reshape(B * L, -1)).reshape(B, L, self.d_state)
        C_matrix = self.C_proj(x.reshape(B * L, -1)).reshape(B, L, self.d_state)

        # A matrix: exp(A_log) for negative A (stability)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # SSM scan with optional checkpointing
        if self.use_checkpoint and self.training:
            y_inner = checkpoint(
                self._ssm_scan,
                x_input,
                delta,
                B_matrix,
                C_matrix,
                A,
                self.D,
                use_reentrant=False,
            )
        else:
            y_inner = self._ssm_scan(x_input, delta, B_matrix, C_matrix, A, self.D)

        # Output projection
        y = self.out_proj(y_inner)  # [B, L, d_model]

        return y
