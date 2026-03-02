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
        time_step_min: float = 1e-3,
        time_step_max: float = 1e-1,
        dt_init: str = "log_uniform",
        a_init: str = "uniform_0_16",
    ):
        """Initialize Mamba SSM.

        Args:
            d_model: Model dimension
            d_state: State dimension (N)
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            act_quant: Whether to quantize activations
            use_checkpoint: Whether to use gradient checkpointing
            time_step_min: Minimum initial dt value for delta softplus bias
            time_step_max: Maximum initial dt value for delta softplus bias
            dt_init: Initialization for delta bias ("log_uniform" or "zeros")
            a_init: Initialization for A_log ("uniform_0_16" or "log_arange")
        """
        super().__init__()
        if time_step_min <= 0 or time_step_max <= 0:
            raise ValueError("time_step_min/time_step_max must be > 0")
        if time_step_min >= time_step_max:
            raise ValueError("time_step_min must be < time_step_max")
        if dt_init not in {"log_uniform", "zeros"}:
            raise ValueError("dt_init must be one of {'log_uniform', 'zeros'}")
        if a_init not in {"uniform_0_16", "log_arange"}:
            raise ValueError("a_init must be one of {'uniform_0_16', 'log_arange'}")

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
        self._init_dt_bias(
            bias=self.delta_proj.bias,
            d_inner=self.d_inner,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            dt_init=dt_init,
        )

        # Input-dependent B/C projections into state space
        self.B_proj = BitLinear(d_model, self.d_state, bias=False, act_quant=act_quant)
        self.C_proj = BitLinear(d_model, self.d_state, bias=False, act_quant=act_quant)

        # Output projection
        self.out_proj = BitLinear(
            self.d_inner, d_model, bias=False, act_quant=act_quant
        )

        # Learnable A matrix: [d_inner, d_state]
        if a_init == "uniform_0_16":
            A_log = torch.empty(self.d_inner, d_state, dtype=torch.float32).uniform_(
                0.0, 16.0
            )
        else:
            A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1)
            A = A.repeat(self.d_inner, 1)  # [d_inner, d_state]
            A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # D parameter (direct connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    @staticmethod
    def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
        # Inverse softplus: y = x + log(1 - exp(-x)).
        # Use expm1 for numerical stability at small x.
        return x + torch.log(-torch.expm1(-x))

    @classmethod
    def _init_dt_bias(
        cls,
        bias: torch.Tensor | None,
        d_inner: int,
        time_step_min: float,
        time_step_max: float,
        dt_init: str,
    ) -> None:
        if bias is None:
            return
        with torch.no_grad():
            if dt_init == "zeros":
                bias.zero_()
                return
            # Log-uniform in [time_step_min, time_step_max]
            dt = torch.exp(
                torch.empty(d_inner).uniform_(
                    math.log(time_step_min),
                    math.log(time_step_max),
                )
            )
            bias.copy_(cls._inv_softplus(dt).to(dtype=bias.dtype, device=bias.device))

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
