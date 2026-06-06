"""Log-Linear SSM — Fenwick-tree multi-scale state hierarchy.

Inspired by: Han Guo et al. "Log-Linear Attention" (ICLR 2026)

The key insight: instead of a single fixed-size SSM state h ∈ R^{d×N},
maintain L = ⌈log₂(T)⌉ parallel states h_0, ..., h_{L-1}, each
covering a disjoint timescale.

Level ℓ updates every 2^ℓ steps (cruder discretisation = coarser
temporal view).  At every output position the model attends to all L
levels via a learned weighted gated sum, giving it fine-grained
access to recent tokens *and* a compact summary of distant context
in a single O(T log T) pass.

Reference structure (Guo et al. §3):
  o_t = Σ_{ℓ=0}^{L-1}  λ_t^{(ℓ)}  q_tᵀ  S_t^{(ℓ)}
  where S_t^{(ℓ)} ∈ R^{d×d} is the Fenwick-bucket state for level ℓ
  at time t and λ_t^{(ℓ)} is a learned (or input-dependent) weight.

This file adapts the idea to Mamba-1: each level keeps a diagonal-SSM
state h_ℓ ∈ R^{d_inner × d_state} that is updated at its own rate.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from quantization.bitlinear import BitLinear


class LogLinearSSM(nn.Module):
    """Multi-scale SSM with Fenwick-tree state hierarchy.

    Args:
        d_model: Model dimension.
        d_state: State dimension per level (N in SSM notation).
        d_conv: Convolution kernel size (same as MambaSSM).
        expand: Expansion factor for inner dimension.
        num_levels: Number of Fenwick levels.  Defaults to ⌈log₂(max_seq_len)⌉.
        act_quant: Whether to use BitLinear activation quantisation.
        use_checkpoint: Gradient checkpointing.
        time_step_min, time_step_max: Range for delta bias init.
        dt_init: Initialisation mode for delta bias.
        a_init: Initialisation mode for A_log.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_levels: int = 9,               # ⌈log₂(512)⌉ = 9
        act_quant: bool = True,
        use_checkpoint: bool = True,
        use_triton: bool = False,
        time_step_min: float = 1e-3,
        time_step_max: float = 1e-1,
        dt_init: str = "log_uniform",
        a_init: str = "log_arange",
    ):
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
        self.d_inner = expand * d_model
        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        try:
            from models.mamba.loglinear_ssm_triton import _USE_TRITON as _UT
        except Exception:
            _UT = False
        self.use_triton = use_triton and _UT

        # ── Shared components (same as MambaSSM) ────────────────────────
        # Convolution for local processing
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        # Input projection
        self.in_proj = BitLinear(d_model, 2 * self.d_inner, bias=False,
                                 act_quant=act_quant)
        # Output projection
        self.out_proj = BitLinear(self.d_inner, d_model, bias=False,
                                  act_quant=act_quant)

        # D parameter (direct connection) — shared across levels
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # Activation
        self.act = nn.SiLU()

        # ── Level-specific projections ──────────────────────────────────
        # Each level has its own delta/ B / C projections so it can learn
        # a different temporal receptive field.
        self.B_projs = nn.ModuleList([
            BitLinear(self.d_inner, self.d_state, bias=False, act_quant=act_quant)
            for _ in range(num_levels)
        ])
        self.C_projs = nn.ModuleList([
            BitLinear(self.d_inner, self.d_state, bias=False, act_quant=act_quant)
            for _ in range(num_levels)
        ])
        self.delta_projs = nn.ModuleList([
            BitLinear(self.d_inner, self.d_inner, bias=True, act_quant=act_quant)
            for _ in range(num_levels)
        ])
        # Initialise delta biases for each level
        for dp in self.delta_projs:
            self._init_dt_bias(
                bias=dp.bias, d_inner=self.d_inner,
                time_step_min=time_step_min,
                time_step_max=time_step_max, dt_init=dt_init,
            )

        # ── Level mixing weights (learned logits, softmax-normalised) ───
        self.level_logits = nn.Parameter(
            torch.full((num_levels,), -1.0)  # start near-uniform, slight bias
        )

        # ── Shared A matrix ─────────────────────────────────────────────
        if a_init == "uniform_0_16":
            A_log = torch.empty(self.d_inner, d_state, dtype=torch.float32).uniform_(0.0, 16.0)
        else:
            A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1)
            A = A.repeat(self.d_inner, 1)
            A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
        return x + torch.log(-torch.expm1(-x))

    @classmethod
    def _init_dt_bias(cls, bias, d_inner, time_step_min, time_step_max, dt_init):
        if bias is None:
            return
        with torch.no_grad():
            if dt_init == "zeros":
                bias.zero_()
                return
            dt = torch.exp(torch.empty(d_inner).uniform_(
                math.log(time_step_min), math.log(time_step_max)))
            bias.copy_(cls._inv_softplus(dt).to(dtype=bias.dtype, device=bias.device))

    @staticmethod
    def _fenwick_levels(t: int, num_levels: int) -> list[int]:
        """Return the Fenwick-bucket levels active at position *t*.

        The prefix ``[0, t+1)`` (1-indexed) is decomposed into disjoint
        buckets whose sizes are powers of two.  Each bucket maps to a
        level ℓ where 2^ℓ = bucket size.

        Example (1-indexed position):
          t+1 = 7 (binary 111) → levels [0, 1, 2]
          t+1 = 8 (binary 1000) → levels [3]

        Returns levels sorted from finest (0) to coarsest (L-1).
        """
        i = t + 1  # convert to 1-indexed
        levels: list[int] = []
        while i > 0:
            lsb = i & -i
            level = (lsb.bit_length() - 1)
            if level < num_levels:
                levels.append(level)
            i -= lsb
        return levels  # naturally from finest to coarser

    @classmethod
    def _fenwick_mask(cls, length: int, num_levels: int) -> list[list[int]]:
        """Pre-compute the list of active levels for every position 0..L-1."""
        return [cls._fenwick_levels(t, num_levels) for t in range(length)]

    # ── multi-level SSM scan ──────────────────────────────────────────────

    @staticmethod
    def _ssm_scan_multilevel(
        x_input: torch.Tensor,        # [B, L, d_inner]
        delta: list[torch.Tensor],     # LL: [B, L, d_inner]
        B_matrix: list[torch.Tensor],  # LL: [B, L, d_state]
        C_matrix: list[torch.Tensor],  # LL: [B, L, d_state]
        A: torch.Tensor,               # [d_inner, d_state]
        D: torch.Tensor,               # [d_inner]
        level_mask: list[list[int]],   # LL active levels per position
        num_levels: int,
        level_logits: torch.Tensor,    # [num_levels]
    ) -> torch.Tensor:
        Bsz, L, d_inner = x_input.shape
        d_state = A.shape[1]

        # Level states: list of [B, d_inner, d_state]
        h = [torch.zeros(Bsz, d_inner, d_state, device=x_input.device,
                         dtype=x_input.dtype) for _ in range(num_levels)]
        # Output placeholder
        y_ssm = torch.zeros(Bsz, L, d_inner, device=x_input.device,
                            dtype=x_input.dtype)

        level_weights = F.softmax(level_logits, dim=0)  # [num_levels]

        for t in range(L):
            x_t = x_input[:, t, :]          # [B, d_inner]
            active = level_mask[t]          # list of ℓ indices

            # Update ONLY the active levels' states
            for ℓ in active:
                d_t = F.softplus(delta[ℓ][:, t])   # [B, d_inner]
                B_t = B_matrix[ℓ][:, t]             # [B, d_state]
                A_bar_t = torch.exp(d_t.unsqueeze(-1) * A)  # [B, d_inner, d_state]
                h[ℓ] = (
                    A_bar_t * h[ℓ]
                    + d_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
                )

            # Output = weighted sum over ALL levels
            out_t = torch.zeros(Bsz, d_inner, device=x_input.device,
                                dtype=x_input.dtype)
            for ℓ in range(num_levels):
                C_t = C_matrix[ℓ][:, t]  # [B, d_state]
                y_ℓ = torch.sum(C_t.unsqueeze(1) * h[ℓ], dim=-1)  # [B, d_inner]
                out_t = out_t + level_weights[ℓ] * y_ℓ

            y_ssm[:, t, :] = out_t

        return y_ssm + D.view(1, 1, -1) * x_input

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Same interface as MambaSSM:
          x: [B, L, d_model] → out: [B, L, d_model]
        """
        B, L, _ = x.shape

        # Pre-compute Fenwick level mask for this sequence length
        level_mask = self._fenwick_mask(L, self.num_levels)

        # Input projection + split
        x_proj = self.in_proj(x)                       # [B, L, 2*d_inner]
        x_inner, z = x_proj.chunk(2, dim=-1)            # 2 × [B, L, d_inner]

        # Convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))   # [B, d_inner, L]
        x_conv = x_conv[:, :, :L].contiguous()
        x_inner = self.act(x_conv.transpose(1, 2))      # [B, L, d_inner]

        # Compute level-specific delta, B, C from the convolved input
        delta = [F.softplus(proj(x_inner)) for proj in self.delta_projs]
        B_mat = [proj(x_inner.reshape(B * L, -1)).reshape(B, L, self.d_state)
                 for proj in self.B_projs]
        C_mat = [proj(x_inner.reshape(B * L, -1)).reshape(B, L, self.d_state)
                 for proj in self.C_projs]

        A = -torch.exp(self.A_log.float())

        # Multi-level SSM scan
        if self.use_triton and not self.training:
            try:
                from models.mamba.loglinear_ssm_triton import triton_level_scan_forward
                y_inner = triton_level_scan_forward(
                    x_inner, delta, B_mat, C_mat,
                    A, self.D,
                    level_mask, self.num_levels,
                    self.level_logits, x.device,
                )
            except Exception:
                # Fall back to Python on *any* Triton/runtime issue
                y_inner = self._ssm_scan_multilevel(
                    x_inner, delta, B_mat, C_mat,
                    A, self.D,
                    level_mask, self.num_levels, self.level_logits,
                )
        elif self.use_checkpoint and self.training:
            y_inner = checkpoint(
                self._ssm_scan_multilevel,
                x_inner,
                delta, B_mat, C_mat,
                A, self.D,
                level_mask, self.num_levels, self.level_logits,
                use_reentrant=False,
            )
        else:
            y_inner = self._ssm_scan_multilevel(
                x_inner, delta, B_mat, C_mat,
                A, self.D,
                level_mask, self.num_levels, self.level_logits,
            )

        # Gate + out projection
        y_gated = y_inner * F.silu(z)
        return self.out_proj(y_gated)
