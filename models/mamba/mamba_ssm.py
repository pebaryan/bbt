"""Mamba SSM module - a more complete implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Convolution layer for local processing
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1
        )
        
        # Projection to inner dimension
        self.in_proj = BitLinear(d_model, 2 * self.d_inner, bias=False, act_quant=act_quant)
        
        # Selective delta projection
        self.delta_proj = BitLinear(self.d_inner, self.d_state, bias=True, act_quant=act_quant)
        
        # Input-dependent B/C projections into state space
        self.B_proj = BitLinear(self.d_inner, self.d_state, bias=False, act_quant=act_quant)
        self.C_proj = BitLinear(self.d_inner, self.d_state, bias=False, act_quant=act_quant)
        
        # Output projection
        self.out_proj = BitLinear(self.d_inner, d_model, bias=False, act_quant=act_quant)
        
        # Logarithmic spacing for A matrix
        self.register_buffer(
            "log_A",
            -0.5 * torch.log(1 + torch.arange(1, d_state + 1, dtype=torch.float32) / d_state)
        )
        
        # D parameter (direct connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
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
        x_conv = self.conv1d(x_input.transpose(1, 2))  # [B, d_inner, L]
        x_conv = x_conv[:, :, :L]  # Trim padding
        x_input = x_conv.transpose(1, 2)  # [B, L, d_inner]
        
        # Compute delta (selective)
        delta = self.delta_proj(x_input)  # [B, L, d_state]
        
        # Discretize A with selective per-token delta: A_d in (0,1)
        A_d = torch.exp(delta * self.log_A.view(1, 1, -1))  # [B, L, d_state]

        # Input-dependent B/C in state space
        B_matrix = self.B_proj(x_input.view(B * L, self.d_inner)).view(B, L, self.d_state)
        C_matrix = self.C_proj(x_input.view(B * L, self.d_inner)).view(B, L, self.d_state)

        # Single state vector per token stream (lightweight approximation).
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        state_signal = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
        for t in range(L):
            h = A_d[:, t] * h + delta[:, t] * B_matrix[:, t]
            state_signal[:, t, 0] = torch.sum(C_matrix[:, t] * h, dim=-1)

        # Modulate token features with SSM signal and keep direct pathway.
        y_inner = x_input * (torch.sigmoid(state_signal) + self.D.view(1, 1, -1))

        # Output projection
        y = self.out_proj(y_inner)  # [B, L, d_model]
        
        return y
