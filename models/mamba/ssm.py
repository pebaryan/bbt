"""State Space Model (SSM) kernels for Mamba."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.bitlinear import BitLinear


class S4Kernel(nn.Module):
    """S4 (Structured State Space) kernel.
    
    Implements the S4 kernel with Diagonal State Space (DSS) discretization.
    """
    
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, 
                 dt_max: float = 0.1, trainable_dt: bool = False):
        """Initialize S4 kernel.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (N)
            dt_min: Minimum value for time step delta
            dt_max: Maximum value for time step delta
            trainable_dt: Whether to make delta trainable
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Logarithmic spacing for A matrix diagonal
        # A = diag(-1/2 * log(1 + 1/d_state * arange(1, d_state+1)))
        log_A = -0.5 * torch.log(1 + 1/d_state * torch.arange(1, d_state + 1, dtype=torch.float32))
        self.register_buffer("log_A", log_A)
        
        # Initialize B and C randomly
        self.B = nn.Parameter(torch.randn(d_state, dtype=torch.complex64) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_state, dtype=torch.complex64) / math.sqrt(d_state))
        
        # Delta (time step) - can be fixed or learned
        log_dt = torch.uniform(math.log(dt_min), math.log(dt_max), (d_model,))
        if trainable_dt:
            self.log_dt = nn.Parameter(log_dt)
        else:
            self.register_buffer("log_dt", log_dt)
        
        # D parameter (direct connection)
        self.D = nn.Parameter(torch.ones(d_model))
    
    def forward(self, u: torch.Tensor, initial_states: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass using DSS discretization.
        
        Args:
            u: Input sequence [B, L, d_model]
            initial_states: Initial hidden states [B, d_state] (optional)
            
        Returns:
            Output sequence [B, L, d_model]
        """
        B, L, _ = u.shape
        
        # Get parameters
        delta = torch.exp(self.log_dt)  # [d_model]
        A = torch.exp(self.log_A)  # [d_state]
        
        # Discretize using ZOH
        # A_d = exp(A * delta), B_d = (A_d - I) * B * delta
        A_d = torch.exp(torch.outer(delta, A))  # [d_model, d_state]
        B_d = (A_d - 1.0) * torch.real(self.B) * delta.unsqueeze(-1)  # [d_model, d_state]
        
        # Process sequence
        x = initial_states if initial_states is not None else torch.zeros(B, self.d_state, dtype=torch.complex64, device=u.device)
        outputs = []
        
        for t in range(L):
            # State update: x_{t+1} = A_d * x_t + B_d * u_t
            u_t = u[:, t, :].unsqueeze(-1)  # [B, d_model, 1]
            x = A_d * x + B_d * u_t.squeeze(-1)  # [B, d_state]
            
            # Output: y_t = C * x_t + D * u_t
            y_t = torch.real(torch.sum(C * x.unsqueeze(-2), dim=-1)) + self.D * u[:, t, :]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # [B, L, d_model]


class SelectiveSSM(nn.Module):
    """Selective State Space Model.
    
    Extends S4 with input-dependent state transitions.
    """
    
    def __init__(self, d_model: int, d_state: int = 64, expand_factor: int = 2):
        """Initialize selective SSM.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (N)
            expand_factor: Expansion factor for intermediate dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model
        
        # Project input to higher dimension
        self.in_proj = BitLinear(d_model, 2 * self.d_inner, bias=False, act_quant=True)
        
        # SSM kernel
        self.ssm_kernel = S4Kernel(self.d_inner, d_state)
        
        # Output projection
        self.out_proj = BitLinear(self.d_inner, d_model, bias=False, act_quant=True)
        
        # Selective delta projection
        self.delta_proj = BitLinear(self.d_inner, d_state, bias=True, act_quant=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with selective SSM.
        
        Args:
            x: Input [B, L, d_model]
            
        Returns:
            Output [B, L, d_model]
        """
        # Project to higher dimension
        x_projected = self.in_proj(x)  # [B, L, 2*d_inner]
        
        # Split for gate and input
        x_gate, x_input = x_projected.chunk(2, dim=-1)
        
        # Apply silu gate
        x_input = F.silu(x_gate) * x_input
        
        # Compute delta (selective)
        delta = self.delta_proj(x_input)  # [B, L, d_state]
        
        # Apply SSM with selective delta
        # Note: This is a simplified version - full selective SSM requires
        # parameterized discretization based on delta
        y = self.ssm_kernel(x_input)
        
        # Output projection
        y = self.out_proj(y)
        
        return y