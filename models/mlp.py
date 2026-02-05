import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.bitlinear import BitLinear


class SwiGLU(nn.Module):
    """SwiGLU activation function.
    
    Swish-Gated Linear Unit: silu(a) * b
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor [..., 2*d_ff]
            
        Returns:
            Output tensor [..., d_ff]
        """
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class MLP(nn.Module):
    """MLP with SwiGLU and quantization.
    
    Implements the feed-forward network with:
    - Gate and up projections (2*d_ff)
    - SwiGLU activation
    - Down projection
    """
    
    def __init__(self, d_model: int, d_ff: int, act_quant: bool = True):
        """Initialize MLP.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (actual is 2*d_ff for SwiGLU)
            act_quant: Whether to quantize activations
        """
        super().__init__()
        # SwiGLU uses 2*d_ff for gate+up
        self.gate_up = BitLinear(
            d_model, 2 * d_ff, bias=False, act_quant=act_quant)
        self.down = BitLinear(d_ff, d_model, bias=False, act_quant=act_quant)
        self.act = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            Output tensor [B, T, C]
        """
        return self.down(self.act(self.gate_up(x)))