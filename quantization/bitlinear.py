import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary_quant import TernaryQuantSTE
from .act_quant import act_quant_per_token


class BitLinear(nn.Module):
    """BitLinear layer with ternary quantization and activation quantization.
    
    Implements the linear layer from BitNet b1.58 with:
    - Ternary quantization of weights using STE
    - Optional per-token activation quantization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        act_quant: bool = True,
    ):
        """Initialize BitLinear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            act_quant: Whether to quantize activations
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.act_quant = act_quant
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantized linear transformation
        """
        if self.act_quant:
            x = act_quant_per_token(x)
        wq = TernaryQuantSTE.apply(self.weight)
        return F.linear(x, wq, self.bias)