import torch
import torch.nn as nn


def act_quant_per_token(x: torch.Tensor, q: int = 127, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric per-token quantization on the last dimension.
    
    Args:
        x: Input tensor [..., C]
        q: Quantization levels (default 127 for int8)
        eps: Small value for numerical stability
        
    Returns:
        Quantized tensor with same shape as input
    """
    # x: [..., C]
    x2 = x.reshape(-1, x.shape[-1])
    s = x2.abs().amax(dim=-1, keepdim=True) / q
    s = torch.clamp(s, min=eps)
    xq = torch.clamp((x2 / s).round(), -q, q)
    return (xq * s).reshape_as(x)