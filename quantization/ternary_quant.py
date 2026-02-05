import torch
import torch.nn as nn


class TernaryQuantSTE(torch.autograd.Function):
    """Ternary quantization with Straight-Through Estimator (STE).
    
    Implements absmean scaling similar to BitNet b1.58.
    """
    
    @staticmethod
    def forward(ctx, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Forward pass: quantize weights to {-1, 0, +1}.
        
        Args:
            w: Weight tensor to quantize
            eps: Small value for numerical stability
            
        Returns:
            Quantized weights with original magnitude preserved
        """
        # absmean scaling (BitNet b1.58 style)
        gamma = w.abs().mean()
        gamma = torch.clamp(gamma, min=eps)
        w_scaled = w / gamma

        wq = torch.clamp(w_scaled, -1.0, 1.0).round()  # -> {-1,0,+1}
        ctx.save_for_backward(w_scaled)
        # Re-apply scale so the quantized weight keeps the original magnitude.
        return wq * gamma

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        """Backward pass: use STE with optional clamp mask for stability."""
        (w_scaled,) = ctx.saved_tensors
        # STE: pass gradients through; optional clamp mask for stability:
        mask = (w_scaled.abs() <= 1.0).to(grad_out.dtype)
        return grad_out * mask, None