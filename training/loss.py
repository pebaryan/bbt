import math
import torch


def bits_per_byte(loss: torch.Tensor) -> float:
    """Convert cross-entropy loss to bits per byte.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Bits per byte metric
    """
    return float(loss.item() / math.log(2.0))