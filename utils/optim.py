import torch


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    use_bnb: bool = True,
) -> torch.optim.Optimizer:
    """Create optimizer with bitsandbytes fallback.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        betas: Adam beta parameters
        weight_decay: Weight decay
        use_bnb: Whether to use bitsandbytes if available
        
    Returns:
        Optimizer instance
    """
    if use_bnb:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            return opt, True
        except ImportError:
            pass
    
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    return opt, False


def create_grad_scaler() -> torch.amp.GradScaler:
    """Create gradient scaler for mixed precision training.
    
    Returns:
        GradScaler instance
    """
    return torch.amp.GradScaler()