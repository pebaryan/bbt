import math


def lr_for_step(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_factor: float = 0.1,
) -> float:
    """Learning rate scheduler with warmup and cosine decay.
    
    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        base_lr: Base learning rate
        min_lr_factor: Minimum LR as fraction of base_lr
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    # cosine decay to min_lr_factor of lr
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_lr_factor + (1 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * t)))


def seq_len_for_step(
    step: int,
    total_steps: int,
    warmup_frac: float = 0.2,
    base_seq_lens: tuple[int, int, int] = (2048, 4096, 8192),
    cap: int | None = None,
) -> int:
    """Curriculum learning sequence length schedule.
    
    Args:
        step: Current step
        total_steps: Total training steps
        warmup_frac: Fraction of steps for warmup phase
        base_seq_lens: Sequence lengths for each phase
        cap: Optional maximum sequence length
        
    Returns:
        Sequence length for current step
    """
    frac = step / max(1, total_steps)
    
    if frac < warmup_frac:
        seq_len = base_seq_lens[0]
    elif frac < 0.5:
        seq_len = base_seq_lens[1]
    else:
        seq_len = base_seq_lens[2]
    
    if cap is not None:
        seq_len = min(seq_len, cap)
    
    return seq_len