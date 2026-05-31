import torch
import torch.nn as nn


def _get_galore_param_groups(
    model: torch.nn.Module,
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    proj_type: str = "std",
) -> list[dict]:
    """Split parameters into GaLore (attn/mlp weights) and regular groups.
    
    GaLore is applied to linear projections in attention and MLP layers.
    All other parameters (embeddings, norms, biases, lm_head) use regular AdamW.
    """
    galore_params = []
    # LLaMA-style names kept for compatibility; our SwiGLU MLP uses
    # "gate_up" (fused gate+up) and "down" instead of gate_proj/up_proj/down_proj.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "gate_up", "mlp.down"]

    for module_name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        if not any(target in module_name for target in target_modules):
            continue
        galore_params.append(module.weight)

    id_galore = {id(p) for p in galore_params}
    regular_params = [p for p in model.parameters() if id(p) not in id_galore]

    return [
        {"params": regular_params},
        {"params": galore_params,
         "rank": rank,
         "update_proj_gap": update_proj_gap,
         "scale": scale,
         "proj_type": proj_type},
    ]


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    use_bnb: bool = True,
    use_galore: bool = False,
    galore_rank: int = 128,
) -> torch.optim.Optimizer:
    """Create optimizer with GaLore or bitsandbytes options.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        betas: Adam beta parameters
        weight_decay: Weight decay
        use_bnb: Whether to use bitsandbytes if available
        use_galore: Whether to use GaLore memory-efficient optimizer
        galore_rank: GaLore projection rank
        
    Returns:
        Optimizer instance, use_bnb flag
    """
    if use_galore:
        try:
            from galore_torch import GaLoreAdamW
            param_groups = _get_galore_param_groups(
                model, rank=galore_rank)
            opt = GaLoreAdamW(param_groups, lr=lr, betas=betas,
                              weight_decay=weight_decay)
            return opt, False
        except ImportError:
            print("[WARN] galore-torch not available, falling back to AdamW")

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