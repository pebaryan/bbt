import torch
import torch.nn.functional as F


def sample_timesteps(
    batch_size: int,
    num_diffusion_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Uniformly sample timesteps in [1, num_diffusion_steps]."""
    return torch.randint(1, num_diffusion_steps + 1, (batch_size,), device=device, dtype=torch.long)


def mask_prob_for_t(
    t: torch.Tensor,
    num_diffusion_steps: int,
    min_mask_prob: float,
    max_mask_prob: float,
) -> torch.Tensor:
    """Linear mask schedule from low noise (small t) to high noise (large t)."""
    frac = t.float() / float(max(1, num_diffusion_steps))
    return min_mask_prob + (max_mask_prob - min_mask_prob) * frac


def corrupt_with_mask(
    x0: torch.Tensor,
    t: torch.Tensor,
    num_diffusion_steps: int,
    mask_token_id: int,
    min_mask_prob: float = 0.05,
    max_mask_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt clean bytes by replacing random positions with mask token."""
    if x0.ndim != 2:
        raise ValueError("x0 must have shape [B, T]")
    if t.ndim != 1:
        raise ValueError("t must have shape [B]")
    if x0.size(0) != t.size(0):
        raise ValueError("batch size mismatch between x0 and t")

    probs = mask_prob_for_t(
        t=t,
        num_diffusion_steps=num_diffusion_steps,
        min_mask_prob=min_mask_prob,
        max_mask_prob=max_mask_prob,
    )[:, None]
    mask = torch.rand(x0.shape, device=x0.device) < probs

    # Keep loss well-defined: ensure each sample has >=1 masked token.
    no_mask = ~mask.any(dim=1)
    if no_mask.any():
        rows = torch.nonzero(no_mask, as_tuple=False).squeeze(1)
        cols = torch.randint(0, x0.size(1), (rows.numel(),), device=x0.device)
        mask[rows, cols] = True

    x_t = x0.clone()
    x_t[mask] = mask_token_id
    return x_t, mask


def masked_denoise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on masked positions only."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if targets.ndim != 2 or mask.ndim != 2:
        raise ValueError("targets and mask must have shape [B, T]")

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (token_loss * mask_f).sum() / denom
