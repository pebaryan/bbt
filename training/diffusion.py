import math

import torch
import torch.nn.functional as F


def _build_span_mask(
    probs: torch.Tensor,
    seq_len: int,
    span_len: float,
    device: torch.device,
) -> torch.Tensor:
    """Create contiguous span masks with approximately probs * seq_len masked tokens."""
    if span_len <= 0:
        raise ValueError("span_len must be > 0")

    batch = probs.size(0)
    mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)

    # Convert per-sample mask probability into per-sample token budget.
    targets = torch.clamp((probs * float(seq_len)).round().long(), min=1, max=seq_len)

    # Geometric-like span length sampling: mean length ~= span_len.
    p = min(1.0, max(1e-6, 1.0 / float(span_len)))
    log_one_minus_p = None if p >= 1.0 else math.log(1.0 - p)

    for i in range(batch):
        target = int(targets[i].item())
        if target <= 0:
            continue

        masked = 0
        tries = 0
        max_tries = max(64, target * 8)
        while masked < target and tries < max_tries:
            remaining = target - masked
            # Sample geometric span length with mean ~= span_len.
            if p >= 1.0:
                span = 1
            else:
                u = float(torch.rand((), device=device).item())
                u = max(u, 1e-12)
                span = int(math.floor(math.log(u) / log_one_minus_p)) + 1
            span = max(1, min(span, remaining, seq_len))

            start_hi = seq_len - span
            start = int(torch.randint(0, start_hi + 1, (1,), device=device).item())
            end = start + span

            before = int(mask[i].sum().item())
            mask[i, start:end] = True
            after = int(mask[i].sum().item())
            masked = after
            tries += 1

            # If no progress (heavy overlap), jump out earlier.
            if after == before and tries >= target:
                break

        if masked < target:
            # Backfill any deficit with random unmasked positions.
            deficit = target - masked
            unmasked_idx = torch.nonzero(~mask[i], as_tuple=False).squeeze(1)
            if unmasked_idx.numel() > 0:
                take = min(deficit, int(unmasked_idx.numel()))
                perm = torch.randperm(unmasked_idx.numel(), device=device)[:take]
                mask[i, unmasked_idx[perm]] = True

    return mask


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
    mask_mode: str = "token",
    span_len: float = 8.0,
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
    )
    probs_2d = probs[:, None]
    if mask_mode == "token":
        mask = torch.rand(x0.shape, device=x0.device) < probs_2d
    elif mask_mode == "span":
        mask = _build_span_mask(
            probs=probs,
            seq_len=x0.size(1),
            span_len=span_len,
            device=x0.device,
        )
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode!r}. Expected 'token' or 'span'.")

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
