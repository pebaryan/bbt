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


# ─── Noise schedule ───────────────────────────────────────────────────────────

def get_noise_schedule(t_norm: torch.Tensor, eps: float = 0.001) -> tuple[torch.Tensor, torch.Tensor]:
    """CosineSquared noise schedule adapted from MDLM/SFM.

    Args:
        t_norm: Float tensor in [0, 1], t=0 = clean, t=1 = max noise.
        eps: Minimum alpha (prevent masking everything with prob=1).

    Returns:
        alpha_t:  [*t_norm.shape] probability a token stays clean at timestep t.
        dalpha_t: Derivative of alpha_t w.r.t. t_norm.
    """
    angle = (math.pi / 2) * (1 - t_norm)
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)

    base_alpha = sin_a ** 2                                      # sin²(π/2 · (1-t))
    alpha = eps + (1 - eps) * base_alpha                         # scaled to [eps, 1]

    dalpha = -(1 - eps) * 2 * sin_a * cos_a * (math.pi / 2)      # derivative
    return alpha, dalpha


def sample_timesteps(
    batch_size: int,
    num_diffusion_steps: int,
    device: torch.device,
    antithetic: bool = True,
) -> torch.Tensor:
    """Sample timesteps in [1, num_diffusion_steps].

    With antithetic=True, uses stratified sampling for more uniform coverage
    of the timestep range (following MDLM/SFM).
    """
    if antithetic:
        eps = torch.rand(batch_size, device=device)
        offset = torch.arange(batch_size, device=device).float() / batch_size
        t_cont = (eps / batch_size + offset).clamp(0.0, 1.0 - 1e-6)
        t = (t_cont * num_diffusion_steps).long().clamp(1, num_diffusion_steps)
    else:
        t = torch.randint(1, num_diffusion_steps + 1, (batch_size,), device=device, dtype=torch.long)
    return t


# ─── Mask corruption ─────────────────────────────────────────────────────────

def mask_prob_for_t(
    t: torch.Tensor,
    num_diffusion_steps: int,
    min_mask_prob: float,
    max_mask_prob: float,
) -> torch.Tensor:
    """Mask probability from CosineSquared noise schedule.

    Uses alpha_t = P(token stays clean), so mask_prob = 1 - alpha_t.
    Clamped to [min_mask_prob, max_mask_prob].

    Args:
        t: Timestep tensor [B], values in [1, num_diffusion_steps].
        num_diffusion_steps: Total diffusion steps.
        min_mask_prob: Minimum mask probability (floor).
        max_mask_prob: Maximum mask probability (ceiling).

    Returns:
        Mask probability for each sample, shape [B].
    """
    t_norm = t.float() / float(max(1, num_diffusion_steps))  # → [1/T, 1]
    alpha, _ = get_noise_schedule(t_norm)
    mask_prob = 1.0 - alpha
    return mask_prob.clamp(min_mask_prob, max_mask_prob)


def get_alpha_for_t(
    t: torch.Tensor,
    num_diffusion_steps: int,
):
    """Return (alpha_t, dalpha_t) for timestep tensor t."""
    t_norm = t.float() / float(max(1, num_diffusion_steps))
    return get_noise_schedule(t_norm)


def corrupt_with_mask(
    x0: torch.Tensor,
    t: torch.Tensor,
    num_diffusion_steps: int,
    mask_token_id: int,
    min_mask_prob: float = 0.01,
    max_mask_prob: float = 0.99,
    mask_mode: str = "token",
    span_len: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt clean bytes by replacing random positions with mask token.

    Uses cosine-based masking schedule: at low t, few tokens are masked;
    at high t, most tokens are masked.

    Args:
        x0: Clean byte tokens [B, T].
        t: Timesteps [B], values in [1, num_diffusion_steps].
        num_diffusion_steps: Total diffusion steps.
        mask_token_id: Special mask token ID.
        min_mask_prob: Floor mask probability (default 0.01).
        max_mask_prob: Ceiling mask probability (default 0.99).
        mask_mode: "token" for random token masking, "span" for contiguous spans.
        span_len: Average span length for span masking.

    Returns:
        x_t: Corrupted sequence [B, T].
        mask: Boolean mask of corrupted positions [B, T].
    """
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


# ─── Loss ─────────────────────────────────────────────────────────────────────

def masked_denoise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha_t: torch.Tensor | None = None,
    dalpha_t: torch.Tensor | None = None,
    loss_type: str = "standard",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-entropy on masked positions, optionally weighted by noise schedule.

    Args:
        logits: Model predictions [B, T, V].
        targets: Clean byte tokens [B, T].
        mask: Boolean mask of corrupted positions [B, T].
        alpha_t: P(token stays clean) [B] or [B, 1] — used for vb weighting.
        dalpha_t: Derivative of alpha_t [B] or [B, 1] — used for vb weighting.
        loss_type: One of:
            - "uniform": equal weight per masked token (training default, stable).
            - "vb": weight by dalpha_t / (1 - alpha_t) (variational bound, higher variance).
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar loss.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if targets.ndim != 2 or mask.ndim != 2:
        raise ValueError("targets and mask must have shape [B, T]")

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)  # [B, T]

    mask_f = mask.float()

    if loss_type == "vb" and alpha_t is not None and dalpha_t is not None:
        # Weight from MDLM variational bound: loss_coeff = -dalpha / (1 - alpha).
        # dalpha <= 0 (alpha = P(clean) decreases with noise), so negate to keep
        # the NELBO weight positive.
        alpha = alpha_t.view(-1, 1) if alpha_t.dim() == 1 else alpha_t
        dalpha = dalpha_t.view(-1, 1) if dalpha_t.dim() == 1 else dalpha_t
        weight = -dalpha / (1 - alpha + eps)  # [B, 1]
        weighted = token_loss * mask_f * weight
        denom = (mask_f * weight).sum().clamp_min(eps)
    else:
        # Uniform: equal weight across all timesteps
        weighted = token_loss * mask_f
        denom = mask_f.sum().clamp_min(1.0)

    return weighted.sum() / denom
