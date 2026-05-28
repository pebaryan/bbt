"""Shared validation harness for byte-level models (1 token == 1 byte).

Provides a results dataclass plus reusable validation loops for the
autoregressive and masked-diffusion variants. Because the models operate on
raw bytes, bits-per-byte is simply loss / ln(2) and token count == byte count.
"""
import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class ValLossResults:
    loss: float          # mean per-token NLL in nats
    perplexity: float
    bits_per_byte: float
    num_tokens: int
    accuracy: float | None = None  # masked-token accuracy (diffusion only)


def _autocast(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _all_reduce_sum(device, nll_sum, tok_count, correct):
    stats = torch.tensor(
        [nll_sum, float(tok_count), float(correct)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item()), int(stats[2].item())


def _finalize(nll_sum, tok_count, correct) -> ValLossResults:
    if tok_count <= 0:
        acc = None if correct is None else float("nan")
        return ValLossResults(float("nan"), float("nan"), float("nan"), 0, acc)
    loss = nll_sum / tok_count
    ppl = math.exp(loss) if loss < 100 else float("inf")
    bpb = loss / math.log(2.0)  # byte-level: 1 token == 1 byte
    acc = None if correct is None else correct / tok_count
    return ValLossResults(
        loss=loss, perplexity=ppl, bits_per_byte=bpb,
        num_tokens=tok_count, accuracy=acc,
    )


@torch.no_grad()
def run_ar_validation(
    model, dataloader, device, *, max_batches, is_ddp, use_amp=True,
) -> ValLossResults:
    """Autoregressive next-byte validation (mean per-token NLL / ppl / bpb)."""
    if max_batches <= 0:
        return _finalize(0.0, 0, None)

    was_training = model.training
    model.eval()
    nll_sum = 0.0
    tok_count = 0
    try:
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _autocast(device, use_amp):
                logits = model(x)
                ce_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )
            nll_sum += float(ce_sum.item())
            tok_count += int(y.numel())
    finally:
        if was_training:
            model.train()

    if is_ddp:
        nll_sum, tok_count, _ = _all_reduce_sum(device, nll_sum, tok_count, 0)
    return _finalize(nll_sum, tok_count, None)


@torch.no_grad()
def run_diffusion_validation(
    model, dataloader, device, *, max_batches, is_ddp,
    diffusion_steps, mask_token_id, min_mask_prob, max_mask_prob,
    mask_mode, span_len, use_amp=True, seed=12345,
) -> ValLossResults:
    """Masked-denoise validation: mean NLL over masked tokens + masked accuracy.

    Corruption and timestep sampling are seeded (and the global RNG state is
    restored afterwards) so the metric is comparable across calls and does not
    perturb the training RNG stream.
    """
    # Imported here to avoid a circular import (training.diffusion is heavy).
    from training.diffusion import corrupt_with_mask, sample_timesteps

    if max_batches <= 0:
        return _finalize(0.0, 0, 0)

    was_training = model.training
    model.eval()

    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    torch.manual_seed(seed)

    nll_sum = 0.0
    tok_count = 0
    correct = 0
    try:
        for i, (x, _y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x0 = x.to(device, non_blocking=True)
            t = sample_timesteps(
                x0.size(0), diffusion_steps, device=device, antithetic=True,
            )
            x_t, mask = corrupt_with_mask(
                x0=x0, t=t, num_diffusion_steps=diffusion_steps,
                mask_token_id=mask_token_id,
                min_mask_prob=min_mask_prob, max_mask_prob=max_mask_prob,
                mask_mode=mask_mode, span_len=span_len,
            )
            with _autocast(device, use_amp):
                logits = model(x_t, t)
                ce_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    x0.view(-1),
                    reduction="none",
                ).view_as(x0)
            mask_f = mask.float()
            nll_sum += float((ce_tok * mask_f).sum().item())
            tok_count += int(mask.sum().item())
            correct += int(((logits.argmax(dim=-1) == x0) & mask).sum().item())
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if was_training:
            model.train()

    if is_ddp:
        nll_sum, tok_count, correct = _all_reduce_sum(
            device, nll_sum, tok_count, correct,
        )
    return _finalize(nll_sum, tok_count, correct)
