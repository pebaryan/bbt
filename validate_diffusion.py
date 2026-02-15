#!/usr/bin/env python3
"""Validation script for Diffusion models."""

import argparse
import math
from contextlib import nullcontext

import torch

from data.byte_shard_dataset import ByteShardDataset
from models.bitbyte_diffusion import BitByteDiffusionLM
from training.diffusion import corrupt_with_mask, masked_denoise_loss, sample_timesteps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--data",
        type=str,
        default="artifacts/datasets/misc_shards/shard_*.bin",
        help="Shard glob or single shard path",
    )
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    ckpt = torch.load(args.ckpt, map_location=device)

    # Extract model config from checkpoint
    cfg = ckpt.get("args", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__

    ds = ByteShardDataset(args.data, args.seq_len, seed=0, rank=0, world_size=1)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)

    # Get diffusion steps from checkpoint or use default
    diffusion_steps = cfg.get("diffusion_steps", 24)
    mask_token_id = cfg.get("mask_token_id", 256)

    model = BitByteDiffusionLM(
        vocab_size=256,
        n_layer=cfg.get("n_layer", 6),
        d_model=cfg.get("d_model", 320),
        n_head=cfg.get("n_head", 5),
        d_ff=cfg.get("d_ff", 640),
        num_diffusion_steps=diffusion_steps,
        mask_token_id=mask_token_id,
        act_quant=not cfg.get("no_act_quant", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tot_loss = 0.0
    tot_tokens = 0
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        for i, (x, y) in enumerate(dl):
            if i >= args.batches:
                break
            x = x.to(device)  # shape [1, seq_len]
            y = y.to(device)  # shape [1, seq_len]

            # For diffusion models, compute denoising loss
            # Sample a timestep
            t = sample_timesteps(x.size(0), diffusion_steps, device)

            # Corrupt the input
            x_t, mask = corrupt_with_mask(x, t, diffusion_steps, mask_token_id)

            # Get model predictions
            logits = model(x_t, t)  # shape [1, seq_len, vocab_size]

            # Compute masked denoising loss
            loss = masked_denoise_loss(logits, x, mask)

            tot_loss += float(loss.item())
            tot_tokens += int(mask.sum().item())

    bpb = (tot_loss / max(1, tot_tokens)) / math.log(2.0)
    print(
        f"val bpb {bpb:.4f} on {tot_tokens} masked tokens (diffusion_steps={diffusion_steps})"
    )


if __name__ == "__main__":
    main()
