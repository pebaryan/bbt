#!/usr/bin/env python3
"""Validation script for Diffusion models."""
import argparse

import torch

from data.byte_shard_dataset import ByteShardDataset
from models.bitbyte_diffusion import BitByteDiffusionLM
from training.eval import run_diffusion_validation


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
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP (fp16)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Extract model config from checkpoint
    cfg = ckpt.get("args", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__

    ds = ByteShardDataset(args.data, args.seq_len, seed=0, rank=0, world_size=1)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)

    # Get diffusion config from checkpoint
    diffusion_steps = cfg.get("diffusion_steps", cfg.get("num_diffusion_steps", 24))
    mask_token_id = cfg.get("mask_token_id", 256)
    mask_mode = cfg.get("mask_mode", "token")
    span_len = float(cfg.get("span_len", 8.0))
    min_mask_prob = float(cfg.get("min_mask_prob", 0.01))
    max_mask_prob = float(cfg.get("max_mask_prob", 0.99))

    model = BitByteDiffusionLM(
        vocab_size=256,
        n_layer=cfg.get("n_layer", 6),
        d_model=cfg.get("d_model", 320),
        n_head=cfg.get("n_head", 5),
        d_ff=cfg.get("d_ff", 640),
        num_diffusion_steps=diffusion_steps,
        act_quant=not cfg.get("no_act_quant", False),
        use_sdpa=cfg.get("use_sdpa", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    res = run_diffusion_validation(
        model, dl, device,
        max_batches=args.batches,
        is_ddp=False,
        diffusion_steps=diffusion_steps,
        mask_token_id=mask_token_id,
        min_mask_prob=min_mask_prob,
        max_mask_prob=max_mask_prob,
        mask_mode=mask_mode,
        span_len=span_len,
        use_amp=use_amp,
    )
    print(
        f"val bpb {res.bits_per_byte:.4f} on {res.num_tokens} masked tokens "
        f"(diffusion_steps={diffusion_steps}, mask_mode={mask_mode}, "
        f"mask_range=[{min_mask_prob:.2f},{max_mask_prob:.2f}])"
    )
    print(f"masked acc {res.accuracy:.4%}")


if __name__ == "__main__":
    main()