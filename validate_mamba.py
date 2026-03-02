#!/usr/bin/env python3
"""Validation script for Mamba models."""

import argparse
import math
from contextlib import nullcontext

import torch

from data.byte_shard_dataset import ByteShardDataset
from models.mamba_mlm import MambaMLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--data",
        type=str,
        default="artifacts/datasets/misc_shards/shard_*.bin",
        help="Shard glob or single shard path",
    )
    ap.add_argument("--seq_len", type=int, default=2048)
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

    model = MambaMLM(
        vocab_size=256,
        n_layer=cfg.get("n_layer", 6),
        d_model=cfg.get("d_model", 384),
        d_ff=cfg.get("d_ff", 768),
        d_state=cfg.get("d_state", 16),
        d_conv=cfg.get("d_conv", 4),
        expand=cfg.get("expand", 2),
        act_quant=not cfg.get("no_act_quant", False),
        time_step_min=cfg.get("time_step_min", 1e-3),
        time_step_max=cfg.get("time_step_max", 1e-1),
        dt_init=cfg.get("dt_init", "log_uniform"),
        a_init=cfg.get("a_init", "uniform_0_16"),
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
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 256),
                y.view(-1),
                reduction="sum",
            )
            tot_loss += float(loss.item())
            tot_tokens += int(y.numel())

    bpb = (tot_loss / max(1, tot_tokens)) / math.log(2.0)
    print(f"val bpb {bpb:.4f} on {tot_tokens} tokens")


if __name__ == "__main__":
    main()
