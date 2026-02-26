#!/usr/bin/env python3
"""Validation script for BitByteLM checkpoint."""

import argparse
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.bitbytelm import BitByteLM
from data.byte_shard_dataset import ByteShardDataset


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})
    if not isinstance(cfg, dict):
        cfg = vars(cfg)

    model = BitByteLM(
        vocab_size=256,
        n_layer=cfg.get("n_layer", 24),
        d_model=cfg.get("d_model", 1536),
        n_head=cfg.get("n_head", 12),
        d_ff=cfg.get("d_ff", 4096),
        act_quant=not cfg.get("no_act_quant", False),
        use_sdpa=cfg.get("use_sdpa", True),
        ckpt=not cfg.get("no_ckpt", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def validate(model, dataloader, device, max_batches=0):
    model.eval()
    tot_loss = 0.0
    tot_tokens = 0
    use_amp = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    with torch.no_grad(), amp_ctx:
        for i, (x, y) in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1), reduction="sum")
            tot_loss += float(loss.item())
            tot_tokens += int(y.numel())

    bpb = (tot_loss / max(1, tot_tokens)) / math.log(2.0)
    return bpb, tot_loss / tot_tokens, tot_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=str, default="artifacts/checkpoints/ckpt_transformer-24.pt"
    )
    ap.add_argument(
        "--data", type=str, default="artifacts/datasets/tinystories/shards/data"
    )
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for validation data"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, cfg = load_model(args.ckpt, device)
    print(f"Loaded checkpoint from step {cfg.get('step', '?')}")
    print(f"Model: {cfg.get('n_layer', 24)} layers, {cfg.get('d_model', 1536)} d_model")

    ds = ByteShardDataset(
        shard_glob=f"{args.data}/shard_*.bin",
        seq_len=args.seq_len,
        seed=args.seed,
        rank=0,
        world_size=1,
    )
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    bpb, loss, tokens = validate(model, dl, device, max_batches=args.batches)
    print(f"\nValidation Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  BPB: {bpb:.4f}")
    print(f"  Tokens: {tokens}")


if __name__ == "__main__":
    main()
