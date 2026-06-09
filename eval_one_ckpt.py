#!/usr/bin/env python3
"""Compute held-out PPL/BPB for a single blockwise checkpoint with optional bootstrap CI.

Usage:
    python eval_one_ckpt.py /path/to/ckpt.pt [max_test_bytes] [--bootstrap N] [--seed 1234]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, '/home/peb/code/bbt')

import numpy as np
import torch
import torch.nn.functional as F

from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
SEQ_LEN = 512
STRIDE = 256
TEST_SHARD = '/home/peb/code/bbt/combined_shards/shard_ts_3.bin'
MAX_TEST_BYTES_DEFAULT = 200_000


@torch.no_grad()
def window_losses(model, data, seq_len=SEQ_LEN, stride=STRIDE):
    """Return per-window CE losses weighted by predicted token counts."""
    model.eval()
    losses = []
    weights = []
    for i in range(0, len(data) - seq_len, stride):
        x = data[i:i + seq_len].unsqueeze(0).to(DEVICE)
        t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        logits = model(x, t, is_causal=True)
        loss = F.cross_entropy(logits[0, :-1], x[0, 1:])
        losses.append(float(loss.item()))
        weights.append(seq_len - 1)
    if not losses:
        raise ValueError(f"not enough data for seq_len={seq_len}: got {len(data)} bytes")
    return np.asarray(losses, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def weighted_ce(losses, weights):
    return float(np.sum(losses * weights) / np.sum(weights))


def bootstrap_ci(losses, weights, n_boot=1000, seed=1234):
    rng = np.random.default_rng(seed)
    n = len(losses)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = weighted_ce(losses[idx], weights[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi), math.exp(float(lo)), math.exp(float(hi))


def infer_mv_dim(raw, ckpt_path):
    mv_dim = raw.get('mv_dim')
    if mv_dim is not None:
        return mv_dim
    stem = os.path.basename(ckpt_path)
    for part in stem.replace('.pt', '').split('_'):
        if part.startswith('dim') and part[3:].isdigit():
            return int(part[3:])
    return 8


def load_model(ckpt_path):
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    variant = raw.get('variant', 'ga')
    args = raw.get('args', {})

    if variant == 'ga':
        model = GA_Blockwise(
            n_layer=args['n_layer'], d_model=args['d_model'],
            n_head=args['n_head'], d_ff=args['d_ff'],
            num_diffusion_steps=64, mv_dim=infer_mv_dim(raw, ckpt_path),
        ).to(DEVICE)
    elif variant == 'vanilla':
        model = Vanilla_Blockwise(
            n_layer=args['n_layer'], d_model=args['d_model'],
            n_head=args['n_head'], d_ff=args['d_ff'],
            num_diffusion_steps=64,
        ).to(DEVICE)
    else:
        raise ValueError(f"unknown variant: {variant}")

    model.load_state_dict(raw['model'])
    model.eval()
    return raw, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt_path')
    ap.add_argument('max_test_bytes', nargs='?', type=int, default=MAX_TEST_BYTES_DEFAULT)
    ap.add_argument('--bootstrap', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f"CKPT_MISSING {args.ckpt_path}", flush=True)
        sys.exit(1)
    if not os.path.exists(TEST_SHARD):
        print(f"TEST_SHARD_MISSING {TEST_SHARD}", flush=True)
        sys.exit(1)

    raw, model = load_model(args.ckpt_path)
    data_np = np.fromfile(TEST_SHARD, dtype='uint8')[:args.max_test_bytes]
    data = torch.from_numpy(data_np).long()
    losses, weights = window_losses(model, data)
    bpb = weighted_ce(losses, weights)
    ppl = math.exp(bpb)

    step = raw.get('step', '?')
    tokens_seen = raw.get('tokens_seen', '?')
    print(
        f"  EVAL step={step} tokens={tokens_seen} bytes={len(data_np)} "
        f"windows={len(losses)} → PPL={ppl:.3f} BPB={bpb:.4f}",
        flush=True,
    )
    if args.bootstrap:
        lo_bpb, hi_bpb, lo_ppl, hi_ppl = bootstrap_ci(
            losses, weights, n_boot=args.bootstrap, seed=args.seed
        )
        print(
            f"  BOOTSTRAP n={args.bootstrap} seed={args.seed} "
            f"BPB95=[{lo_bpb:.4f},{hi_bpb:.4f}] "
            f"PPL95=[{lo_ppl:.3f},{hi_ppl:.3f}]",
            flush=True,
        )


if __name__ == '__main__':
    main()
