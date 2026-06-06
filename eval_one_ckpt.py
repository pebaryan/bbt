#!/usr/bin/env python3
"""Compute test-set PPL/BPB for a single blockwise checkpoint.

Reads the saved dict to recover the model class, hyperparams, and the
amount of data seen so far. Prints one concise line suitable for grep /
cron consumption.

Usage:
    python eval_one_ckpt.py /path/to/ckpt.pt
"""
import sys
import os
import math

sys.path.insert(0, '/home/peb/code/bbt')

import torch
import torch.nn.functional as F
import numpy as np

from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
SEQ_LEN = 512
STRIDE = 256
TEST_SHARD = '/home/peb/code/bbt/combined_shards/shard_ts_3.bin'
# 200KB for 16L (keeps eval fast); bigger for 2L where the model is cheap.
MAX_TEST_BYTES_DEFAULT = 200_000


@torch.no_grad()
def sliding_ppl(model, data, seq_len=SEQ_LEN, stride=STRIDE):
    model.eval()
    nll = 0.0
    n_tokens = 0
    for i in range(0, len(data) - seq_len, stride):
        x = data[i:i + seq_len].unsqueeze(0).to(DEVICE)
        t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        logits = model(x, t, is_causal=True)
        loss = F.cross_entropy(logits[0, :-1], x[0, 1:])
        n_tokens += seq_len - 1
        nll += loss.item() * (seq_len - 1)
    return math.exp(nll / n_tokens), nll / n_tokens


def main():
    if len(sys.argv) < 2:
        print("usage: eval_one_ckpt.py <ckpt.pt> [max_test_bytes]", file=sys.stderr)
        sys.exit(2)

    ckpt_path = sys.argv[1]
    max_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_TEST_BYTES_DEFAULT

    if not os.path.exists(ckpt_path):
        print(f"CKPT_MISSING {ckpt_path}", flush=True)
        sys.exit(1)

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    variant = raw.get('variant', 'ga')
    args = raw.get('args', {})

    if variant == 'ga':
        # Ckpts saved before 2026-06-05 don't have an explicit mv_dim key.
        # Fall back to parsing the suffix ("_dim16") from the filename.
        mv_dim = raw.get('mv_dim')
        if mv_dim is None:
            stem = os.path.basename(ckpt_path)
            for part in stem.replace('.pt', '').split('_'):
                if part.startswith('dim') and part[3:].isdigit():
                    mv_dim = int(part[3:])
                    break
        if mv_dim is None:
            mv_dim = 8  # last-resort default
        model = GA_Blockwise(
            n_layer=args['n_layer'], d_model=args['d_model'],
            n_head=args['n_head'], d_ff=args['d_ff'],
            num_diffusion_steps=64, mv_dim=mv_dim,
        ).to(DEVICE)
    elif variant == 'vanilla':
        model = Vanilla_Blockwise(
            n_layer=args['n_layer'], d_model=args['d_model'],
            n_head=args['n_head'], d_ff=args['d_ff'],
            num_diffusion_steps=64,
        ).to(DEVICE)
    else:
        print(f"UNKNOWN_VARIANT {variant}", flush=True)
        sys.exit(1)

    model.load_state_dict(raw['model'])

    data = torch.from_numpy(
        np.fromfile(TEST_SHARD, dtype='uint8')[:max_bytes]
    ).long()
    ppl, bpb = sliding_ppl(model, data)

    step = raw.get('step', '?')
    tokens_seen = raw.get('tokens_seen', '?')
    print(f"  EVAL step={step} tokens={tokens_seen} → PPL={ppl:.3f} BPB={bpb:.4f}")


if __name__ == '__main__':
    main()
