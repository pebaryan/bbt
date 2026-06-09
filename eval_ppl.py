#!/usr/bin/env python3
"""Compute test-set perplexity on held-out shard for available checkpoints.

For the focused 16L 1B comparison with bootstrap CIs, prefer:
    python eval_one_ckpt.py /path/to/checkpoint.pt 5000000 --bootstrap 500
"""
import math
import sys

sys.path.insert(0, '/home/peb/code/bbt')

import numpy as np
import torch
import torch.nn.functional as F

from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise
from experiments.train_ar_ce_baseline import AR_Transformer

DEVICE = 'cuda'
SEQ_LEN = 512
TEST_SHARD = '/home/peb/code/bbt/combined_shards/shard_ts_3.bin'
MAX_TEST_BYTES = 200_000
STRIDE = 256

ckpts = [
    ('GA dim=4 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim4.pt', GA_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256, 'mv_dim': 4}),
    ('GA dim=8 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim8.pt', GA_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256, 'mv_dim': 8}),
    ('GA dim=16 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim16.pt', GA_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256, 'mv_dim': 16}),
    ('Vanilla 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_vanilla.pt', Vanilla_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256}),
    ('Vanilla 16L @82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla_82M_backup.pt', Vanilla_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048}),
    ('GA 16L dim8 @82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt', GA_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'mv_dim': 8}),
    ('Vanilla 16L @1B', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt', Vanilla_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048}),
    ('GA 16L dim16 @1B', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt', GA_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'mv_dim': 16}),
    ('AR CE baseline 2L', '/home/peb/data/bbt_checkpoints/ce_baseline_ar_2L_128d.pt', AR_Transformer,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256}),
]


@torch.no_grad()
def sliding_ppl(model, data, seq_len=SEQ_LEN, stride=STRIDE, is_ar=False):
    model.eval()
    nll = 0.0
    n_tokens = 0
    for i in range(0, len(data) - seq_len, stride):
        x = data[i:i + seq_len].unsqueeze(0).to(DEVICE)
        if is_ar:
            logits = model(x)
        else:
            t = torch.zeros(1, dtype=torch.long, device=DEVICE)
            logits = model(x, t, is_causal=True)
        loss = F.cross_entropy(logits[0, :-1], x[0, 1:])
        n_tokens += seq_len - 1
        nll += loss.item() * (seq_len - 1)
    return math.exp(nll / n_tokens), nll / n_tokens


data = torch.from_numpy(np.fromfile(TEST_SHARD, dtype='uint8')[:MAX_TEST_BYTES]).long()
print(f"Test data: {TEST_SHARD} ({len(data):,} bytes, {len(data) / 1e6:.1f}MB slice)")

for name, ckpt_path, ModelClass, model_args in ckpts:
    print(f"\n  Loading {name} from {ckpt_path}")
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ModelClass(**model_args).to(DEVICE)
    model.load_state_dict(raw['model'])
    model.eval()
    ppl, bpb = sliding_ppl(model, data, is_ar=(ModelClass is AR_Transformer))
    print(f"  {name}: step={raw.get('step')} tokens={raw.get('tokens_seen')} PPL={ppl:.3f} BPB={bpb:.4f}")
    del model
    torch.cuda.empty_cache()
