#!/usr/bin/env python3
"""Compute test-set perplexity on a held-out shard."""
import sys, os, math
sys.path.insert(0, '/home/peb/code/bbt')

import torch
import torch.nn.functional as F
from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
SEQ_LEN = 512
TEST_SHARD = '/home/peb/code/bbt/combined_shards/shard_ts_3.bin'
MAX_TEST_BYTES = 200_000  # 200KB slice for 16L models (keeps eval fast)
STRIDE = 256  # overlapping windows for long-form PPL

ckpts = [
    # 2L ablation
    ('GA dim=4 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim4.pt', GA_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256, 'mv_dim': 4}),
    ('GA dim=8 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim8.pt', GA_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256, 'mv_dim': 8}),
    ('GA dim=16 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim16.pt', GA_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256, 'mv_dim': 16}),
    ('Vanilla 2L', '/home/peb/data/bbt_checkpoints/blockwise_2L_vanilla.pt', Vanilla_Blockwise,
     {'n_layer': 2, 'd_model': 128, 'n_head': 4, 'd_ff': 256}),
    # 16L comparison
    ('GA 16L (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt', GA_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'mv_dim': 8}),
    ('Vanilla 16L (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt', Vanilla_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048}),
]

@torch.no_grad()
def sliding_ppl(model, data, seq_len=SEQ_LEN, stride=STRIDE):
    """Compute perplexity using sliding window (like GPT-2 eval)."""
    model.eval()
    nll = 0.0
    n_tokens = 0
    for i in range(0, len(data) - seq_len, stride):
        x = data[i:i+seq_len].unsqueeze(0).to(DEVICE)
        t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        logits = model(x, t, is_causal=True)
        loss = F.cross_entropy(logits[0, :-1], x[0, 1:])
        n_tokens += seq_len - 1
        nll += loss.item() * (seq_len - 1)
    return math.exp(nll / n_tokens), nll / n_tokens

# Load test data
data = torch.from_numpy(
    __import__('numpy').fromfile(TEST_SHARD, dtype='uint8')[:MAX_TEST_BYTES]
).long()
print(f"Test data: {TEST_SHARD} ({len(data):,} bytes, {len(data)/1e6:.1f}MB slice)")

for name, ckpt_path, ModelClass, model_args in ckpts:
    print(f"\n  Loading {name} from {ckpt_path}")
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if ModelClass is GA_Blockwise:
        model = ModelClass(**model_args).to(DEVICE)
    else:
        model = ModelClass(**model_args).to(DEVICE)
    
    model.load_state_dict(raw['model'])
    model.eval()
    
    ppl, bpb = sliding_ppl(model, data)
    print(f"  {name}:  PPL={ppl:.2f}  BPB={bpb:.4f}")
