#!/usr/bin/env python3
"""Sample from preserved 16L @82M checkpoints and compare.

For matched 82M + 1B samples written to markdown, use sample_compare_16l.py.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/peb/code/bbt')
from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
MAX_NEW = 384
TEMPERATURE = 0.8
TOP_K = 40
PROMPTS = [
    b"Once upon a time, ",
    b"She looked at the door and ",
    b"The scientist explained that ",
]

ckpts = [
    ('Vanilla 16L @82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla_82M_backup.pt', Vanilla_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'num_diffusion_steps': 64}),
    ('GA 16L dim8 @82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt', GA_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'num_diffusion_steps': 64, 'mv_dim': 8}),
]


@torch.no_grad()
def generate(model, prompt_bytes, max_new=MAX_NEW, temp=TEMPERATURE, top_k=TOP_K):
    model.eval()
    generated = list(prompt_bytes)
    for _ in range(max_new):
        x = torch.tensor([generated[-512:]], dtype=torch.long, device=DEVICE)
        t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        logits = model(x, t, is_causal=True)
        next_logits = logits[0, -1, :256] / temp
        if top_k > 0:
            vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < vals[-1]] = float('-inf')
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
    return bytes(generated).decode('utf-8', errors='replace')


for name, ckpt_path, ModelClass, model_args in ckpts:
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"step={raw.get('step')} tokens={raw.get('tokens_seen')}")
    model = ModelClass(**model_args).to(DEVICE)
    model.load_state_dict(raw['model'])
    model.eval()
    for prompt in PROMPTS:
        ptext = prompt.decode('utf-8', errors='replace')
        text = generate(model, prompt)
        print(f"\n  ── {ptext} ──")
        print(f"  {text[:400]}")
    del model
    torch.cuda.empty_cache()
