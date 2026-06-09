#!/home/peb/.venvs/vllm-py311/bin/python3
"""Generate samples from 16L 1B checkpoints for qualitative comparison."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/peb/code/bbt')
from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
PROMPT = "Once upon a time"
MAX_NEW = 200
TEMP = 1.0


@torch.no_grad()
def generate(model, prompt_ids, max_new=MAX_NEW, temp=TEMP):
    model.eval()
    ids = prompt_ids.clone()
    for _ in range(max_new):
        x = ids[-512:] if len(ids) > 512 else ids
        t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        logits = model(x.unsqueeze(0), t, is_causal=True)
        logits = logits[0, -1, :256] / temp
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id])
    return ids


prompt_ids = torch.tensor(list(PROMPT.encode('utf-8')), device=DEVICE, dtype=torch.long)

ckpts = [
    ("GA 16L dim16 @1B", '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt', GA_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'num_diffusion_steps': 64, 'mv_dim': 16}),
    ("Vanilla 16L @1B", '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt', Vanilla_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'num_diffusion_steps': 64}),
]

for name, ckpt_path, ModelClass, kwargs in ckpts:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"step={raw.get('step')} tokens={raw.get('tokens_seen')}")
    model = ModelClass(**kwargs).to(DEVICE)
    model.load_state_dict(raw['model'])
    model.eval()
    for seed in [42, 99, 123]:
        torch.manual_seed(seed)
        out = generate(model, prompt_ids)
        text = bytes(out.cpu().tolist()).decode('utf-8', errors='replace')
        print(f"\n  [seed={seed}]")
        print(f"  {text}")
        print()
    del model
    torch.cuda.empty_cache()
