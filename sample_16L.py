#!/home/peb/.venvs/vllm-py311/bin/python3
"""Generate samples from 16L checkpoints for qualitative comparison."""
import sys, os, math, torch, torch.nn.functional as F
sys.path.insert(0, '/home/peb/code/bbt')
from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
PROMPT = "Once upon a time"
MAX_NEW = 200
TEMP = 1.0

@torch.no_grad()
def generate(model, prompt_ids, max_new=MAX_NEW, temp=TEMP, is_ga=False):
    model.eval()
    ids = prompt_ids.clone()
    for _ in range(max_new):
        x = ids[-512:] if len(ids) > 512 else ids
        t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        logits = model(x.unsqueeze(0), t, is_causal=True)
        logits = logits[0, -1] / temp
        probs = F.softmax(logits[:256], dim=-1)  # exclude mask token
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id])
    return ids

# Encode prompt
prompt_bytes = PROMPT.encode('utf-8')
prompt_ids = torch.tensor(list(prompt_bytes), device=DEVICE, dtype=torch.long)

ckpts = [
    ("GA 16L dim=16", '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt', GA_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048, 'mv_dim': 16}),
    ("Vanilla 16L", '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt', Vanilla_Blockwise,
     {'n_layer': 16, 'd_model': 768, 'n_head': 8, 'd_ff': 2048}),
]

for name, ckpt_path, ModelClass, kwargs in ckpts:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ModelClass(**kwargs).to(DEVICE)
    model.load_state_dict(raw['model'])
    
    for seed in [42, 99, 123]:
        torch.manual_seed(seed)
        out = generate(model, prompt_ids, is_ga=(ModelClass is GA_Blockwise))
        text = bytes(out.cpu().tolist()).decode('utf-8', errors='replace')
        print(f"\n  [seed={seed}]")
        print(f"  {text}")
        print()
