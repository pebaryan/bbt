#!/home/peb/.venvs/vllm-py311/bin/python
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/home/peb/code/bbt')
from experiments.train_2L_blockwise import GA_Blockwise

DEVICE='cuda'
CKPT='/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt'
PROMPTS=[
    'Once upon a time',
    'The meaning of life is',
    'In a shocking finding, scientists discovered that',
]
SEEDS=[42,99,123]
MAX_NEW=220
TEMP=0.85
TOP_K=40

def sanitize(bs):
    text=bytes([b for b in bs if 0 <= b < 256]).decode('utf-8', errors='replace')
    # keep telegram readable
    return text.replace('\x00','␀')

@torch.no_grad()
def generate(model, prompt, seed):
    torch.manual_seed(seed)
    ids=torch.tensor(list(prompt.encode('utf-8')), device=DEVICE, dtype=torch.long)
    for _ in range(MAX_NEW):
        x=ids[-512:]
        t=torch.zeros(1, dtype=torch.long, device=DEVICE)
        with torch.amp.autocast('cuda'):
            logits=model(x.unsqueeze(0), t, is_causal=True)[0, -1, :256].float()/TEMP
        if TOP_K:
            vals,_=torch.topk(logits, TOP_K)
            logits[logits < vals[-1]] = -float('inf')
        probs=F.softmax(logits, dim=-1)
        nxt=torch.multinomial(probs, 1)
        ids=torch.cat([ids, nxt])
    return sanitize(ids.detach().cpu().tolist())

raw=torch.load(CKPT, map_location='cpu', weights_only=False)
args=raw.get('args', {})
model=GA_Blockwise(n_layer=args.get('n_layer',16), d_model=args.get('d_model',768), n_head=args.get('n_head',12), d_ff=args.get('d_ff',2048), mv_dim=raw.get('mv_dim',16)).to(DEVICE)
model.load_state_dict(raw['model'], strict=True)
model.eval()
print(f"ckpt={CKPT}")
print(f"step={raw.get('step')} tokens={raw.get('tokens_seen')} clean_ce={raw.get('loss_clean'):.4f} block_ce={raw.get('block_ce'):.4f}")
print(f"sampling: temp={TEMP}, top_k={TOP_K}, max_new={MAX_NEW}\n")
for prompt in PROMPTS:
    print('='*80)
    print(f"PROMPT: {prompt!r}")
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        print(generate(model, prompt, seed))
        torch.cuda.empty_cache()
    print()
