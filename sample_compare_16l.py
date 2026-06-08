#!/home/peb/.venvs/vllm-py311/bin/python
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/home/peb/code/bbt')
from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise
DEVICE='cuda'; TEMP=0.85; TOP_K=40; MAX_NEW=220
PROMPTS=['Once upon a time','The meaning of life is','In a shocking finding, scientists discovered that']
SEEDS=[99]
CKPTS=[
 ('GA 16L dim16 1B', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt', GA_Blockwise),
 ('Vanilla 16L 82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt', Vanilla_Blockwise),
 ('GA 16L dim8 82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt', GA_Blockwise),
]
def clean(xs): return bytes([int(x) for x in xs if 0 <= int(x) < 256]).decode('utf-8', errors='replace').replace('\x00','␀')
@torch.no_grad()
def gen(model,prompt,seed):
    torch.manual_seed(seed)
    ids=torch.tensor(list(prompt.encode()),device=DEVICE,dtype=torch.long)
    for _ in range(MAX_NEW):
        x=ids[-512:]
        with torch.amp.autocast('cuda'):
            logits=model(x.unsqueeze(0), torch.zeros(1,dtype=torch.long,device=DEVICE), is_causal=True)[0,-1,:256].float()/TEMP
        vals,_=torch.topk(logits,TOP_K); logits[logits<vals[-1]]=-float('inf')
        ids=torch.cat([ids, torch.multinomial(F.softmax(logits,dim=-1),1)])
    return clean(ids.cpu().tolist())
for name,path,Cls in CKPTS:
    raw=torch.load(path,map_location='cpu',weights_only=False); args=raw.get('args',{})
    kwargs=dict(n_layer=args.get('n_layer',16), d_model=args.get('d_model',768), n_head=args.get('n_head',8), d_ff=args.get('d_ff',2048))
    if Cls is GA_Blockwise: kwargs['mv_dim']=raw.get('mv_dim') or (16 if 'dim16' in path else 8)
    model=Cls(**kwargs).to(DEVICE); model.load_state_dict(raw['model']); model.eval()
    print('\n'+'='*88); print(name, 'step', raw.get('step'), 'tokens', raw.get('tokens_seen'), 'clean', raw.get('loss_clean'), 'block', raw.get('block_ce'))
    for prompt in PROMPTS:
        for seed in SEEDS:
            print('\nPROMPT',repr(prompt),'seed',seed)
            print(gen(model,prompt,seed))
    del model; torch.cuda.empty_cache()
