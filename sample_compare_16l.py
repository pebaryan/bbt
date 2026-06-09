#!/home/peb/.venvs/vllm-py311/bin/python
"""Generate matched samples for 16L 82M and 1B checkpoints."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/peb/code/bbt')
from experiments.train_2L_blockwise import GA_Blockwise, Vanilla_Blockwise

DEVICE = 'cuda'
TEMP = 0.85
TOP_K = 40
MAX_NEW = 220
PROMPTS = [
    'Once upon a time',
    'The meaning of life is',
    'In a shocking finding, scientists discovered that',
]
SEEDS = [99]
OUT = Path('/home/peb/code/bbt/matched_16l_samples.md')

CKPTS = [
    ('Vanilla 16L @82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla_82M_backup.pt', Vanilla_Blockwise, None),
    # The 82M GA dim=16 checkpoint was overwritten by the resumed 1B run; keep dim8 as the preserved GA 82M reference.
    ('GA 16L dim8 @82M', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt', GA_Blockwise, 8),
    ('Vanilla 16L @1B', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt', Vanilla_Blockwise, None),
    ('GA 16L dim16 @1B', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt', GA_Blockwise, 16),
]


def clean(xs):
    return bytes([int(x) for x in xs if 0 <= int(x) < 256]).decode('utf-8', errors='replace').replace('\x00', '␀')


@torch.no_grad()
def gen(model, prompt, seed):
    torch.manual_seed(seed)
    ids = torch.tensor(list(prompt.encode()), device=DEVICE, dtype=torch.long)
    for _ in range(MAX_NEW):
        x = ids[-512:]
        with torch.amp.autocast('cuda'):
            logits = model(
                x.unsqueeze(0),
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                is_causal=True,
            )[0, -1, :256].float() / TEMP
        vals, _ = torch.topk(logits, TOP_K)
        logits[logits < vals[-1]] = -float('inf')
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)])
    return clean(ids.cpu().tolist())


def load_model(path, cls, mv_dim_override):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    args = raw.get('args', {})
    kwargs = dict(
        n_layer=args.get('n_layer', 16),
        d_model=args.get('d_model', 768),
        n_head=args.get('n_head', 8),
        d_ff=args.get('d_ff', 2048),
        num_diffusion_steps=64,
    )
    if cls is GA_Blockwise:
        kwargs['mv_dim'] = raw.get('mv_dim') or mv_dim_override or 8
    model = cls(**kwargs).to(DEVICE)
    model.load_state_dict(raw['model'])
    model.eval()
    return raw, model


def main():
    lines = [
        '# Matched 16L Samples',
        '',
        f'Sampling: temp={TEMP}, top_k={TOP_K}, max_new={MAX_NEW}, seeds={SEEDS}',
        '',
        '> Note: the 82M GA dim=16 checkpoint was overwritten by the resumed 1B run; GA dim8 @82M is included as the preserved GA 82M reference.',
        '',
    ]
    for name, path, cls, mv_dim in CKPTS:
        raw, model = load_model(path, cls, mv_dim)
        lines.extend([
            f'## {name}',
            '',
            f'- checkpoint: `{path}`',
            f'- step: `{raw.get("step")}`',
            f'- tokens: `{raw.get("tokens_seen")}`',
            f'- clean_ce: `{raw.get("loss_clean")}`',
            f'- block_ce: `{raw.get("block_ce")}`',
            '',
        ])
        for prompt in PROMPTS:
            for seed in SEEDS:
                text = gen(model, prompt, seed)
                lines.extend([
                    f'### prompt={prompt!r}, seed={seed}',
                    '',
                    '```text',
                    text,
                    '```',
                    '',
                ])
        del model
        torch.cuda.empty_cache()
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
