#!/home/peb/.venvs/vllm-py311/bin/python3
"""
MDLM-style baseline — full-sequence masked diffusion with cosine schedule.
Same backbone as blockwise models, but masks random tokens across the
entire sequence rather than in blocks. No GA embeddings.

This answers: 'Does the blockwise objective add value over standard
masked diffusion for this architecture?'
"""
import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/peb/code/bbt')
from models.rmsnorm import RMSNorm
from models.rope import RoPE
from experiments.ga_diffusion import cosine_mask_schedule

MASK_TOKEN_ID = 256


class MDLM_Transformer(nn.Module):
    """Full-sequence masked diffusion model — same backbone as blockwise."""
    def __init__(self, n_layer=2, d_model=128, n_head=4, d_ff=256,
                 num_diffusion_steps=64):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_ff = d_ff
        self.n_layer = n_layer
        self.num_diffusion_steps = num_diffusion_steps

        self.embed = nn.Embedding(257, d_model)
        self.rope = RoPE(head_dim=self.head_dim, base=10000.0)
        blk = []
        for _ in range(n_layer):
            blk.append(nn.ModuleDict({
                'n1': RMSNorm(d_model), 'n2': RMSNorm(d_model),
                'q_proj': nn.Linear(d_model, d_model, bias=False),
                'k_proj': nn.Linear(d_model, d_model, bias=False),
                'v_proj': nn.Linear(d_model, d_model, bias=False),
                'o_proj': nn.Linear(d_model, d_model, bias=False),
                'gate_up': nn.Linear(d_model, 2 * d_ff, bias=False),
                'down': nn.Linear(d_ff, d_model, bias=False),
            }))
        self.blocks = nn.ModuleList(blk)
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, 257, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, x, attn_mask=None, is_causal=False):
        h = self.embed(x)
        B, L, D = h.shape
        for blk in self.blocks:
            n = blk['n1'](h)
            q = blk['q_proj'](n).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
            k = blk['k_proj'](n).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
            v = blk['v_proj'](n).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
            pos = torch.arange(L, device=h.device)
            q = self.rope(q, pos); k = self.rope(k, pos)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            attn = attn.transpose(1, 2).contiguous().view(B, L, D)
            h = h + blk['o_proj'](attn)
            n2 = blk['n2'](h)
            gate, up = blk['gate_up'](n2).chunk(2, dim=-1)
            h = h + blk['down'](F.silu(gate) * up)
        h = self.norm_f(h)
        return self.lm_head(h)


def train(n_layer=2, d_model=128, n_head=4, d_ff=256,
          steps=15000, batch_size=64, grad_accum=2, seq_len=512, lr=3e-4,
          diffusion_steps=64, seed=1234):
    label = f"mdlm_{n_layer}L_{d_model}d"
    if seed != 1234:
        label += f"_s{seed}"
    OUT = f'/home/peb/data/bbt_checkpoints/{label}.pt'
    LOG = OUT.replace('.pt', '.log')
    log_every = max(200, min(2000, steps // 75))
    save_every = max(2000, steps // 3)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    model = MDLM_Transformer(n_layer=n_layer, d_model=d_model,
                             n_head=n_head, d_ff=d_ff,
                             num_diffusion_steps=diffusion_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MDLM BASELINE] {n_layer}L/{d_model}d  params={n_params/1e6:.2f}M")
    print(f"  Steps={steps}  Batch={batch_size}×{grad_accum}={batch_size*grad_accum}  "
          f"Tok/step={batch_size*grad_accum*seq_len:,}")
    print(f"  Diffusion steps={diffusion_steps}")
    print(f"  Total tokens={steps * batch_size * grad_accum * seq_len / 1e9:.1f}B")
    print(f"  Logging to {LOG}")

    from data.byte_shard_dataset import ByteShardDataset
    ds = ByteShardDataset(
        '/home/peb/code/bbt/combined_shards/shard_*.bin', seq_len,
        seed=seed, rank=0, world_size=1)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler('cuda')

    step = 0
    data_iter = iter(dl)
    opt.zero_grad()
    t_start = time.time()

    while step < steps:
        for micro in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            B, L = x.shape

            with torch.amp.autocast('cuda'):
                # Clean loss (causal) — same as blockwise models
                logits_clean = model(x, is_causal=True)
                loss_clean = F.cross_entropy(
                    logits_clean[:, :-1].reshape(-1, 257), x[:, 1:].reshape(-1))
                
                # Sample noise levels and full-sequence mask
                t = torch.randint(1, diffusion_steps + 1, (B,), device=device)
                mask_prob = cosine_mask_schedule(t.float(), diffusion_steps)
                xb = x.clone()
                n_masked_total = 0
                for b in range(B):
                    p = mask_prob[b].item()
                    n_mask = max(1, int(L * p))
                    idx = torch.randperm(L, device=device)[:n_mask]
                    xb[b, idx] = MASK_TOKEN_ID
                    n_masked_total += n_mask

                # Bidirectional forward pass for diffusion
                logits_diff = model(xb)
                is_masked = (xb == MASK_TOKEN_ID)
                if is_masked.any():
                    loss_diff = F.cross_entropy(
                        logits_diff.view(-1, 257), x.view(-1),
                        reduction='none').view(B, L)
                    loss_diff = (loss_diff * is_masked).sum() / (is_masked.sum() + 1e-8)
                else:
                    loss_diff = torch.tensor(0.0, device=device)

                total_loss = (loss_clean + 0.5 * loss_diff) / grad_accum

            scaler.scale(total_loss).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        opt.zero_grad()
        step += 1

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            sps = step / elapsed if elapsed > 0 else 0
            tok_ps = batch_size * grad_accum * seq_len
            wps = step * tok_ps / elapsed / 1e6 if elapsed > 0 else 0
            lr_now = scheduler.get_last_lr()[0]
            line = (f"step {step:5d}  "
                    f"clean_ce {float(loss_clean.detach().cpu()):.4f}  "
                    f"diff_ce {float(loss_diff.detach().cpu()):.4f}  "
                    f"masked {n_masked_total}  "
                    f"lr {lr_now:.2e}  {elapsed:.0f}s  {sps:.2f}sp/s  {wps:.2f}Mtok/s")
            print(line)
            with open(LOG, 'a') as f:
                f.write(line + '\n')

        if step % save_every == 0 or step == steps:
            torch.save({
                'step': step, 'model': model.state_dict(),
                'variant': 'mdlm',
                'args': {'n_layer': n_layer, 'd_model': d_model,
                        'n_head': n_head, 'd_ff': d_ff,
                        'diffusion_steps': diffusion_steps},
                'clean_ce': float(loss_clean.detach().cpu()),
                'diff_ce': float(loss_diff.detach().cpu()),
                'tokens_seen': step * batch_size * grad_accum * seq_len,
            }, OUT)
            print(f"  saved {OUT}")

    total_time = time.time() - t_start
    print(f"\nDone. {total_time:.0f}s = {total_time/60:.1f}min. Final: {OUT}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-layer', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-head', type=int, default=4)
    ap.add_argument('--d-ff', type=int, default=256)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--diffusion-steps', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()
    train(
        n_layer=args.n_layer, d_model=args.d_model,
        n_head=args.n_head, d_ff=args.d_ff,
        steps=args.steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, seq_len=args.seq_len,
        lr=args.lr, diffusion_steps=args.diffusion_steps,
        seed=args.seed,
    )
