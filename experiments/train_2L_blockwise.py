#!/home/peb/.venvs/vllm-py311/bin/python3
"""
Blockwise diffusion — compare GA vs vanilla embeddings, 2L/128d to 1B tokens.
"""
import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/peb/code/bbt')
from experiments.ga_diffusion import GAEmbedding, GADecoder, geometric_product, grade_norms
from experiments.ga_diffusion import cosine_mask_schedule

MASK_TOKEN_ID = 256

# ── Models ───────────────────────────────────────────────────────────────────

class GA_Blockwise(nn.Module):
    """GA embedding + decoder blockwise model."""
    def __init__(self, n_layer=2, d_model=128, n_head=4, d_ff=256, num_diffusion_steps=64, mv_dim=8):
        super().__init__()
        self.mv_dim = mv_dim
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_ff = d_ff
        self.n_layer = n_layer
        self.num_diffusion_steps = num_diffusion_steps

        self.ga_embed = GAEmbedding(257, self.mv_dim)
        self.ga_decode = GADecoder(self.mv_dim, 257)
        self.proj_up = nn.Linear(self.mv_dim, d_model)
        self.proj_down = nn.Linear(d_model, self.mv_dim)

        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        from models.rmsnorm import RMSNorm
        from models.rope import RoPE
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

    def _sinu_embed(self, t):
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        angles = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, x_t, t, attn_mask=None, is_causal=False):
        mv = self.ga_embed(x_t)
        h = self.proj_up(mv)
        # t_emb unused (matches original GAStructuredLM behavior)
        for blk in self.blocks:
            n = blk['n1'](h)
            B, L, D = n.shape
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
        mv_out = self.proj_down(h)
        return self.ga_decode(mv_out)


class Vanilla_Blockwise(nn.Module):
    """Standard embedding + LM head, same transformer backbone."""
    def __init__(self, n_layer=2, d_model=128, n_head=4, d_ff=256, num_diffusion_steps=64):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_ff = d_ff
        self.n_layer = n_layer
        self.num_diffusion_steps = num_diffusion_steps

        self.embed = nn.Embedding(257, d_model)
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        from models.rmsnorm import RMSNorm
        from models.rope import RoPE
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
        self.lm_head.weight = self.embed.weight  # tied

    def _sinu_embed(self, t):
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        angles = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, x_t, t, attn_mask=None, is_causal=False):
        h = self.embed(x_t)
        # t_emb unused (matches original GAStructuredLM behavior)
        for blk in self.blocks:
            n = blk['n1'](h)
            B, L, D = n.shape
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


# ── Training ─────────────────────────────────────────────────────────────────

def train(variant='ga', n_layer=2, d_model=128, n_head=4, d_ff=256,
          steps=15000, batch_size=64, grad_accum=2, seq_len=512, lr=3e-4,
          block_size=8, num_blocks=2, block_weight=0.5, mv_dim=8, seed=1234):
    suffix = f"_{variant}"
    if variant == 'ga' and mv_dim != 8:
        suffix += f"_dim{mv_dim}"
    if seed != 1234:
        suffix += f"_s{seed}"
    OUT = f'/home/peb/data/bbt_checkpoints/blockwise_{n_layer}L{suffix}.pt'
    LOG = OUT.replace('.pt', '.log')
    diffusion_steps = 64
    log_every = max(200, min(2000, steps // 75))  # ~75 logs per run
    save_every = max(2000, steps // 3)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # Model
    if variant == 'ga':
        model = GA_Blockwise(n_layer=n_layer, d_model=d_model, n_head=n_head, d_ff=d_ff,
                             num_diffusion_steps=diffusion_steps, mv_dim=mv_dim).to(device)
    else:
        model = Vanilla_Blockwise(n_layer=n_layer, d_model=d_model, n_head=n_head, d_ff=d_ff,
                                  num_diffusion_steps=diffusion_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    label = f"{n_layer}L/{d_model}d"
    print(f"[{variant.upper()}] {label}  params={n_params/1e6:.2f}M")
    print(f"  Steps={steps}  Batch={batch_size}×{grad_accum}={batch_size*grad_accum}  "
          f"Tok/step={batch_size*grad_accum*seq_len:,}")
    print(f"  Total tokens={steps * batch_size * grad_accum * seq_len / 1e9:.1f}B")
    print(f"  Logging to {LOG}")

    # Data
    from data.byte_shard_dataset import ByteShardDataset
    ds = ByteShardDataset(
        '/home/peb/code/bbt/combined_shards/shard_*.bin', seq_len,
        seed=seed, rank=0, world_size=1)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    # AMP scaler
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
                # Clean loss (causal)
                logits_clean = model(x, torch.zeros(B, dtype=torch.long, device=device), is_causal=True)
                loss_clean = F.cross_entropy(
                    logits_clean[:, :-1].reshape(-1, 257), x[:, 1:].reshape(-1))

                # Block diffusion loss
                loss_block_total = 0.0
                num_masked = 0
                for bi in range(num_blocks):
                    bs = min(block_size, L // 4)
                    bstart = torch.randint(0, L - bs + 1, (1,)).item()
                    t = torch.randint(1, diffusion_steps + 1, (B,), device=device)

                    mask_prob = cosine_mask_schedule(t.float(), diffusion_steps)
                    xb = x.clone()
                    for b in range(B):
                        p = mask_prob[b].item() if mask_prob.dim() > 0 else mask_prob.item()
                        n_mask = max(1, int(bs * p))
                        idx = torch.randperm(bs, device=device)[:n_mask]
                        xb[b, bstart:bstart + bs][idx] = MASK_TOKEN_ID

                    logits_block = model(xb, t)
                    is_masked = (xb[:, bstart:bstart + bs] == MASK_TOKEN_ID)
                    if is_masked.any():
                        ce = F.cross_entropy(
                            logits_block[:, bstart:bstart + bs].reshape(-1, 257),
                            x[:, bstart:bstart + bs].reshape(-1),
                            reduction='none').reshape(B, -1)
                        loss_block = (ce * is_masked.float()).sum() / (is_masked.sum() + 1e-8)
                        loss_block_total = loss_block_total + loss_block
                        num_masked += is_masked.sum().item()

                avg_block = loss_block_total / max(num_blocks, 1) if num_masked > 0 else torch.tensor(0.0, device=device)
                total_loss = (loss_clean + block_weight * avg_block) / grad_accum

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
                    f"block_ce {float(avg_block.detach().cpu()) if num_masked > 0 else 0:.4f}  "
                    f"masked {num_masked}  "
                    f"lr {lr_now:.2e}  {elapsed:.0f}s  {sps:.2f}sp/s  {wps:.2f}Mtok/s")
            print(line)
            with open(LOG, 'a') as f:
                f.write(line + '\n')

        if step % save_every == 0 or step == steps:
            torch.save({
                'step': step, 'model': model.state_dict(),
                'variant': variant, 'args': {'n_layer': n_layer, 'd_model': d_model,
                                            'n_head': n_head, 'd_ff': d_ff},
                'loss_clean': float(loss_clean.detach().cpu()),
                'block_ce': float(avg_block.detach().cpu()),
                'tokens_seen': step * batch_size * grad_accum * seq_len,
            }, OUT)
            print(f"  saved {OUT}")

    total_time = time.time() - t_start
    print(f"\nDone. {total_time:.0f}s = {total_time/60:.1f}min. Final: {OUT}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['ga', 'vanilla'])
    ap.add_argument('--n-layer', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-head', type=int, default=4)
    ap.add_argument('--d-ff', type=int, default=256)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--block-size', type=int, default=8)
    ap.add_argument('--num-blocks', type=int, default=2)
    ap.add_argument('--block-weight', type=float, default=0.5)
    ap.add_argument('--mv-dim', type=int, default=8, help='GA multivector dimension (GA variant only)')
    ap.add_argument('--seed', type=int, default=1234, help='RNG seed for data ordering')
    args = ap.parse_args()
    train(
        variant=args.variant,
        n_layer=args.n_layer, d_model=args.d_model,
        n_head=args.n_head, d_ff=args.d_ff,
        steps=args.steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, seq_len=args.seq_len,
        lr=args.lr, block_size=args.block_size,
        num_blocks=args.num_blocks, block_weight=args.block_weight,
        mv_dim=args.mv_dim, seed=args.seed,
    )
