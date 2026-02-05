# train_bitbyte.py
import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from byte_shard_dataset import ByteShardDataset

# -------------------------
# Ternary quant + STE
# -------------------------
os.environ["USE_LIBUV"] = "0"

class TernaryQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, eps: float = 1e-8):
        # absmean scaling (BitNet b1.58 style)
        gamma = w.abs().mean()
        gamma = torch.clamp(gamma, min=1e-6)
        w_scaled = w / gamma

        wq = torch.clamp(w_scaled, -1.0, 1.0).round()  # -> {-1,0,+1}
        ctx.save_for_backward(w_scaled)
        # Re-apply scale so the quantized weight keeps the original magnitude.
        return wq * gamma

    @staticmethod
    def backward(ctx, grad_out):
        (w_scaled,) = ctx.saved_tensors
        # STE: pass gradients through; optional clamp mask for stability:
        mask = (w_scaled.abs() <= 1.0).to(grad_out.dtype)
        return grad_out * mask, None


def act_quant_per_token(x, q=127, eps=1e-8):
    # Symmetric per-row quantization on last dim.
    # x: [..., C]
    x2 = x.reshape(-1, x.shape[-1])
    s = x2.abs().amax(dim=-1, keepdim=True) / q
    s = torch.clamp(s, min=eps)
    xq = torch.clamp((x2 / s).round(), -q, q)
    return (xq * s).reshape_as(x)


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, act_quant=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.act_quant = act_quant
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        if self.act_quant:
            x = act_quant_per_token(x)
        wq = TernaryQuantSTE.apply(self.weight)
        return F.linear(x, wq, self.bias)

# -------------------------
# LLaMA-ish blocks
# -------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / \
            (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, positions):
        # x: [B, H, T, D]
        # positions: [T]
        freqs = torch.einsum("t,d->td", positions.float(),
                             self.inv_freq)  # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
        cos = emb.cos()[None, None, :, :]  # [1,1,T,D]
        sin = emb.sin()[None, None, :, :]
        return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, rope_base=10000.0, act_quant=True, use_sdpa=True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.use_sdpa = use_sdpa

        self.q_proj = BitLinear(
            d_model, d_model, bias=False, act_quant=act_quant)
        self.k_proj = BitLinear(
            d_model, d_model, bias=False, act_quant=act_quant)
        self.v_proj = BitLinear(
            d_model, d_model, bias=False, act_quant=act_quant)
        self.o_proj = BitLinear(
            d_model, d_model, bias=False, act_quant=act_quant)

        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(self, x):
        # x: [B,T,C]
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head,
                                self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_head,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head,
                                self.head_dim).transpose(1, 2)

        pos = torch.arange(T, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch SDPA (may use flash on supported configs)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True)  # [B,H,T,D]
        else:
            # fallback attention
            att = (q @ k.transpose(-2, -1)) / \
                math.sqrt(self.head_dim)  # [B,H,T,T]
            att = att.masked_fill(torch.triu(torch.ones(
                T, T, device=x.device), 1).bool(), float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, act_quant=True):
        super().__init__()
        # SwiGLU uses 2*d_ff for gate+up
        self.gate_up = BitLinear(
            d_model, 2 * d_ff, bias=False, act_quant=act_quant)
        self.down = BitLinear(d_ff, d_model, bias=False, act_quant=act_quant)
        self.act = SwiGLU()

    def forward(self, x):
        return self.down(self.act(self.gate_up(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, act_quant=True, rope_base=10000.0, use_sdpa=True, ckpt=True):
        super().__init__()
        self.ckpt = ckpt
        self.n1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_head, rope_base=rope_base, act_quant=act_quant, use_sdpa=use_sdpa)
        self.n2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, act_quant=act_quant)

    def forward(self, x):
        def attn_fn(x):
            return x + self.attn(self.n1(x))

        def mlp_fn(x):
            return x + self.mlp(self.n2(x))

        if self.ckpt and self.training:
            x = checkpoint(attn_fn, x, use_reentrant=False)
            x = checkpoint(mlp_fn, x, use_reentrant=False)
        else:
            x = attn_fn(x)
            x = mlp_fn(x)
        return x


class BitByteLM(nn.Module):
    def __init__(self, vocab_size=256, n_layer=24, d_model=1536, n_head=12, d_ff=4096,
                 act_quant=True, rope_base=10000.0, use_sdpa=True, ckpt=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Lower init scale keeps logits in a sane range for fp16.
        nn.init.normal_(self.tok_emb.weight, mean=0.0,
                        std=1.0 / math.sqrt(d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff, act_quant=act_quant,
                  rope_base=rope_base, use_sdpa=use_sdpa, ckpt=ckpt)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx):
        # idx: [B,T] bytes
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B,T,256]
        return logits

# -------------------------
# Byte dataset (streaming from a single binary file)
# -------------------------


class ByteStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, seq_len, seed=1234, rank=0, world_size=1):
        super().__init__()
        self.path = path
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank)

        with open(self.path, "rb") as f:
            data = f.read()

        n = len(data)
        # shard by rank: each rank samples offsets in a disjoint strided pattern
        stride = self.world_size

        # infinite stream
        i = self.rank
        while True:
            # pseudo-random-ish offset
            off = int(torch.randint(
                0, n - (self.seq_len + 1), (1,), generator=g).item())
            off = (off + i) % (n - (self.seq_len + 1))
            chunk = data[off: off + self.seq_len + 1]
            x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)
            yield x, y
            i += stride

# -------------------------
# Train
# -------------------------

def setup_ddp(use_ddp: bool):
    if not use_ddp:
        # Single-process (no torchrun)
        return 0, 0, 1, False

    backend = "nccl" if os.name != "nt" else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size(), True


# def setup_ddp():
#     dist.init_process_group(backend="nccl")
#     rank = dist.get_rank()
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     return rank, local_rank, dist.get_world_size()


def bits_per_byte(loss):
    return float(loss.item() / math.log(2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="Path to a large binary file of raw bytes")
    ap.add_argument("--out", type=str, default="ckpt.pt")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--n_layer", type=int, default=24)
    ap.add_argument("--d_model", type=int, default=1536)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=4096)

    ap.add_argument("--ddp", action="store_true", help="Enable DDP (requires torchrun)")

    ap.add_argument("--use_sdpa", action="store_true")
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--no_act_quant", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=1,
                    help="microbatch sequences per GPU")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=None,
                    help="Fixed sequence length for all steps (disables curriculum).")
    ap.add_argument("--seq_len_cap", type=int, default=None,
                    help="Upper bound applied to curriculum sequence length.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume (loads model/opt/scaler/args.step).")
    ap.add_argument("--no_opt_state", action="store_true",
                    help="When resuming, load model weights only (fresh optimizer/scaler).")

    args = ap.parse_args()

    rank, local_rank, world_size, is_ddp = setup_ddp(args.ddp)
    device = torch.device("cuda", local_rank)


    # Curriculum schedule (by step fraction)
    def seq_len_for_step(step):
        # If user fixes seq len, skip curriculum
        if args.seq_len is not None:
            return args.seq_len
        frac = step / max(1, args.steps)
        if frac < 0.2:
            return 2048
        elif frac < 0.5:
            return 4096
        else:
            return 8192
    # Optional cap
    if args.seq_len_cap is not None:
        _orig = seq_len_for_step
        def seq_len_for_step(step):
            return min(_orig(step), args.seq_len_cap)

    model = BitByteLM(
        vocab_size=256,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        act_quant=not args.no_act_quant,
        use_sdpa=args.use_sdpa,
        ckpt=not args.no_ckpt,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


    # Optimizer (bnb 8-bit if available; falls back to torch)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(
            0.9, 0.95), weight_decay=args.wd)
        if rank == 0:
            print("Using bitsandbytes AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(
            0.9, 0.95), weight_decay=args.wd)
        if rank == 0:
            print("Using torch AdamW (bnb not available)")

    scaler = torch.amp.GradScaler()

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["model"]
        if is_ddp:
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

        if not args.no_opt_state:
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt["scaler"])
            start_step = int(ckpt.get("step", 0)) + 1
        else:
            # Fresh optimizer/scaler; keep step for logging/decay
            start_step = int(ckpt.get("step", 0)) + 1
            if rank == 0:
                print("Skipping optimizer/scaler state (fresh opt).")

        if rank == 0:
            print(f"Resumed from {args.resume} at step {start_step}")

    # Data
    # ds = ByteStreamDataset(args.data, seq_len=2048, rank=rank, world_size=world_size)
    ds = ByteShardDataset(
        shard_glob=os.path.join(args.data, "shard_*.bin"),
        seq_len=2048,
        seed=1234,
        rank=rank,
        world_size=world_size,
    )

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    warmup_steps = int(args.steps * args.warmup_frac)

    def lr_for_step(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        # cosine decay to 10% of lr
        t = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))

    model.train()
    t0 = time.time()
    it = iter(dl)

    for step in range(start_step, args.steps):
        seq_len = seq_len_for_step(step)
        ds.set_seq_len(seq_len)

        for g in opt.param_groups:
            g["lr"] = lr_for_step(step)

        opt.zero_grad(set_to_none=True)

        did_backward = False
        total_loss = 0.0

        for micro in range(args.grad_accum):
            x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                loss = loss / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {step}, skipping microbatch {micro}")
                continue

            scaler.scale(loss).backward()
            did_backward = True
            total_loss += float(loss.item())

        if not did_backward:
            # nothing to step; move on
            opt.zero_grad(set_to_none=True)
            continue

        # clip
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            # all-reduce loss for reporting
            loss_avg = total_loss
            if is_ddp:
                loss_t = torch.tensor([total_loss], device=device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                loss_avg = loss_t.item() / world_size

            if rank == 0:
                dt = time.time() - t0
                bpb = loss_avg / math.log(2.0)
                print(
                    f"step {step:6d}  seq {seq_len:4d}  loss {loss_avg:.4f}  bpb {bpb:.4f}  {dt:.1f}s")
            t0 = time.time()

        if rank == 0 and (step % args.save_every == 0) and step > 0:
            # Save non-DDP state
            ckpt = {
                "step": step,
                "model": (model.module.state_dict() if is_ddp else model.state_dict()),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, args.out)
            print(f"saved {args.out}")

    if is_ddp:
        dist.destroy_process_group()



if __name__ == "__main__":
    main()
