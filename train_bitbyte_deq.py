#!/usr/bin/env python3
"""Train the experimental DEQ byte LM (BitByteDEQ).

Autoregressive next-byte training, mirroring train_bitbyte_diffusion.py's infra
(shard dataset, cosine LR, shared eval harness) but for the fixed-point model.
Logs solver convergence (iters / residual / converged) alongside loss/bpb so the
DEQ dynamics can be measured against a stacked baseline (train_bitbyte.py).

Default is full precision (nn.Linear) to de-risk the DEQ dynamics; pass
--quantize to switch the linears to BitNet ternary BitLinear.
"""
import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from data.byte_shard_dataset import ByteShardDataset
from models.bitbyte_deq import BitByteDEQ
from training.eval import run_ar_validation
from training.lr_scheduler import lr_for_step, seq_len_for_step
from utils.optim import create_grad_scaler, create_optimizer


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="Dir of training shard_*.bin")
    ap.add_argument("--out", type=str, default="artifacts/checkpoints/deq/ckpt_deq.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--n_prelude", type=int, default=2)
    ap.add_argument("--n_core", type=int, default=1)
    ap.add_argument("--n_coda", type=int, default=0)

    ap.add_argument("--solver", type=str, default="anderson", choices=["anderson", "fpi"])
    ap.add_argument("--max_iter", type=int, default=24)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--solver_beta", type=float, default=1.0)
    ap.add_argument("--anderson_m", type=int, default=5)
    ap.add_argument("--layer_scale_init", type=float, default=0.1)
    ap.add_argument("--gamma_max", type=float, default=1.0)

    ap.add_argument("--quantize", action="store_true",
                    help="Use BitNet ternary BitLinear instead of full-precision nn.Linear")
    ap.add_argument("--no_act_quant", action="store_true")
    ap.add_argument("--no_sdpa", action="store_true")
    ap.add_argument("--no_bnb", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=1024)

    ap.add_argument("--val_data", type=str, default=None)
    ap.add_argument("--val_every", type=int, default=0)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--val_seq_len", type=int, default=1024)

    ap.add_argument("--smoke_test", action="store_true",
                    help="Tiny CPU synthetic run for CI/testing")
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = build_args()

    if args.smoke_test:
        args.steps = min(args.steps, 10)
        args.log_every = 1
        args.save_every = args.steps + 1
        args.d_model = 64
        args.n_head = 4
        args.d_ff = 128
        args.n_prelude = 1
        args.n_core = 1
        args.n_coda = 0
        args.max_iter = 8
        args.batch_size = 2
        args.grad_accum = 1
        args.seq_len = 64
        args.no_sdpa = True

    if args.val_every < 0:
        raise ValueError("--val_every must be >= 0")
    if args.val_every > 0 and not args.val_data:
        raise ValueError("--val_data is required when --val_every > 0")
    if not args.smoke_test and not args.data:
        raise ValueError("--data is required unless --smoke_test is used")

    torch.manual_seed(args.seed)
    if args.smoke_test or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    out_parent = Path(args.out).parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.out}; use --allow_overwrite or a new --out.")

    model = BitByteDEQ(
        vocab_size=256,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        n_prelude=args.n_prelude,
        n_core=args.n_core,
        n_coda=args.n_coda,
        quantize=args.quantize,
        act_quant=not args.no_act_quant,
        use_sdpa=not args.no_sdpa,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        solver_beta=args.solver_beta,
        anderson_m=args.anderson_m,
        layer_scale_init=args.layer_scale_init,
        gamma_max=args.gamma_max,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt, use_bnb = create_optimizer(
        model, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
        use_bnb=(device.type == "cuda" and not args.no_bnb))
    scaler = create_grad_scaler()

    print(f"BitByteDEQ  params={n_params/1e6:.2f}M  quantize={args.quantize}  "
          f"prelude={args.n_prelude} core={args.n_core} coda={args.n_coda}  "
          f"solver={args.solver} max_iter={args.max_iter} tol={args.tol}")
    print(f"Optimizer: {'AdamW8bit' if use_bnb else 'torch AdamW'}  device={device}")

    if args.smoke_test:
        it = None
    else:
        ds = ByteShardDataset(
            shard_glob=os.path.join(args.data, "shard_*.bin"),
            seq_len=args.seq_len, seed=args.seed, rank=0, world_size=1)
        it = iter(torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=0, pin_memory=True))

    val_dl = None
    if args.val_data:
        val_ds = ByteShardDataset(
            shard_glob=os.path.join(args.val_data, "shard_*.bin"),
            seq_len=args.val_seq_len, seed=5678, rank=0, world_size=1)
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=1, num_workers=0, pin_memory=True)

    warmup_steps = int(args.steps * args.warmup_frac)
    amp_enabled = device.type == "cuda" and not args.no_amp

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_for_step(step, warmup_steps, args.steps, args.lr)

        opt.zero_grad(set_to_none=True)
        did_backward = False
        loss_sum = 0.0
        iters_sum = 0
        res_sum = 0.0
        conv_sum = 0

        for _ in range(args.grad_accum):
            if args.smoke_test:
                x = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device)
                y = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device)
            else:
                x, y = next(it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
                       if amp_enabled else nullcontext())
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1)) / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {step}, skipping microbatch")
                continue

            scaler.scale(loss).backward()
            did_backward = True
            loss_sum += float(loss.item()) * args.grad_accum
            info = model.last_info
            iters_sum += int(info.get("iters", 0))
            res_sum += float(info.get("rel_residual", 0.0))
            conv_sum += int(bool(info.get("converged", False)))

        if not did_backward:
            opt.zero_grad(set_to_none=True)
            continue

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            n = max(1, args.grad_accum)
            loss_avg = loss_sum / n
            bpb = loss_avg / math.log(2.0)
            dt = time.time() - t0
            print(f"step {step:6d}  seq {args.seq_len:4d}  loss {loss_avg:.4f}  "
                  f"bpb {bpb:.4f}  fp_iters {iters_sum / n:.1f}  "
                  f"fp_res {res_sum / n:.2e}  fp_conv {conv_sum}/{n}  {dt:.1f}s")
            t0 = time.time()

        if (step % args.save_every == 0) and step > 0:
            torch.save(
                {"step": step, "variant": "deq", "model": model.state_dict(),
                 "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "args": vars(args)},
                args.out)
            print(f"saved {args.out}")

        if args.val_every > 0 and (step % args.val_every == 0) and step > 0:
            res = run_ar_validation(
                model, val_dl, device, max_batches=args.val_batches,
                is_ddp=False, use_amp=amp_enabled)
            print(f"val  step {step:6d}  loss {res.loss:.4f}  bpb {res.bits_per_byte:.4f}  "
                  f"ppl {res.perplexity:.2f}  tokens {res.num_tokens}")


if __name__ == "__main__":
    main()
