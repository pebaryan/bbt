#!/usr/bin/env python3
"""Train a parameter-matched AR baseline for DEQ comparison.

Mirrors train_bitbyte_deq.py exactly (args, data pipeline, logging format)
so the two trainers are directly comparable. Default config produces a
~33.7M-param model matching the DEQ checkpoints:
    n_layer=8  d_model=512  n_head=8  d_ff=2048

Run with the same flags used for the DEQ to get a fair comparison:
    python train_ar_baseline.py \\
        --data artifacts/datasets/fineweb_edu_4gb/shards/train \\
        --val_data artifacts/datasets/fineweb_edu_4gb/shards/val \\
        --steps 5000 --batch_size 8 --seq_len 512 --lr 6e-4 --no_amp
"""
import argparse
import math
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from byte_shard_dataset import ByteShardDataset
from models.bitbytelm import BitByteLM
from training.eval import run_ar_validation
from training.lr_scheduler import lr_for_step, seq_len_for_step
from utils.optim import create_grad_scaler, create_optimizer


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=None, help="Dir of training shard_*.bin")
    ap.add_argument("--out", type=str,
                    default="artifacts/checkpoints/ar_baseline/ckpt_ar_baseline.pt")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--no_opt_state", action="store_true")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)

    # Model — defaults match DEQ checkpoint config.
    ap.add_argument("--n_layer", type=int, default=8,
                    help="AR depth. 8 layers ≈ 33.7M params (matches DEQ 4+2+2 config)")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_kv_head", type=int, default=None,
                    help="KV heads for GQA. None = MHA.")
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--no_act_quant", action="store_true")
    ap.add_argument("--no_ckpt", action="store_true",
                    help="Disable gradient checkpointing (faster fwd, more VRAM)")
    ap.add_argument("--no_sdpa", action="store_true")
    ap.add_argument("--no_bnb", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--galore", action="store_true",
                    help="Use GaLore memory-efficient optimizer (requires galore-torch)")
    ap.add_argument("--galore_rank", type=int, default=128,
                    help="GaLore projection rank")

    # Training — defaults match DEQ.
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=None,
                    help="Fixed seq len (disables curriculum). Default: curriculum capped by --seq_len_cap.")
    ap.add_argument("--seq_len_cap", type=int, default=512,
                    help="Curriculum cap (default 512, matching DEQ training seq_len).")

    ap.add_argument("--val_data", type=str, default=None)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--val_seq_len", type=int, default=512)

    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def _save_atomic(ckpt: dict, path: str) -> None:
    out_parent = os.path.dirname(path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=out_parent or ".")
    os.close(fd)
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    args = build_args()

    if args.smoke_test:
        args.steps = min(args.steps, 10)
        args.log_every = 1
        args.save_every = args.steps + 1
        args.n_layer = 2
        args.d_model = 64
        args.n_head = 4
        args.d_ff = 128
        args.batch_size = 2
        args.grad_accum = 1
        args.seq_len = 64
        args.no_sdpa = True

    if not args.smoke_test and not args.data:
        raise ValueError("--data is required unless --smoke_test is used")
    if args.val_every > 0 and not args.val_data and not args.smoke_test:
        raise ValueError("--val_data is required when --val_every > 0")

    resume_same_out = (args.resume is not None
                       and Path(args.resume).resolve() == Path(args.out).resolve())
    out_parent = Path(args.out).parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists() and not args.allow_overwrite and not resume_same_out:
        raise FileExistsError(
            f"Refusing to overwrite {args.out}; use --allow_overwrite or a new --out.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BitByteLM(
        vocab_size=256,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        d_ff=args.d_ff,
        act_quant=not args.no_act_quant,
        use_sdpa=not args.no_sdpa,
        ckpt=not args.no_ckpt,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt, use_bnb = create_optimizer(
        model, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
        use_bnb=(device.type == "cuda" and not args.no_bnb),
        use_galore=args.galore, galore_rank=args.galore_rank)
    scaler = create_grad_scaler()

    print(f"BitByteLM (AR baseline)  params={n_params/1e6:.2f}M  "
          f"n_layer={args.n_layer}  d_model={args.d_model}  d_ff={args.d_ff}")
    print(f"Optimizer: {'AdamW8bit' if use_bnb else 'torch AdamW'}  device={device}")

    start_step = 0
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint {args.resume!r}") from exc
        if ckpt.get("variant") not in (None, "ar"):
            raise ValueError(
                f"Checkpoint variant mismatch: expected 'ar', got {ckpt.get('variant')!r}")
        model.load_state_dict(ckpt["model"])
        if not args.no_opt_state:
            if "opt" not in ckpt or "scaler" not in ckpt:
                raise KeyError("Checkpoint missing optimizer/scaler state; use --no_opt_state.")
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt["scaler"])
        else:
            print("[WARN] --no_opt_state: fresh optimizer/scaler.")
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    init_seq_len = args.seq_len or seq_len_for_step(0, args.steps, cap=args.seq_len_cap)

    ds = None
    it = None
    if not args.smoke_test:
        ds = ByteShardDataset(
            shard_glob=os.path.join(args.data, "shard_*.bin"),
            seq_len=init_seq_len, seed=args.seed, rank=0, world_size=1)
        it = iter(torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=0, pin_memory=True))

    val_dl = None
    if args.val_data:
        val_ds = ByteShardDataset(
            shard_glob=os.path.join(args.val_data, "shard_*.bin"),
            seq_len=args.val_seq_len, seed=5678, rank=0, world_size=1)
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    warmup_steps = int(args.steps * args.warmup_frac)
    amp_enabled = device.type == "cuda" and not args.no_amp

    model.train()
    t0 = time.time()
    for step in range(start_step, args.steps):
        seq_len = (args.seq_len
                   if args.seq_len is not None
                   else seq_len_for_step(step, args.steps, cap=args.seq_len_cap))
        if ds is not None:
            ds.set_seq_len(seq_len)

        for g in opt.param_groups:
            g["lr"] = lr_for_step(step, warmup_steps, args.steps, args.lr)

        opt.zero_grad(set_to_none=True)
        did_backward = False
        loss_sum = 0.0

        for _ in range(args.grad_accum):
            if args.smoke_test:
                x = torch.randint(0, 256, (args.batch_size, seq_len), device=device)
                y = torch.randint(0, 256, (args.batch_size, seq_len), device=device)
            else:
                x, y = next(it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
                       if amp_enabled else nullcontext())
            with amp_ctx:
                logits = model(x)
                ce = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                loss = ce / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {step}, skipping microbatch")
                continue

            scaler.scale(loss).backward()
            did_backward = True
            loss_sum += float(ce.item())

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
            print(f"step {step:6d}  seq {seq_len:4d}  loss {loss_avg:.4f}  "
                  f"bpb {bpb:.4f}  {dt:.1f}s")
            t0 = time.time()

        if (step % args.save_every == 0) and step > 0:
            _save_atomic(
                {"step": step, "variant": "ar", "model": model.state_dict(),
                 "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "args": vars(args)},
                args.out)
            print(f"saved {args.out}")

        if args.val_every > 0 and (step % args.val_every == 0) and step > 0:
            model.eval()
            res = run_ar_validation(
                model, val_dl, device, max_batches=args.val_batches,
                is_ddp=False, use_amp=amp_enabled)
            model.train()
            print(f"val  step {step:6d}  loss {res.loss:.4f}  bpb {res.bits_per_byte:.4f}  "
                  f"ppl {res.perplexity:.2f}  tokens {res.num_tokens}")


if __name__ == "__main__":
    main()
