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
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from byte_shard_dataset import ByteShardDataset  # mmap-backed (efficient on large shards)
from models.bitbyte_deq import BitByteDEQ
from training.eval import run_ar_validation
from training.lr_scheduler import lr_for_step, seq_len_for_step
from utils.optim import create_grad_scaler, create_optimizer


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="Dir of training shard_*.bin")
    ap.add_argument("--out", type=str, default="artifacts/checkpoints/deq/ckpt_deq.pt")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    ap.add_argument("--no_opt_state", action="store_true",
                    help="When resuming, load model weights only (fresh optimizer/scaler)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_kv_head", type=int, default=None,
                    help="KV heads for GQA. None = MHA (n_head).")
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--n_prelude", type=int, default=2)
    ap.add_argument("--n_core", type=int, default=1)
    ap.add_argument("--n_coda", type=int, default=0)

    ap.add_argument("--solver", type=str, default="anderson", choices=["anderson", "fpi"])
    ap.add_argument("--max_iter", type=int, default=24)
    ap.add_argument("--tol", type=float, default=None,
                    help="Solver convergence tol. Default 1e-3 (fp) / 3e-3 (--quantize): "
                         "ternary quantization gives a residual noise floor ~2-3e-3, below "
                         "which the solver cannot converge, so a tighter tol false-alarms.")
    ap.add_argument("--solver_beta", type=float, default=1.0)
    ap.add_argument("--anderson_m", type=int, default=5)
    ap.add_argument("--layer_scale_init", type=float, default=0.1)
    ap.add_argument("--gamma_max", type=float, default=1.0)
    ap.add_argument("--jac_reg", type=float, default=0.0,
                    help="Frobenius Jacobian-reg weight (lambda * ||J_f||_F^2). 0 disables.")
    ap.add_argument("--spec_reg", type=float, default=0.0,
                    help="Spectral-norm reg weight: lambda * relu(sigma_max - margin)^2. 0 disables.")
    ap.add_argument("--spec_margin", type=float, default=0.9,
                    help="Contraction margin; penalize sigma_max above this.")
    ap.add_argument("--spec_iters", type=int, default=3,
                    help="Power-iteration steps for the sigma_max estimate.")
    ap.add_argument("--reg_every", type=int, default=1,
                    help="Apply the Jacobian/spectral penalty every N steps (amortizes cost).")

    ap.add_argument("--quantize", action="store_true",
                    help="Use BitNet ternary BitLinear instead of full-precision nn.Linear")
    ap.add_argument("--no_act_quant", action="store_true")
    ap.add_argument("--no_sdpa", action="store_true")
    ap.add_argument("--no_ckpt", action="store_true",
                    help="Disable gradient checkpointing in prelude/coda blocks")
    ap.add_argument("--no_bnb", action="store_true")
    ap.add_argument("--galore", action="store_true",
                    help="Use GaLore memory-efficient optimizer (requires galore-torch)")
    ap.add_argument("--galore_rank", type=int, default=128,
                    help="GaLore projection rank")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=None,
                    help="Fixed sequence length for all steps (disables curriculum). "
                         "Default: curriculum via seq_len_for_step capped by --seq_len_cap.")
    ap.add_argument("--seq_len_cap", type=int, default=1024,
                    help="Upper bound on curriculum sequence length (default 1024).")

    ap.add_argument("--val_data", type=str, default=None)
    ap.add_argument("--val_every", type=int, default=0)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--val_seq_len", type=int, default=1024)

    ap.add_argument("--smoke_test", action="store_true",
                    help="Tiny CPU synthetic run for CI/testing")
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

    if args.tol is None:
        # Ternary quantization makes the fixed-point map slightly non-smooth,
        # producing a residual noise floor ~2-3e-3; a tighter tol can never be
        # reached and the solver reports false non-convergence.
        args.tol = 3e-3 if args.quantize else 1e-3

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

    resume_same_out = (args.resume is not None
                       and Path(args.resume).resolve() == Path(args.out).resolve())
    out_parent = Path(args.out).parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists() and not args.allow_overwrite and not resume_same_out:
        raise FileExistsError(
            f"Refusing to overwrite {args.out}; use --allow_overwrite or a new --out.")

    torch.manual_seed(args.seed)
    if args.smoke_test or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model = BitByteDEQ(
        vocab_size=256,
        d_model=args.d_model,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
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
        ckpt=not args.no_ckpt,
        layer_scale_init=args.layer_scale_init,
        gamma_max=args.gamma_max,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt, use_bnb = create_optimizer(
        model, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
        use_bnb=(device.type == "cuda" and not args.no_bnb),
        use_galore=args.galore, galore_rank=args.galore_rank)
    scaler = create_grad_scaler()

    print(f"BitByteDEQ  params={n_params/1e6:.2f}M  quantize={args.quantize}  "
          f"prelude={args.n_prelude} core={args.n_core} coda={args.n_coda}  "
          f"solver={args.solver} max_iter={args.max_iter} tol={args.tol}")
    print(f"Optimizer: {'AdamW8bit' if use_bnb else 'torch AdamW'}  device={device}")

    start_step = 0
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint {args.resume!r}") from exc
        if ckpt.get("variant") not in (None, "deq"):
            raise ValueError(
                f"Checkpoint variant mismatch: expected 'deq', got {ckpt.get('variant')!r}")
        model.load_state_dict(ckpt["model"])
        if not args.no_opt_state:
            if "opt" not in ckpt or "scaler" not in ckpt:
                raise KeyError("Checkpoint missing optimizer/scaler state; use --no_opt_state.")
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt["scaler"])
        else:
            print("[WARN] --no_opt_state: resuming model weights with fresh optimizer/scaler.")
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    # Initial seq_len for dataset construction; updated each step for curriculum.
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
    reg_mode = "spec" if args.spec_reg > 0 else ("jac" if args.jac_reg > 0 else None)
    reg_lambda = args.spec_reg if reg_mode == "spec" else args.jac_reg

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

        do_reg = reg_mode is not None and (step % args.reg_every == 0)
        opt.zero_grad(set_to_none=True)
        did_backward = False
        loss_sum = 0.0
        reg_sum = 0.0
        iters_sum = 0
        res_sum = 0.0
        conv_sum = 0

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
                if do_reg:
                    logits, reg_loss, _ = model(
                        x, reg=reg_mode, reg_margin=args.spec_margin,
                        reg_iters=args.spec_iters)
                    ce = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                    loss = (ce + reg_lambda * reg_loss) / args.grad_accum
                    reg_sum += float(reg_loss.item())
                else:
                    logits = model(x)
                    ce = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                    loss = ce / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {step}, skipping microbatch")
                continue

            scaler.scale(loss).backward()
            did_backward = True
            loss_sum += float(ce.item())
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
            if reg_mode == "spec":
                reg_str = (f"sig_mean {model.last_sigma:.3f}  "
                           f"sig_max {model.last_sigma_max:.3f}  "
                           f"reg {reg_sum / n:.3f}  ")
            elif reg_mode == "jac":
                reg_str = f"jac {reg_sum / n:.3f}  "
            else:
                reg_str = ""
            print(f"step {step:6d}  seq {seq_len:4d}  loss {loss_avg:.4f}  "
                  f"bpb {bpb:.4f}  {reg_str}fp_iters {iters_sum / n:.1f}  "
                  f"fp_res {res_sum / n:.2e}  fp_conv {conv_sum}/{n}  {dt:.1f}s")
            t0 = time.time()

        if (step % args.save_every == 0) and step > 0:
            _save_atomic(
                {"step": step, "variant": "deq", "model": model.state_dict(),
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
