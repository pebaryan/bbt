#!/usr/bin/env python3
"""Train a masked-diffusion byte denoiser without changing AR model paths."""

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from data.byte_shard_dataset import ByteShardDataset
from models.bitbyte_diffusion import BitByteDiffusionLM
from training.ddp import setup_ddp
from training.diffusion import corrupt_with_mask, masked_denoise_loss, sample_timesteps
from training.lr_scheduler import lr_for_step, seq_len_for_step
from utils.optim import create_grad_scaler, create_optimizer


def create_model(args: argparse.Namespace, device: torch.device) -> BitByteDiffusionLM:
    model = BitByteDiffusionLM(
        vocab_size=256,
        mask_token_id=args.mask_token_id,
        num_diffusion_steps=args.diffusion_steps,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        act_quant=not args.no_act_quant,
        use_sdpa=args.use_sdpa,
        ckpt=not args.no_ckpt,
    ).to(device)
    return model


def create_dataset(
    args: argparse.Namespace, rank: int, world_size: int
) -> ByteShardDataset:
    initial_seq_len = args.seq_len if args.seq_len is not None else 2048
    return ByteShardDataset(
        shard_glob=os.path.join(args.data, "shard_*.bin"),
        seq_len=initial_seq_len,
        seed=1234,
        rank=rank,
        world_size=world_size,
    )


def _same_path(a: str, b: str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return os.path.abspath(a) == os.path.abspath(b)


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    is_ddp: bool,
    rank: int,
) -> int:
    if not args.resume:
        return 0

    ckpt = torch.load(args.resume, map_location=device)
    ckpt_variant = ckpt.get("variant")
    if ckpt_variant is not None and ckpt_variant != "diffusion":
        raise ValueError(
            f"Checkpoint variant mismatch: expected 'diffusion', got {ckpt_variant!r}"
        )
    ckpt_args = ckpt.get("args", {})
    if (
        isinstance(ckpt_args, dict)
        and "diffusion_steps" not in ckpt_args
        and ckpt_variant is None
    ):
        raise ValueError("Checkpoint does not look like a diffusion checkpoint.")

    state = ckpt["model"]
    if is_ddp:
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    if not args.no_opt_state:
        optimizer.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
    elif rank == 0:
        print("Skipping optimizer/scaler state (fresh opt).")

    start_step = int(ckpt.get("step", 0)) + 1
    if rank == 0:
        print(f"Resumed from {args.resume} at step {start_step}")
    return start_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a directory of shard_*.bin files",
    )
    ap.add_argument(
        "--out", type=str, default="artifacts/checkpoints/diffusion/ckpt_diffusion.pt"
    )
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)

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

    ap.add_argument(
        "--batch_size", type=int, default=1, help="microbatch sequences per GPU"
    )
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument(
        "--seq_len", type=int, default=None, help="Fixed sequence length for all steps"
    )
    ap.add_argument(
        "--seq_len_cap",
        type=int,
        default=None,
        help="Upper bound for curriculum sequence length",
    )
    ap.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume"
    )
    ap.add_argument(
        "--no_opt_state",
        action="store_true",
        help="When resuming, load model weights only (fresh optimizer/scaler)",
    )

    ap.add_argument(
        "--diffusion_steps",
        type=int,
        default=64,
        help="Number of diffusion denoising steps",
    )
    ap.add_argument(
        "--mask_token_id",
        type=int,
        default=256,
        help="Special mask token id (must be >= 256)",
    )
    ap.add_argument(
        "--min_mask_prob",
        type=float,
        default=0.05,
        help="Mask probability at t=1",
    )
    ap.add_argument(
        "--max_mask_prob",
        type=float,
        default=0.50,
        help="Mask probability at t=diffusion_steps",
    )
    ap.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a tiny 10-step CPU-safe synthetic training loop for CI/testing",
    )
    ap.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting --out when file already exists",
    )

    args = ap.parse_args()

    if args.smoke_test:
        args.ddp = False
        args.steps = min(args.steps, 10)
        args.log_every = 1
        args.save_every = args.steps + 1  # disable checkpointing in smoke path
        args.n_layer = 2
        args.d_model = 128
        args.n_head = 4
        args.d_ff = 256
        args.batch_size = 2
        args.grad_accum = 1
        args.seq_len = args.seq_len if args.seq_len is not None else 128
        args.seq_len = min(args.seq_len, 128)
        args.seq_len_cap = args.seq_len
        args.diffusion_steps = min(args.diffusion_steps, 8)
        args.no_ckpt = True
        args.use_sdpa = False
        if args.out == "artifacts/checkpoints/diffusion/ckpt_diffusion.pt":
            args.out = "artifacts/checkpoints/diffusion/ckpt_diffusion_smoke.pt"

    if not args.smoke_test and not args.data:
        raise ValueError("--data is required unless --smoke_test is used")

    if args.mask_token_id < 256:
        raise ValueError("--mask_token_id must be >= 256")
    if not (0.0 < args.min_mask_prob <= args.max_mask_prob < 1.0):
        raise ValueError("Mask probabilities must satisfy 0 < min <= max < 1")

    rank, local_rank, world_size, is_ddp = setup_ddp(args.ddp)
    if args.smoke_test:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        raise RuntimeError("CUDA is required for non-smoke runs")

    torch.manual_seed(args.seed + rank)

    resume_same_out = args.resume is not None and _same_path(args.resume, args.out)
    if rank == 0:
        out_parent = Path(args.out).parent
        if str(out_parent) not in ("", "."):
            out_parent.mkdir(parents=True, exist_ok=True)
        if Path(args.out).exists() and not args.allow_overwrite and not resume_same_out:
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {args.out}\n"
                "Use --allow_overwrite to replace it, or choose a different --out path."
            )

    model = create_model(args, device)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    opt, use_bnb = create_optimizer(
        model,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.wd,
        use_bnb=(device.type == "cuda"),
    )
    scaler = create_grad_scaler()

    if rank == 0:
        print(f"Using {'bitsandbytes AdamW8bit' if use_bnb else 'torch AdamW'}")

    start_step = maybe_resume(
        model=model,
        optimizer=opt,
        scaler=scaler,
        args=args,
        device=device,
        is_ddp=is_ddp,
        rank=rank,
    )

    dl = None
    it = None
    if not args.smoke_test:
        ds = create_dataset(args, rank, world_size)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=0, pin_memory=True
        )
        it = iter(dl)

    warmup_steps = int(args.steps * args.warmup_frac)
    t0 = time.time()
    model.train()

    if rank == 0 and args.smoke_test:
        print(
            f"Smoke test config: steps={args.steps}, seq_len={args.seq_len}, "
            f"layers={args.n_layer}, d_model={args.d_model}, diffusion_steps={args.diffusion_steps}"
        )

    for step in range(start_step, args.steps):
        seq_len = (
            args.seq_len
            if args.seq_len is not None
            else seq_len_for_step(
                step,
                args.steps,
                cap=args.seq_len_cap,
            )
        )
        if dl is not None:
            dl.dataset.set_seq_len(seq_len)

        for group in opt.param_groups:
            group["lr"] = lr_for_step(step, warmup_steps, args.steps, args.lr)

        opt.zero_grad(set_to_none=True)
        did_backward = False
        total_loss = 0.0
        total_mask_frac = 0.0
        valid_micros = 0
        grad_norm = 0.0

        for micro in range(args.grad_accum):
            if args.smoke_test:
                x0 = torch.randint(
                    0, 256, (args.batch_size, seq_len), device=device, dtype=torch.long
                )
            else:
                x0, _ = next(it)
                x0 = x0.to(device, non_blocking=True)

            t = sample_timesteps(x0.size(0), args.diffusion_steps, device=device)
            x_t, mask = corrupt_with_mask(
                x0=x0,
                t=t,
                num_diffusion_steps=args.diffusion_steps,
                mask_token_id=args.mask_token_id,
                min_mask_prob=args.min_mask_prob,
                max_mask_prob=args.max_mask_prob,
            )

            amp_ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                logits = model(x_t, t)
                raw_loss = masked_denoise_loss(logits=logits, targets=x0, mask=mask)
                loss = raw_loss / args.grad_accum

            if not torch.isfinite(loss):
                print(
                    f"[WARN] non-finite loss at step {step}, skipping microbatch {micro}"
                )
                continue

            scaler.scale(loss).backward()
            did_backward = True
            valid_micros += 1
            total_loss += float(raw_loss.item())
            total_mask_frac += float(mask.float().mean().item())

        if not did_backward:
            opt.zero_grad(set_to_none=True)
            continue

        scaler.unscale_(opt)

        # Compute gradient norm for monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Check for exploding gradients
        if not torch.isfinite(grad_norm):
            print(f"[WARN] non-finite grad norm {grad_norm} at step {step}, skipping")
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue

        if grad_norm > args.grad_clip * 10:
            print(
                f"[WARN] large grad norm {grad_norm:.2f} at step {step}, clipped to {args.grad_clip}"
            )

        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            denom = max(1, valid_micros)
            loss_avg = total_loss / denom
            mask_avg = total_mask_frac / denom
            if is_ddp:
                stats = torch.tensor([loss_avg, mask_avg], device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats /= world_size
                loss_avg = float(stats[0].item())
                mask_avg = float(stats[1].item())

            if rank == 0:
                dt = time.time() - t0
                bpb_masked = loss_avg / math.log(2.0)
                print(
                    f"step {step:6d}  seq {seq_len:4d}  "
                    f"loss {loss_avg:.4f}  masked_bpb {bpb_masked:.4f}  "
                    f"mask {mask_avg * 100.0:.1f}%  gn {grad_norm:.2f}  {dt:.1f}s"
                )
            t0 = time.time()

        if rank == 0 and (step % args.save_every == 0) and step > 0:
            ckpt = {
                "step": step,
                "variant": "diffusion",
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
