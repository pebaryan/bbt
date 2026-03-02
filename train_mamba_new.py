#!/usr/bin/env python3
"""Main training script for MambaMLM with checkpoint safety guards."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from data.byte_shard_dataset import ByteShardDataset
from models.mamba_mlm import MambaMLM
from training.ddp import setup_ddp
from training.trainer import Trainer
from utils.optim import create_optimizer, create_grad_scaler


def create_model(args: argparse.Namespace, device: torch.device) -> MambaMLM:
    model = MambaMLM(
        vocab_size=256,
        n_layer=args.n_layer,
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        act_quant=not args.no_act_quant,
        use_checkpoint=not args.no_checkpoint,
        time_step_min=args.time_step_min,
        time_step_max=args.time_step_max,
        dt_init=args.dt_init,
        a_init=args.a_init,
    ).to(device)
    return model


def create_dataset(
    args: argparse.Namespace, rank: int, world_size: int
) -> ByteShardDataset:
    seq_len = args.seq_len if args.seq_len is not None else 2048
    ds = ByteShardDataset(
        shard_glob=os.path.join(args.data, "shard_*.bin"),
        seq_len=seq_len,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    return ds


def _same_path(a: str, b: str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return os.path.abspath(a) == os.path.abspath(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to a directory of shard_*.bin files",
    )
    ap.add_argument(
        "--out", type=str, default="artifacts/checkpoints/mamba/ckpt_mamba.pt"
    )
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--n_layer", type=int, default=24)
    ap.add_argument("--d_model", type=int, default=1536)
    ap.add_argument("--d_ff", type=int, default=4096)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument(
        "--time_step_min",
        type=float,
        default=1e-3,
        help="Minimum initial dt value for Mamba delta bias initialization",
    )
    ap.add_argument(
        "--time_step_max",
        type=float,
        default=1e-1,
        help="Maximum initial dt value for Mamba delta bias initialization",
    )
    ap.add_argument(
        "--dt_init",
        type=str,
        default="log_uniform",
        choices=["log_uniform", "zeros"],
        help="Initialization for Mamba delta bias",
    )
    ap.add_argument(
        "--a_init",
        type=str,
        default="uniform_0_16",
        choices=["uniform_0_16", "log_arange"],
        help="Initialization for Mamba A_log parameter",
    )

    ap.add_argument("--ddp", action="store_true", help="Enable DDP (requires torchrun)")
    ap.add_argument("--no_act_quant", action="store_true")
    ap.add_argument(
        "--no_checkpoint", action="store_true", help="Disable gradient checkpointing"
    )

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
        help="Upper bound applied to curriculum sequence length",
    )
    ap.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume"
    )
    ap.add_argument(
        "--no_opt_state",
        action="store_true",
        help="When resuming, load model weights only",
    )
    ap.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting --out when file already exists",
    )

    args = ap.parse_args()
    args.model_family = "mamba"

    rank, local_rank, world_size, is_ddp = setup_ddp(args.ddp)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

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
        model, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd
    )
    if rank == 0:
        print(f"Using {'bitsandbytes AdamW8bit' if use_bnb else 'torch AdamW'}")

    scaler = create_grad_scaler()
    trainer = Trainer(
        model=model,
        optimizer=opt,
        scaler=scaler,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_ddp=is_ddp,
        args=args,
    )

    if args.resume:
        trainer.resume(args.resume, load_opt_state=not args.no_opt_state)

    ds = create_dataset(args, rank, world_size)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0, pin_memory=True
    )
    trainer.train(dl)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
