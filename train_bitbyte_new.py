#!/usr/bin/env python3
"""Main training script for BitByteLM using modularized code."""

import argparse
import math
import os
import torch
import torch.distributed as dist

from models.bitbytelm import BitByteLM
from data.byte_shard_dataset import ByteShardDataset
from training.ddp import setup_ddp
from training.trainer import Trainer
from training.lr_scheduler import seq_len_for_step
from utils.optim import create_optimizer, create_grad_scaler


def create_model(args: argparse.Namespace, device: torch.device) -> BitByteLM:
    """Create and initialize the model."""
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
    return model


def create_dataset(args: argparse.Namespace, rank: int, world_size: int) -> ByteShardDataset:
    """Create the training dataset."""
    ds = ByteShardDataset(
        shard_glob=os.path.join(args.data, "shard_*.bin"),
        seq_len=2048,
        seed=1234,
        rank=rank,
        world_size=world_size,
    )
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="Path to a directory of shard_*.bin files")
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
                    help="Fixed sequence length for all steps (disables curriculum)")
    ap.add_argument("--seq_len_cap", type=int, default=None,
                    help="Upper bound applied to curriculum sequence length")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume")
    ap.add_argument("--no_opt_state", action="store_true",
                    help="When resuming, load model weights only (fresh optimizer/scaler)")

    args = ap.parse_args()

    rank, local_rank, world_size, is_ddp = setup_ddp(args.ddp)
    device = torch.device("cuda", local_rank)

    # Create model
    model = create_model(args, device)
    
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Create optimizer
    opt, use_bnb = create_optimizer(
        model, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    
    if rank == 0:
        print(f"Using {'bitsandbytes AdamW8bit' if use_bnb else 'torch AdamW'}")

    # Create grad scaler
    scaler = create_grad_scaler()

    # Resume if specified
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

    # Create dataset
    ds = create_dataset(args, rank, world_size)
    
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # Train
    trainer.train(dl)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()