import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from quantization.bitlinear import BitLinear
from models.bitbytelm import BitByteLM
from data.byte_shard_dataset import ByteShardDataset
from training.ddp import setup_ddp
from training.lr_scheduler import lr_for_step, seq_len_for_step
from training.loss import bits_per_byte

import os


class Trainer:
    """Trainer class for BitByteLM."""
    
    def __init__(
        self,
        model: BitByteLM,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        rank: int,
        local_rank: int,
        world_size: int,
        is_ddp: bool,
        args: argparse.Namespace,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            scaler: Gradient scaler for mixed precision
            rank: Process rank
            local_rank: Local process rank
            world_size: Total number of processes
            is_ddp: Whether using DDP
            args: Command line arguments
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_ddp = is_ddp
        self.args = args
        
        self.device = torch.device("cuda", local_rank)
        self.start_step = 0
        
    def resume(self, path: str, load_opt_state: bool = True):
        """Resume from checkpoint.
        
        Args:
            path: Path to checkpoint
            load_opt_state: Whether to load optimizer state
        """
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model"]
        
        if self.is_ddp:
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)

        if load_opt_state:
            self.optimizer.load_state_dict(ckpt["opt"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.start_step = int(ckpt.get("step", 0)) + 1
        else:
            self.start_step = int(ckpt.get("step", 0)) + 1
            if self.rank == 0:
                print("Skipping optimizer/scaler state (fresh opt).")

        if self.rank == 0:
            print(f"Resumed from {path} at step {self.start_step}")
            
    def train(self, dataloader: DataLoader):
        """Train the model.
        
        Args:
            dataloader: Data loader for training data
        """
        args = self.args
        warmup_steps = int(args.steps * args.warmup_frac)

        self.model.train()
        t0 = time.time()
        it = iter(dataloader)

        for step in range(self.start_step, args.steps):
            seq_len = seq_len_for_step(
                step,
                args.steps,
                cap=args.seq_len_cap if args.seq_len is None else args.seq_len
            )
            dataloader.dataset.set_seq_len(seq_len)

            for g in self.optimizer.param_groups:
                g["lr"] = lr_for_step(step, warmup_steps, args.steps, args.lr)

            self.optimizer.zero_grad(set_to_none=True)

            did_backward = False
            total_loss = 0.0

            for micro in range(args.grad_accum):
                x, y = next(it)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                    loss = loss / args.grad_accum

                if not torch.isfinite(loss):
                    print(f"[WARN] non-finite loss at step {step}, skipping microbatch {micro}")
                    continue

                self.scaler.scale(loss).backward()
                did_backward = True
                total_loss += float(loss.item())

            if not did_backward:
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # clip
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if step % args.log_every == 0:
                # all-reduce loss for reporting
                loss_avg = total_loss
                if self.is_ddp:
                    loss_t = torch.tensor([total_loss], device=self.device)
                    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                    loss_avg = loss_t.item() / self.world_size

                if self.rank == 0:
                    dt = time.time() - t0
                    bpb = loss_avg / math.log(2.0)
                    print(
                        f"step {step:6d}  seq {seq_len:4d}  loss {loss_avg:.4f}  bpb {bpb:.4f}  {dt:.1f}s")
                t0 = time.time()

            if self.rank == 0 and (step % args.save_every == 0) and step > 0:
                # Save non-DDP state
                ckpt = {
                    "step": step,
                    "model": (self.model.module.state_dict() if self.is_ddp else self.model.state_dict()),
                    "opt": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, args.out)
                print(f"saved {args.out}")