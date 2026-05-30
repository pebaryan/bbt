#!/usr/bin/env python3
"""Benchmark DEQ and AR checkpoints side-by-side on a shared validation set.

Evaluates every checkpoint on the same data (BPB, PPL, tokens/sec).
For DEQ checkpoints, also reports solver convergence statistics.
Param counts are printed so size differences are visible.

Usage:
    python benchmark_deq.py --data artifacts/datasets/fineweb_edu_4gb/shards/val
    python benchmark_deq.py --data <val_dir> --ckpts <a.pt> <b.pt> ...
"""
import argparse
import glob
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from byte_shard_dataset import ByteShardDataset
from models.bitbytelm import BitByteLM
from models.bitbyte_deq import BitByteDEQ


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    variant: str          # "ar" or "deq"
    params: int
    step: int
    loss: float
    bpb: float
    ppl: float
    tok_per_sec: float
    num_tokens: int
    # DEQ-only solver stats (NaN for AR)
    fp_iters_mean: float = float("nan")
    fp_res_mean: float = float("nan")
    fp_conv_rate: float = float("nan")  # fraction of batches that converged


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ar(ckpt: dict, device: torch.device) -> BitByteLM:
    a = ckpt.get("args", {})
    if not isinstance(a, dict):
        a = vars(a)
    model = BitByteLM(
        vocab_size=256,
        n_layer=a.get("n_layer", 24),
        d_model=a.get("d_model", 1536),
        n_head=a.get("n_head", 12),
        n_kv_head=a.get("n_kv_head", None),
        d_ff=a.get("d_ff", 4096),
        act_quant=not a.get("no_act_quant", False),
        use_sdpa=a.get("use_sdpa", True),
        ckpt=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def _load_deq(ckpt: dict, device: torch.device) -> BitByteDEQ:
    a = ckpt.get("args", {})
    if not isinstance(a, dict):
        a = vars(a)
    model = BitByteDEQ(
        vocab_size=256,
        d_model=a.get("d_model", 512),
        n_head=a.get("n_head", 8),
        n_kv_head=a.get("n_kv_head", None),
        d_ff=a.get("d_ff", 1024),
        n_prelude=a.get("n_prelude", 2),
        n_core=a.get("n_core", 1),
        n_coda=a.get("n_coda", 0),
        quantize=a.get("quantize", False),
        act_quant=not a.get("no_act_quant", False),
        use_sdpa=not a.get("no_sdpa", False),
        solver=a.get("solver", "anderson"),
        max_iter=a.get("max_iter", 24),
        tol=a.get("tol", 1e-3),
        solver_beta=a.get("solver_beta", 1.0),
        anderson_m=a.get("anderson_m", 5),
        layer_scale_init=a.get("layer_scale_init", 0.1),
        gamma_max=a.get("gamma_max", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_checkpoint(path: str, device: torch.device):
    """Load checkpoint, auto-detect variant, return (model, ckpt_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    variant = ckpt.get("variant")

    # Infer variant from args if tag is missing.
    if variant is None:
        a = ckpt.get("args", {})
        if not isinstance(a, dict):
            a = vars(a)
        if "n_prelude" in a or "n_core" in a or "solver" in a:
            variant = "deq"
        else:
            variant = "ar"

    if variant == "deq":
        model = _load_deq(ckpt, device)
    else:
        model = _load_ar(ckpt, device)

    return model, variant, ckpt


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ar(model, dataloader, device, max_batches: int, use_amp: bool) -> dict:
    model.eval()
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
               if use_amp and device.type == "cuda" else nullcontext())
    nll_sum = 0.0
    tok_count = 0
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
            nll_sum += float(
                F.cross_entropy(logits.view(-1, 256), y.view(-1), reduction="sum").item()
            )
        tok_count += int(y.numel())
    elapsed = time.perf_counter() - t0
    loss = nll_sum / max(1, tok_count)
    return dict(loss=loss, bpb=loss / math.log(2.0), ppl=math.exp(min(loss, 100)),
                tok_per_sec=tok_count / max(elapsed, 1e-6), num_tokens=tok_count)


@torch.no_grad()
def eval_deq(model, dataloader, device, max_batches: int, use_amp: bool) -> dict:
    model.eval()
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
               if use_amp and device.type == "cuda" else nullcontext())
    nll_sum = 0.0
    tok_count = 0
    iters_sum = 0
    res_sum = 0.0
    conv_sum = 0
    n_batches = 0
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
            nll_sum += float(
                F.cross_entropy(logits.view(-1, 256), y.view(-1), reduction="sum").item()
            )
        tok_count += int(y.numel())
        info = model.last_info
        iters_sum += int(info.get("iters", 0))
        res_sum += float(info.get("rel_residual", 0.0))
        conv_sum += int(bool(info.get("converged", False)))
        n_batches += 1
    elapsed = time.perf_counter() - t0
    loss = nll_sum / max(1, tok_count)
    n = max(1, n_batches)
    return dict(loss=loss, bpb=loss / math.log(2.0), ppl=math.exp(min(loss, 100)),
                tok_per_sec=tok_count / max(elapsed, 1e-6), num_tokens=tok_count,
                fp_iters_mean=iters_sum / n, fp_res_mean=res_sum / n,
                fp_conv_rate=conv_sum / n)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    return f"{n/1e6:.1f}M"


def print_table(results: list[BenchResult]) -> None:
    # Header
    hdr = (f"{'Name':<36}  {'Var':>4}  {'Params':>7}  {'Step':>6}  "
           f"{'BPB':>6}  {'PPL':>8}  {'Tok/s':>8}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<36}  {r.variant:>4}  {_fmt_params(r.params):>7}  "
              f"{r.step:>6}  {r.bpb:>6.4f}  {r.ppl:>8.2f}  {r.tok_per_sec:>8.0f}")
    print("=" * len(hdr))

    # DEQ solver detail table
    deq = [r for r in results if r.variant == "deq"]
    if deq:
        print(f"\n{'Name':<36}  {'fp_iters':>8}  {'fp_res':>8}  {'conv%':>6}")
        print("-" * 65)
        for r in deq:
            print(f"{r.name:<36}  {r.fp_iters_mean:>8.1f}  {r.fp_res_mean:>8.2e}  "
                  f"{r.fp_conv_rate * 100:>5.1f}%")
        print("-" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_checkpoints(deq_dir: str, ar_dir: str) -> list[str]:
    paths = []
    if deq_dir:
        paths += sorted(glob.glob(os.path.join(deq_dir, "*.pt")))
    if ar_dir:
        paths += sorted(glob.glob(os.path.join(ar_dir, "*.pt")))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str,
                    default="artifacts/datasets/fineweb_edu_4gb/shards/val",
                    help="Validation shard directory (shard_*.bin)")
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="Explicit checkpoint paths. If omitted, scans --deq_dir and --ar_dir.")
    ap.add_argument("--deq_dir", type=str,
                    default="artifacts/checkpoints/deq",
                    help="Directory to scan for DEQ checkpoints (used when --ckpts is omitted)")
    ap.add_argument("--ar_dir", type=str, default=None,
                    help="Directory to scan for AR checkpoints (used when --ckpts is omitted)")
    ap.add_argument("--seq_len", type=int, default=512,
                    help="Validation sequence length (default 512, matches DEQ training)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=200,
                    help="Validation batches per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp
    print(f"Device: {device}  AMP: {use_amp}")

    # Resolve checkpoint list.
    if args.ckpts:
        paths = args.ckpts
    else:
        paths = discover_checkpoints(args.deq_dir, args.ar_dir)
    if not paths:
        print("No checkpoints found. Pass --ckpts or set --deq_dir / --ar_dir.")
        return

    # Validation DataLoader (shared across all models).
    ds = ByteShardDataset(
        shard_glob=os.path.join(args.data, "shard_*.bin"),
        seq_len=args.seq_len, seed=args.seed, rank=0, world_size=1)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0, pin_memory=(device.type == "cuda"))

    results = []
    for path in paths:
        name = os.path.basename(path)
        print(f"\n[{name}]  loading ...", end=" ", flush=True)
        try:
            model, variant, ckpt = load_checkpoint(path, device)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        step = int(ckpt.get("step", 0))
        a = ckpt.get("args", {})
        if not isinstance(a, dict):
            a = vars(a)
        trained_data = os.path.basename(os.path.dirname(a.get("data", "") or ""))

        print(f"{variant.upper()}  {_fmt_params(n_params)}  step={step}  "
              f"train_data={trained_data or '?'}  evaluating ...", end=" ", flush=True)

        if variant == "deq":
            stats = eval_deq(model, dl, device, args.batches, use_amp)
            r = BenchResult(
                name=name, variant=variant, params=n_params, step=step,
                loss=stats["loss"], bpb=stats["bpb"], ppl=stats["ppl"],
                tok_per_sec=stats["tok_per_sec"], num_tokens=stats["num_tokens"],
                fp_iters_mean=stats["fp_iters_mean"],
                fp_res_mean=stats["fp_res_mean"],
                fp_conv_rate=stats["fp_conv_rate"],
            )
        else:
            stats = eval_ar(model, dl, device, args.batches, use_amp)
            r = BenchResult(
                name=name, variant=variant, params=n_params, step=step,
                loss=stats["loss"], bpb=stats["bpb"], ppl=stats["ppl"],
                tok_per_sec=stats["tok_per_sec"], num_tokens=stats["num_tokens"],
            )

        print(f"bpb={r.bpb:.4f}  ppl={r.ppl:.1f}  {r.tok_per_sec:,.0f} tok/s")
        results.append(r)

        # Free VRAM between models.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if results:
        print_table(results)
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
