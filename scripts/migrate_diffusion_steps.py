#!/usr/bin/env python3
"""Migrate diffusion checkpoints to a different diffusion_steps value.

This resizes timestep embedding `t_emb.weight` from old_steps+1 rows to
new_steps+1 rows and updates checkpoint args accordingly.
"""

import argparse
import math
from pathlib import Path

import torch


def _as_dict_args(raw_args):
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None:
        return {}
    try:
        return vars(raw_args)
    except Exception:
        return {}


def _set_diffusion_steps_arg(ckpt: dict, new_steps: int) -> None:
    raw_args = ckpt.get("args")
    if isinstance(raw_args, dict):
        raw_args["diffusion_steps"] = new_steps
        return
    if raw_args is not None and hasattr(raw_args, "__dict__"):
        setattr(raw_args, "diffusion_steps", new_steps)
        return
    ckpt["args"] = {"diffusion_steps": new_steps}


def _find_t_emb_key(state: dict) -> str:
    if "t_emb.weight" in state:
        return "t_emb.weight"
    candidates = [k for k in state.keys() if k.endswith("t_emb.weight")]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError("Could not find t_emb.weight in checkpoint model state.")
    raise KeyError(f"Ambiguous timestep embedding keys: {candidates}")


def _resize_t_emb(weight: torch.Tensor, new_rows: int, mode: str, seed: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D t_emb weight, got shape {tuple(weight.shape)}")
    old_rows, d_model = weight.shape
    if new_rows < 1:
        raise ValueError("new_rows must be >= 1")
    if new_rows == old_rows:
        return weight.clone()

    if mode == "repeat_last":
        out = weight.new_empty((new_rows, d_model))
        keep = min(old_rows, new_rows)
        out[:keep] = weight[:keep]
        if new_rows > old_rows:
            out[keep:] = weight[old_rows - 1].unsqueeze(0).expand(new_rows - keep, -1)
        return out

    if mode == "interpolate":
        if old_rows == 1:
            return weight[0].unsqueeze(0).expand(new_rows, -1).clone()
        pos = torch.linspace(0.0, float(old_rows - 1), new_rows, device=weight.device)
        lo = torch.floor(pos).long()
        hi = torch.ceil(pos).long().clamp(max=old_rows - 1)
        w = (pos - lo.float()).unsqueeze(1).to(weight.dtype)
        out = weight[lo] * (1.0 - w) + weight[hi] * w
        # Keep endpoints exact.
        out[0] = weight[0]
        out[-1] = weight[-1]
        return out

    if mode == "random":
        out = weight.new_empty((new_rows, d_model))
        keep = min(old_rows, new_rows)
        out[:keep] = weight[:keep]
        if new_rows > old_rows:
            g = torch.Generator(device=weight.device)
            g.manual_seed(seed)
            std = float(weight.std().item())
            if not math.isfinite(std) or std <= 0.0:
                std = 1.0 / math.sqrt(max(1, d_model))
            out[keep:] = torch.randn(
                (new_rows - keep, d_model),
                device=weight.device,
                dtype=weight.dtype,
                generator=g,
            ) * std
        return out

    raise ValueError(f"Unsupported resize mode: {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Input checkpoint path (.pt)")
    ap.add_argument(
        "--new_diffusion_steps",
        type=int,
        required=True,
        help="Target diffusion_steps value",
    )
    ap.add_argument("--out", type=str, default=None, help="Output checkpoint path (.pt)")
    ap.add_argument("--inplace", action="store_true", help="Write back to --ckpt in place")
    ap.add_argument(
        "--mode",
        type=str,
        default="repeat_last",
        choices=["repeat_last", "interpolate", "random"],
        help="Resize strategy for t_emb rows when new steps differ",
    )
    ap.add_argument("--seed", type=int, default=1234, help="Seed for --mode random")
    ap.add_argument("--dry_run", action="store_true", help="Print migration plan without writing")
    ap.add_argument("--allow_overwrite", action="store_true", help="Allow overwriting output file")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow migration even if variant/model_family tags are missing or non-diffusion",
    )
    args = ap.parse_args()

    in_path = Path(args.ckpt)
    if not in_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")
    if in_path.suffix.lower() != ".pt":
        raise ValueError("Only .pt checkpoints are supported")
    if args.new_diffusion_steps < 1:
        raise ValueError("--new_diffusion_steps must be >= 1")
    if args.inplace and args.out is not None:
        raise ValueError("Use either --inplace or --out, not both")

    if args.inplace:
        out_path = in_path
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = in_path.with_name(f"{in_path.stem}.ds{args.new_diffusion_steps}{in_path.suffix}")

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint must be a dict containing 'model'")
    if not isinstance(ckpt["model"], dict):
        raise ValueError("Checkpoint 'model' must be a state_dict mapping")

    variant = ckpt.get("variant")
    family = ckpt.get("model_family")
    args_dict = _as_dict_args(ckpt.get("args"))
    looks_diffusion = (
        variant == "diffusion"
        or family == "diffusion"
        or "diffusion_steps" in args_dict
    )
    if not looks_diffusion and not args.force:
        raise ValueError(
            "Checkpoint does not look like diffusion. Use --force to migrate anyway."
        )

    state = ckpt["model"]
    t_emb_key = _find_t_emb_key(state)
    old_weight = state[t_emb_key]
    if not isinstance(old_weight, torch.Tensor):
        raise ValueError(f"{t_emb_key} is not a Tensor")

    old_rows = int(old_weight.shape[0])
    old_steps_from_temb = old_rows - 1
    old_steps_from_args = args_dict.get("diffusion_steps")
    new_rows = args.new_diffusion_steps + 1

    new_weight = _resize_t_emb(
        weight=old_weight,
        new_rows=new_rows,
        mode=args.mode,
        seed=args.seed,
    )

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"key: {t_emb_key}")
    print(f"mode: {args.mode}")
    print(f"diffusion_steps: args={old_steps_from_args!r}, t_emb={old_steps_from_temb} -> {args.new_diffusion_steps}")
    print(f"t_emb shape: {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}")

    if args.dry_run:
        return

    state[t_emb_key] = new_weight
    _set_diffusion_steps_arg(ckpt, args.new_diffusion_steps)

    out_parent = out_path.parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.resolve() != in_path.resolve() and not args.allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --allow_overwrite to replace it."
        )
    if out_path.exists() and out_path.resolve() == in_path.resolve() and not args.allow_overwrite and not args.inplace:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --allow_overwrite to replace it."
        )

    torch.save(ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
