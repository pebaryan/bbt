#!/usr/bin/env python3
"""Tag legacy checkpoints with variant/model_family metadata."""

import argparse
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


def infer_family(args_dict: dict, state_keys: list[str]) -> str:
    if any(k in args_dict for k in ("diffusion_steps", "mask_token_id", "min_mask_prob", "max_mask_prob")):
        return "diffusion"
    if any(k in args_dict for k in ("d_state", "d_conv", "expand")):
        return "mamba"
    joined = " ".join(state_keys[:200]).lower()
    if "ssm" in joined or "mamba" in joined:
        return "mamba"
    return "bitbyte"


def infer_variant(family: str, args_dict: dict) -> str:
    if family == "diffusion" or "diffusion_steps" in args_dict:
        return "diffusion"
    return "ar"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Input checkpoint path (.pt)")
    ap.add_argument("--out", type=str, default=None, help="Output checkpoint path (.pt)")
    ap.add_argument("--inplace", action="store_true", help="Write back to --ckpt in place")
    ap.add_argument("--variant", choices=["ar", "diffusion"], default=None, help="Override inferred variant")
    ap.add_argument(
        "--model_family",
        choices=["bitbyte", "mamba", "diffusion"],
        default=None,
        help="Override inferred model family",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing variant/model_family")
    ap.add_argument("--dry_run", action="store_true", help="Print inferred tags without writing")
    ap.add_argument("--allow_overwrite", action="store_true", help="Allow overwriting output file")
    args = ap.parse_args()

    in_path = Path(args.ckpt)
    if not in_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")
    if in_path.suffix.lower() != ".pt":
        raise ValueError("Only .pt checkpoints are supported")
    if args.inplace and args.out is not None:
        raise ValueError("Use either --inplace or --out, not both")

    if args.inplace:
        out_path = in_path
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = in_path.with_name(f"{in_path.stem}.tagged{in_path.suffix}")

    ckpt = torch.load(in_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint must be a dict containing 'model'")

    args_dict = _as_dict_args(ckpt.get("args"))
    state = ckpt.get("model", {})
    state_keys = list(state.keys()) if isinstance(state, dict) else []

    existing_variant = ckpt.get("variant")
    existing_family = ckpt.get("model_family")

    family = args.model_family or infer_family(args_dict, state_keys)
    variant = args.variant or infer_variant(family, args_dict)

    if existing_variant is not None and existing_variant != variant and not args.force:
        raise ValueError(
            f"Existing variant={existing_variant!r} differs from requested/inferred {variant!r}. "
            "Use --force to override."
        )
    if existing_family is not None and existing_family != family and not args.force:
        raise ValueError(
            f"Existing model_family={existing_family!r} differs from requested/inferred {family!r}. "
            "Use --force to override."
        )

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"variant: {existing_variant!r} -> {variant!r}")
    print(f"model_family: {existing_family!r} -> {family!r}")

    if args.dry_run:
        return

    ckpt["variant"] = variant
    ckpt["model_family"] = family

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
