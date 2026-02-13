#!/usr/bin/env python3
"""Bulk tag checkpoints with variant/model_family metadata."""

import argparse
from pathlib import Path

import torch

def _as_dict_args(raw_args):
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None:
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
    try:
        return vars(raw_args)
    except Exception:
        return {}


def inspect_checkpoint(path: Path):
    try:
        ckpt = torch.load(path, map_location="meta")
    except Exception as e:
        return {"path": path, "status": "error", "error": f"{type(e).__name__}: {e}"}

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        return {"path": path, "status": "skip", "reason": "not a standard dict checkpoint with 'model'"}

    args_dict = _as_dict_args(ckpt.get("args"))
    state = ckpt.get("model", {})
    state_keys = list(state.keys()) if isinstance(state, dict) else []
    existing_variant = ckpt.get("variant")
    existing_family = ckpt.get("model_family")

    inferred_family = infer_family(args_dict, state_keys)
    inferred_variant = infer_variant(inferred_family, args_dict)

    return {
        "path": path,
        "status": "ok",
        "existing_variant": existing_variant,
        "existing_family": existing_family,
        "inferred_variant": inferred_variant,
        "inferred_family": inferred_family,
    }


def apply_tags(path: Path, variant: str, family: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    ckpt["variant"] = variant
    ckpt["model_family"] = family
    torch.save(ckpt, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="artifacts/checkpoints")
    ap.add_argument("--glob", type=str, default="*.pt", help="Filename pattern (applied recursively)")
    ap.add_argument("--apply", action="store_true", help="Write tags in place (default is dry-run)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing tags if they differ")
    ap.add_argument(
        "--retag_tagged",
        action="store_true",
        help="Also process checkpoints that already have both tags (default skips them)",
    )
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of files processed (0 = no limit)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    paths = sorted(root.rglob(args.glob))
    if args.max_files > 0:
        paths = paths[: args.max_files]

    if not paths:
        print("No matching checkpoints found.")
        return

    print(f"Scanning {len(paths)} checkpoint(s) under {root}")
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")

    changed = 0
    skipped = 0
    errors = 0

    for p in paths:
        info = inspect_checkpoint(p)
        if info["status"] == "error":
            errors += 1
            print(f"[ERROR] {p}: {info['error']}")
            continue
        if info["status"] == "skip":
            skipped += 1
            print(f"[SKIP ] {p}: {info['reason']}")
            continue

        existing_variant = info["existing_variant"]
        existing_family = info["existing_family"]
        inferred_variant = info["inferred_variant"]
        inferred_family = info["inferred_family"]

        fully_tagged = existing_variant is not None and existing_family is not None
        if fully_tagged and not args.retag_tagged:
            skipped += 1
            print(f"[SKIP ] {p}: already tagged variant={existing_variant!r}, model_family={existing_family!r}")
            continue

        target_variant = existing_variant if existing_variant is not None else inferred_variant
        target_family = existing_family if existing_family is not None else inferred_family

        # If force is set, inferred tags win.
        if args.force:
            target_variant = inferred_variant
            target_family = inferred_family

        will_change = (existing_variant != target_variant) or (existing_family != target_family)
        status = "PLAN " if not args.apply else "WRITE"
        print(
            f"[{status}] {p} | "
            f"variant {existing_variant!r} -> {target_variant!r}, "
            f"model_family {existing_family!r} -> {target_family!r}"
        )

        if args.apply and will_change:
            apply_tags(p, target_variant, target_family)
            changed += 1
        elif will_change:
            changed += 1

    print(
        f"Done. changed={changed} skipped={skipped} errors={errors} "
        f"({'applied' if args.apply else 'planned'})"
    )


if __name__ == "__main__":
    main()
