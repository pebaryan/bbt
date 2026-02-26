#!/usr/bin/env python3
"""Extract an inference-only checkpoint with frozen ternary weights."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bitbytelm import BitByteLM
from models.bitbytelm_fast import FastBitByteLM
from quantization.bitlinear import BitLinear


def _ckpt_arg(cfg: dict, key: str, default):
    return cfg.get(key, default)


def _to_cfg_dict(cfg):
    if isinstance(cfg, dict):
        return dict(cfg)
    return vars(cfg)


def ternarize_weight(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w32 = w.detach().to(dtype=torch.float32)
    gamma = torch.clamp(w32.abs().mean(), min=eps)
    return torch.clamp(w32 / gamma, -1.0, 1.0).round() * gamma


def resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True, type=str)
    ap.add_argument("--out_ckpt", required=True, type=str)
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Storage dtype for extracted weights.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output checkpoint if it already exists.",
    )
    args = ap.parse_args()

    out_path = Path(args.out_ckpt)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {out_path} (use --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    ckpt = torch.load(args.in_ckpt, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model': {args.in_ckpt}")

    cfg = _to_cfg_dict(ckpt.get("args", {}))

    src_model = BitByteLM(
        vocab_size=256,
        n_layer=_ckpt_arg(cfg, "n_layer", 24),
        d_model=_ckpt_arg(cfg, "d_model", 1536),
        n_head=_ckpt_arg(cfg, "n_head", 12),
        d_ff=_ckpt_arg(cfg, "d_ff", 4096),
        act_quant=not _ckpt_arg(cfg, "no_act_quant", False),
        rope_base=_ckpt_arg(cfg, "rope_base", 10000.0),
        use_sdpa=_ckpt_arg(cfg, "use_sdpa", True),
        ckpt=False,
    )
    src_model.load_state_dict(ckpt["model"], strict=True)
    src_model.eval()

    fast_model = FastBitByteLM(
        vocab_size=256,
        n_layer=_ckpt_arg(cfg, "n_layer", 24),
        d_model=_ckpt_arg(cfg, "d_model", 1536),
        n_head=_ckpt_arg(cfg, "n_head", 12),
        d_ff=_ckpt_arg(cfg, "d_ff", 4096),
        rope_base=_ckpt_arg(cfg, "rope_base", 10000.0),
        use_sdpa=_ckpt_arg(cfg, "use_sdpa", True),
    )

    ternary_weight_keys = {
        f"{name}.weight" for name, module in src_model.named_modules() if isinstance(module, BitLinear)
    }
    src_sd = src_model.state_dict()
    dst_dtype = resolve_dtype(args.dtype)

    new_sd = {}
    converted = 0
    for key, value in src_sd.items():
        if key in ternary_weight_keys:
            q = ternarize_weight(value).to(dtype=dst_dtype)
            new_sd[key] = q
            converted += 1
        else:
            if value.is_floating_point():
                new_sd[key] = value.to(dtype=dst_dtype)
            else:
                new_sd[key] = value

    fast_model.load_state_dict(new_sd, strict=True)
    fast_model.eval()

    out_cfg = dict(cfg)
    out_cfg["no_act_quant"] = True
    out_cfg["no_ckpt"] = True

    out = {
        "format": "fast_ternary_v1",
        "source_ckpt": str(args.in_ckpt),
        "dtype": args.dtype,
        "ternary_linear_layers": converted,
        "args": out_cfg,
        "model": fast_model.state_dict(),
    }
    torch.save(out, out_path)

    print(f"Saved: {out_path}")
    print(f"Converted ternary linear weights: {converted}")
    print(f"Format: fast_ternary_v1")


if __name__ == "__main__":
    main()
