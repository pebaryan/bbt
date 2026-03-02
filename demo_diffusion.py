#!/usr/bin/env python3
"""Demo infilling with a trained Diffusion checkpoint.

The diffusion model denoises masked sequences. This demo shows how to:
1. Start with a prompt and mask some positions
2. Use the diffusion model to predict the masked positions
3. Show the infilled result
"""

import argparse
from contextlib import nullcontext
import json
import sys
import time

import torch
import torch.nn.functional as F

from models.bitbyte_diffusion import BitByteDiffusionLM
from training.diffusion import corrupt_with_mask, sample_timesteps


def _ckpt_arg(cfg: dict, key: str, default):
    return cfg.get(key, default)


def load_model(
    ckpt_path: str, device: torch.device
) -> tuple[BitByteDiffusionLM, dict, int, int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model' state dict: {ckpt_path}")

    cfg = ckpt.get("args", {})
    if not isinstance(cfg, dict):
        cfg = vars(cfg)

    num_diffusion_steps = _ckpt_arg(cfg, "diffusion_steps", 24)
    mask_token_id = _ckpt_arg(cfg, "mask_token_id", 256)

    model = BitByteDiffusionLM(
        vocab_size=256,
        n_layer=_ckpt_arg(cfg, "n_layer", 6),
        d_model=_ckpt_arg(cfg, "d_model", 320),
        n_head=_ckpt_arg(cfg, "n_head", 5),
        d_ff=_ckpt_arg(cfg, "d_ff", 640),
        num_diffusion_steps=num_diffusion_steps,
        mask_token_id=mask_token_id,
        act_quant=not _ckpt_arg(cfg, "no_act_quant", False),
        use_sdpa=_ckpt_arg(cfg, "use_sdpa", False),
        ckpt=not _ckpt_arg(cfg, "no_ckpt", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, num_diffusion_steps, mask_token_id


def _prompt_to_ids(prompt: str, mask_marker: str, mask_token_id: int) -> list[int]:
    """Encode prompt and optionally replace marker spans with diffusion mask token."""
    if not mask_marker:
        return list(prompt.encode("utf-8", errors="replace"))

    parts = prompt.split(mask_marker)
    if len(parts) == 1:
        return list(prompt.encode("utf-8", errors="replace"))

    ids: list[int] = []
    for i, part in enumerate(parts):
        ids.extend(part.encode("utf-8", errors="replace"))
        if i < len(parts) - 1:
            ids.append(mask_token_id)
    return ids


def infill(
    model: BitByteDiffusionLM,
    prompt: str,
    max_length: int,
    num_diffusion_steps: int,
    mask_token_id: int,
    device: torch.device,
    use_amp: bool,
    temperature: float = 1.0,
    space_penalty: float = 0.0,
    mask_prob: float = 0.15,
    mask_seed: int = 1234,
    mask_mode: str = "token",
    span_len: float = 8.0,
    mask_marker: str = "[MASK]",
    t_start: int | None = None,
    t_end: int = 1,
) -> bytes:
    """Infill masked positions in the prompt using diffusion.

    If no explicit mask token positions are present, randomly masks positions with
    `mask_prob`. Denoising runs iteratively from `t_start` down to `t_end`.
    """
    prompt_list = _prompt_to_ids(prompt, mask_marker=mask_marker, mask_token_id=mask_token_id)
    prompt_list = prompt_list[:max_length]
    if not prompt_list:
        raise ValueError("Prompt becomes empty after encoding/truncation")

    x = torch.tensor([prompt_list], dtype=torch.long, device=device)

    # Check if there are any mask tokens
    mask_positions = x == mask_token_id
    t_start_eff = (
        num_diffusion_steps
        if t_start is None
        else max(1, min(int(t_start), int(num_diffusion_steps)))
    )
    t_end_eff = max(1, min(int(t_end), t_start_eff))

    # If no explicit masks, create random masks at the requested start timestep.
    if not mask_positions.any():
        torch.manual_seed(mask_seed)
        t = torch.tensor([t_start_eff], device=device, dtype=torch.long)
        x_t, mask_positions = corrupt_with_mask(
            x,
            t,
            num_diffusion_steps,
            mask_token_id,
            min_mask_prob=mask_prob,
            max_mask_prob=mask_prob,
            mask_mode=mask_mode,
            span_len=span_len,
        )
    else:
        x_t = x.clone()

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad(), amp_ctx:
        # Iteratively denoise from high noise to low noise.
        result = x_t.clone()
        for t_val in range(t_start_eff, t_end_eff - 1, -1):
            t = torch.full(
                (result.size(0),), t_val, device=device, dtype=torch.long
            )
            logits = model(result, t)  # [1, seq_len, vocab_size]
            if space_penalty > 0.0:
                logits[..., 32] = logits[..., 32] - space_penalty

            if temperature <= 0:
                predictions = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(
                    probs.view(-1, logits.size(-1)),
                    num_samples=1,
                ).view_as(result)

            result[mask_positions] = predictions[mask_positions]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    num_masked = int(mask_positions.sum().item())
    metrics = {
        "prompt_tokens": len(prompt_list),
        "masked_positions": num_masked,
        "infill_time_s": elapsed_s,
        "tokens_per_sec": (num_masked / elapsed_s) if elapsed_s > 0 else 0.0,
        "ms_per_token": ((elapsed_s * 1000.0) / num_masked) if num_masked > 0 else 0.0,
        "denoise_steps": (t_start_eff - t_end_eff + 1),
        "t_start": t_start_eff,
        "t_end": t_end_eff,
        "space_penalty": space_penalty,
        "mask_prob": mask_prob,
        "mask_seed": mask_seed,
        "mask_mode": mask_mode,
        "span_len": span_len,
    }

    return bytes(result[0].tolist()), metrics, mask_positions[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infill masked positions using a diffusion model. "
        "Use byte 256 (mask token) in your prompt to mark positions to infill."
    )
    ap.add_argument(
        "--ckpt", type=str, default="artifacts/checkpoints/diffusion/ckpt_diffusion.pt"
    )
    ap.add_argument("--prompt", type=str, default="Once upon a time there was a ")
    ap.add_argument(
        "--max_length", type=int, default=50, help="Maximum sequence length"
    )
    ap.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (0=greedy)"
    )
    ap.add_argument(
        "--space_penalty",
        type=float,
        default=0.0,
        help="Subtract from space-token logit (byte 32) to reduce blank-space bias",
    )
    ap.add_argument(
        "--mask_prob",
        type=float,
        default=0.15,
        help="Mask probability used when prompt has no explicit mask marker",
    )
    ap.add_argument(
        "--mask_seed",
        type=int,
        default=1234,
        help="Random seed used to sample masks when prompt has no explicit marker",
    )
    ap.add_argument(
        "--mask_mode",
        type=str,
        default="token",
        choices=["token", "span"],
        help="Masking pattern used when prompt has no explicit mask marker",
    )
    ap.add_argument(
        "--span_len",
        type=float,
        default=8.0,
        help="Average span length for --mask_mode span",
    )
    ap.add_argument(
        "--mask_marker",
        type=str,
        default="[MASK]",
        help="Text marker in prompt to indicate explicit mask positions",
    )
    ap.add_argument(
        "--t_start",
        type=int,
        default=0,
        help="Start denoising timestep (0 means checkpoint diffusion_steps)",
    )
    ap.add_argument(
        "--t_end",
        type=int,
        default=1,
        help="End denoising timestep (inclusive)",
    )
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Print performance metrics after infilling",
    )
    ap.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON",
    )
    args = ap.parse_args()

    if args.max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if args.space_penalty < 0.0:
        raise ValueError("--space_penalty must be >= 0")
    if not (0.0 < args.mask_prob < 1.0):
        raise ValueError("--mask_prob must be in (0, 1)")
    if args.span_len <= 0:
        raise ValueError("--span_len must be > 0")
    if args.t_start < 0:
        raise ValueError("--t_start must be >= 0")
    if args.t_end < 1:
        raise ValueError("--t_end must be >= 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    load_t0 = time.perf_counter()
    model, cfg, num_diffusion_steps, mask_token_id = load_model(args.ckpt, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_time_s = time.perf_counter() - load_t0

    out_bytes, infill_metrics, mask_array = infill(
        model=model,
        prompt=args.prompt,
        max_length=args.max_length,
        num_diffusion_steps=num_diffusion_steps,
        mask_token_id=mask_token_id,
        device=device,
        use_amp=use_amp,
        temperature=args.temperature,
        space_penalty=args.space_penalty,
        mask_prob=args.mask_prob,
        mask_seed=args.mask_seed,
        mask_mode=args.mask_mode,
        span_len=args.span_len,
        mask_marker=args.mask_marker,
        t_start=(None if args.t_start == 0 else args.t_start),
        t_end=args.t_end,
    )

    # Display result with mask positions highlighted
    print("Input prompt:")
    print(args.prompt)
    print("\nMask positions (if any):")
    mask_str = "".join(["^" if m else " " for m in mask_array[: len(out_bytes)]])
    print(mask_str)
    print("\nInfilled result:")
    result_str = out_bytes.decode("utf-8", errors="replace")
    enc = sys.stdout.encoding or "utf-8"
    safe_result = result_str.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe_result)

    num_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "device": str(device),
        "checkpoint": args.ckpt,
        "model_load_time_s": load_time_s,
        "parameter_count": int(num_params),
        "n_layer": _ckpt_arg(cfg, "n_layer", 6),
        "d_model": _ckpt_arg(cfg, "d_model", 320),
        "num_diffusion_steps": num_diffusion_steps,
        "temperature": args.temperature,
        **infill_metrics,
    }
    if device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (
            1024 * 1024
        )

    if args.metrics or args.metrics_json:
        print("\n[metrics]")
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

    if args.metrics_json:
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
