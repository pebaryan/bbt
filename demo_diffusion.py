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
import time

import torch
import torch.nn.functional as F

from models.bitbyte_diffusion import BitByteDiffusionLM
from training.diffusion import corrupt_with_mask, sample_timesteps


def _ckpt_arg(cfg: dict, key: str, default):
    return cfg.get(key, default)


def load_model(ckpt_path: str, device: torch.device) -> tuple[BitByteDiffusionLM, dict]:
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


def infill(
    model: BitByteDiffusionLM,
    prompt: bytes,
    max_length: int,
    num_diffusion_steps: int,
    mask_token_id: int,
    device: torch.device,
    use_amp: bool,
    temperature: float = 1.0,
) -> bytes:
    """Infill masked positions in the prompt using diffusion.

    If no mask token (256) is present in the prompt, randomly masks some positions.
    """
    # Convert prompt to list of ints (mask_token_id can be > 255)
    prompt_list = list(prompt)

    # Ensure we have at least max_length tokens
    if len(prompt_list) < max_length:
        prompt_list = prompt_list + [mask_token_id] * (max_length - len(prompt_list))
    prompt_list = prompt_list[:max_length]

    x = torch.tensor([prompt_list], dtype=torch.long, device=device)

    # Check if there are any mask tokens
    mask_positions = x == mask_token_id

    # If no explicit masks, create random masks at high timestep
    if not mask_positions.any():
        t = torch.tensor([num_diffusion_steps], device=device, dtype=torch.long)
        min_mask_prob = 0.15  # default reasonable value
        max_mask_prob = 0.5
        x_t, mask_positions = corrupt_with_mask(
            x, t, num_diffusion_steps, mask_token_id, min_mask_prob, max_mask_prob
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
        # Run denoising at a middle timestep for infilling
        # Use t = num_diffusion_steps // 2 for moderate corruption
        t = torch.tensor([num_diffusion_steps // 2], device=device, dtype=torch.long)

        logits = model(x_t, t)  # [1, seq_len, vocab_size]

        # Sample from logits for masked positions
        if temperature <= 0:
            predictions = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            predictions = torch.multinomial(probs.view(-1, 256), num_samples=1).view(
                1, -1
            )

        # Replace masked positions with predictions
        result = x_t.clone()
        result[mask_positions] = predictions[mask_positions]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    num_masked = int(mask_positions.sum().item())
    metrics = {
        "prompt_tokens": len(prompt),
        "masked_positions": num_masked,
        "infill_time_s": elapsed_s,
        "tokens_per_sec": (num_masked / elapsed_s) if elapsed_s > 0 else 0.0,
        "ms_per_token": ((elapsed_s * 1000.0) / num_masked) if num_masked > 0 else 0.0,
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

    # Encode prompt, replacing special characters if needed
    prompt_bytes = args.prompt.encode("utf-8", errors="replace")

    out_bytes, infill_metrics, mask_array = infill(
        model=model,
        prompt=prompt_bytes,
        max_length=args.max_length,
        num_diffusion_steps=num_diffusion_steps,
        mask_token_id=mask_token_id,
        device=device,
        use_amp=use_amp,
        temperature=args.temperature,
    )

    # Display result with mask positions highlighted
    print("Input prompt:")
    print(args.prompt)
    print("\nMask positions (if any):")
    mask_str = "".join(["^" if m else " " for m in mask_array[: len(out_bytes)]])
    print(mask_str)
    print("\nInfilled result:")
    # Filter out non-printable bytes and mask token (256) for display
    display_bytes = bytes([b for b in out_bytes if b < 128 and b >= 32])
    result_str = display_bytes.decode("utf-8", errors="replace")
    print(result_str)

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
