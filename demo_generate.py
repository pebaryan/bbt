#!/usr/bin/env python3
"""Generate text from a trained autoregressive BitByte checkpoint."""

import argparse
from contextlib import nullcontext
import json
import time

import torch

from models.bitbytelm import BitByteLM


def _ckpt_arg(cfg: dict, key: str, default):
    return cfg.get(key, default)


def load_model(ckpt_path: str, device: torch.device) -> tuple[BitByteLM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model' state dict: {ckpt_path}")

    cfg = ckpt.get("args", {})
    if not isinstance(cfg, dict):
        cfg = vars(cfg)

    model = BitByteLM(
        vocab_size=256,
        n_layer=_ckpt_arg(cfg, "n_layer", 24),
        d_model=_ckpt_arg(cfg, "d_model", 1536),
        n_head=_ckpt_arg(cfg, "n_head", 12),
        d_ff=_ckpt_arg(cfg, "d_ff", 4096),
        act_quant=not _ckpt_arg(cfg, "no_act_quant", False),
        use_sdpa=_ckpt_arg(cfg, "use_sdpa", True),
        ckpt=not _ckpt_arg(cfg, "no_ckpt", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0 and top_k < logits.size(-1):
        vals, idx = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return idx.gather(-1, choice)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: BitByteLM,
    prompt: bytes,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    use_amp: bool,
) -> bytes:
    x = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad(), amp_ctx:
        for _ in range(max_new_tokens):
            logits = model(x)[:, -1, :]
            nxt = sample_next_token(logits=logits, temperature=temperature, top_k=top_k)
            x = torch.cat([x, nxt], dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    generated = x.size(1) - len(prompt)
    metrics = {
        "prompt_tokens": len(prompt),
        "generated_tokens": generated,
        "total_tokens": x.size(1),
        "generation_time_s": elapsed_s,
        "tokens_per_sec": (generated / elapsed_s) if elapsed_s > 0 else 0.0,
        "ms_per_token": ((elapsed_s * 1000.0) / generated) if generated > 0 else 0.0,
    }
    return bytes(x[0].tolist()), metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/checkpoints/ckpt.pt")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0, help="0 disables top-k filtering")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--metrics", action="store_true", help="Print performance metrics after generation")
    ap.add_argument("--metrics_json", type=str, default=None, help="Optional path to save metrics as JSON")
    args = ap.parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if args.top_k < 0:
        raise ValueError("--top_k must be >= 0")

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
    model, cfg = load_model(args.ckpt, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_time_s = time.perf_counter() - load_t0

    out_bytes, gen_metrics = generate(
        model=model,
        prompt=args.prompt.encode("utf-8", errors="replace"),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        use_amp=use_amp,
    )
    print(out_bytes.decode("utf-8", errors="replace"))

    num_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "device": str(device),
        "checkpoint": args.ckpt,
        "model_load_time_s": load_time_s,
        "parameter_count": int(num_params),
        "n_layer": _ckpt_arg(cfg, "n_layer", 24),
        "d_model": _ckpt_arg(cfg, "d_model", 1536),
        "n_head": _ckpt_arg(cfg, "n_head", 12),
        "d_ff": _ckpt_arg(cfg, "d_ff", 4096),
        "temperature": args.temperature,
        "top_k": args.top_k,
        **gen_metrics,
    }
    if device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

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
