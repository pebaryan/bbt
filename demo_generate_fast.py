#!/usr/bin/env python3
"""Generate text from an extracted fast ternary checkpoint."""

import argparse
from contextlib import nullcontext
import time

import torch

from models.bitbytelm_fast import FastBitByteLM


def _ckpt_arg(cfg: dict, key: str, default):
    return cfg.get(key, default)


def load_model(ckpt_path: str, device: torch.device, compile_model: bool) -> tuple[FastBitByteLM, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model' state dict: {ckpt_path}")
    if ckpt.get("format") != "fast_ternary_v1":
        raise ValueError(
            f"Expected extracted fast checkpoint format 'fast_ternary_v1', got: {ckpt.get('format')}"
        )

    cfg = ckpt.get("args", {})
    if not isinstance(cfg, dict):
        cfg = vars(cfg)

    model = FastBitByteLM(
        vocab_size=256,
        n_layer=_ckpt_arg(cfg, "n_layer", 24),
        d_model=_ckpt_arg(cfg, "d_model", 1536),
        n_head=_ckpt_arg(cfg, "n_head", 12),
        d_ff=_ckpt_arg(cfg, "d_ff", 4096),
        rope_base=_ckpt_arg(cfg, "rope_base", 10000.0),
        use_sdpa=_ckpt_arg(cfg, "use_sdpa", True),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    return model, cfg


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
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
    model: FastBitByteLM,
    prompt: bytes,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[bytes, dict]:
    prompt_ids = list(prompt)
    x_prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad(), amp_ctx:
        max_cache_len = len(prompt_ids) + max_new_tokens
        logits, kv_cache = model.forward_with_cache(
            x_prompt,
            kv_cache=None,
            start_pos=0,
            max_cache_len=max_cache_len,
        )
        logits = logits[:, -1, :]
        generated_ids: list[int] = []
        cur_pos = len(prompt_ids)

        for i in range(max_new_tokens):
            nxt = sample_next_token(logits=logits, temperature=temperature, top_k=top_k)
            token_id = int(nxt.item())
            generated_ids.append(token_id)
            if i + 1 >= max_new_tokens:
                break
            x_step = torch.tensor([[token_id]], dtype=torch.long, device=device)
            logits, kv_cache = model.forward_with_cache(
                x_step,
                kv_cache=kv_cache,
                start_pos=cur_pos,
                max_cache_len=max_cache_len,
            )
            logits = logits[:, -1, :]
            cur_pos += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    generated = len(generated_ids)
    metrics = {
        "generated_tokens": generated,
        "generation_time_s": elapsed_s,
        "tokens_per_sec": (generated / elapsed_s) if elapsed_s > 0 else 0.0,
    }
    return bytes(prompt_ids + generated_ids), metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0, help="0 disables top-k filtering")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--compile", action="store_true", help="Use torch.compile for speed.")
    ap.add_argument("--metrics", action="store_true", help="Print generation speed metrics.")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = device.type == "cuda"
    model, _ = load_model(args.ckpt, device=device, compile_model=args.compile)
    out_bytes, metrics = generate(
        model=model,
        prompt=args.prompt.encode("utf-8", errors="replace"),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        use_amp=use_amp,
    )
    print(out_bytes.decode("utf-8", errors="replace"))
    if args.metrics:
        print(f"tokens_per_sec: {metrics['tokens_per_sec']:.2f}")
        print(f"generation_time_s: {metrics['generation_time_s']:.3f}")


if __name__ == "__main__":
    main()
