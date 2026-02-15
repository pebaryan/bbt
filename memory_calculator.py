#!/usr/bin/env python3
"""Memory calculator for BitByteLM and Mamba models.

Estimates GPU memory requirements for training and inference.
"""

import argparse


def format_bytes(bytes_val: float) -> str:
    """Convert bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def calculate_bitbyte_memory(
    vocab_size: int,
    d_model: int,
    n_layer: int,
    n_head: int,
    d_ff: int,
    seq_len: int,
    batch_size: int = 1,
    grad_accum: int = 1,
) -> dict:
    """Calculate memory requirements for BitByteLM (transformer)."""

    # Embedding parameters
    embed_params = vocab_size * d_model

    # Per-layer parameters
    # Attention: Q, K, V, O projections
    attn_params_per_layer = 4 * (d_model * d_model)  # QKV + O
    # FFN: up-proj + down-proj
    ffn_params_per_layer = d_model * d_ff + d_ff * d_model
    # Layer norms (2 per layer)
    ln_params_per_layer = 2 * d_model

    layer_params = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    total_layer_params = n_layer * layer_params

    # Final layer norm + LM head
    final_ln_params = d_model
    lm_head_params = d_model * vocab_size

    # Total parameters
    total_params = embed_params + total_layer_params + final_ln_params + lm_head_params

    # Memory calculations (assuming float32 for simplicity, adjust for actual dtype)
    bytes_per_param = 4  # float32

    # Model weights memory
    model_memory = total_params * bytes_per_param

    # Activations memory (rough estimate for forward pass)
    # Each layer produces: attention scores + attention output + FFN activations
    activations_per_layer = (
        batch_size * seq_len * d_model  # attention output
        + batch_size * seq_len * d_model  # FFN intermediate
        + batch_size * seq_len * d_model  # FFN output
    ) * bytes_per_param
    total_activations = activations_per_layer * n_layer

    # Training memory (gradients + optimizer states)
    gradient_memory = total_params * bytes_per_param
    # AdamW stores 2 momentum states per parameter
    optimizer_memory = total_params * 2 * bytes_per_param

    # Effective batch size with gradient accumulation
    effective_batch = batch_size * grad_accum

    # Inference memory (weights + 1x activations)
    inference_memory = model_memory + total_activations

    # Training memory (weights + gradients + optimizer + activations)
    # Note: Activations scale with batch size
    training_memory = (
        model_memory
        + gradient_memory
        + optimizer_memory
        + total_activations * effective_batch
    )

    return {
        "architecture": "BitByteLM (Transformer)",
        "total_params": total_params,
        "model_memory": model_memory,
        "activations_memory": total_activations,
        "gradient_memory": gradient_memory,
        "optimizer_memory": optimizer_memory,
        "inference_memory": inference_memory,
        "training_memory": training_memory,
    }


def calculate_mamba_memory(
    vocab_size: int,
    d_model: int,
    n_layer: int,
    d_ff: int,
    d_state: int,
    d_conv: int,
    expand: int,
    seq_len: int,
    batch_size: int = 1,
    grad_accum: int = 1,
) -> dict:
    """Calculate memory requirements for Mamba model."""

    # Embedding parameters
    embed_params = vocab_size * d_model

    # Per-layer parameters
    # Standard components
    ln_params_per_layer = d_model  # RMSNorm
    mlp_params_per_layer = d_model * d_ff + d_ff * d_model  # MLP

    # Mamba SSM parameters
    # SSM expansion: d_inner = expand * d_model
    d_inner = expand * d_model

    # x_proj: projects input to delta, B, C
    x_proj_params = d_model * (d_state * 2 + d_inner)  # delta + B + C
    # dt_proj: projects delta
    dt_proj_params = d_inner * d_inner
    # A and D parameters
    ssm_params = d_state * d_inner + d_inner  # A_log + D
    # Convolution
    conv_params = d_conv * d_inner

    mamba_params_per_layer = x_proj_params + dt_proj_params + ssm_params + conv_params

    layer_params = mamba_params_per_layer + mlp_params_per_layer + ln_params_per_layer
    total_layer_params = n_layer * layer_params

    # Final layer norm + LM head
    final_ln_params = d_model
    lm_head_params = d_model * vocab_size

    # Total parameters
    total_params = embed_params + total_layer_params + final_ln_params + lm_head_params

    # Memory calculations
    bytes_per_param = 4  # float32

    model_memory = total_params * bytes_per_param

    # Activations (Mamba uses recurrent scans, less memory than attention)
    activations_per_layer = (
        batch_size * seq_len * d_inner  # SSM states
        + batch_size * seq_len * d_model  # output
    ) * bytes_per_param
    total_activations = activations_per_layer * n_layer

    # Training memory
    gradient_memory = total_params * bytes_per_param
    optimizer_memory = total_params * 2 * bytes_per_param  # AdamW states

    effective_batch = batch_size * grad_accum

    inference_memory = model_memory + total_activations
    training_memory = (
        model_memory
        + gradient_memory
        + optimizer_memory
        + total_activations * effective_batch
    )

    return {
        "architecture": "Mamba",
        "total_params": total_params,
        "model_memory": model_memory,
        "activations_memory": total_activations,
        "gradient_memory": gradient_memory,
        "optimizer_memory": optimizer_memory,
        "inference_memory": inference_memory,
        "training_memory": training_memory,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for language models"
    )
    ap.add_argument(
        "--arch",
        type=str,
        default="mamba",
        choices=["bitbyte", "mamba"],
        help="Model architecture",
    )
    ap.add_argument("--d_model", type=int, default=384, help="Model dimension")
    ap.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    ap.add_argument("--d_ff", type=int, default=768, help="Feed-forward dimension")
    ap.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    ap.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    ap.add_argument("--vocab_size", type=int, default=256, help="Vocabulary size")

    # BitByteLM specific
    ap.add_argument(
        "--n_head",
        type=int,
        default=12,
        help="Number of attention heads (BitByteLM only)",
    )

    # Mamba specific
    ap.add_argument(
        "--d_state", type=int, default=16, help="SSM state dimension (Mamba only)"
    )
    ap.add_argument(
        "--d_conv", type=int, default=4, help="Convolution kernel size (Mamba only)"
    )
    ap.add_argument(
        "--expand", type=int, default=2, help="SSM expansion factor (Mamba only)"
    )

    args = ap.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Memory Calculator for {args.arch.upper()}")
    print(f"{'=' * 60}\n")

    print("Configuration:")
    print(f"  Architecture: {args.arch}")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layer: {args.n_layer}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  grad_accum: {args.grad_accum}")
    print(f"  vocab_size: {args.vocab_size}")

    if args.arch == "bitbyte":
        print(f"  n_head: {args.n_head}")
        result = calculate_bitbyte_memory(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_ff=args.d_ff,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
    else:
        print(f"  d_state: {args.d_state}")
        print(f"  d_conv: {args.d_conv}")
        print(f"  expand: {args.expand}")
        result = calculate_mamba_memory(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            d_ff=args.d_ff,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )

    print(f"\n{'=' * 60}")
    print("Memory Estimates")
    print(f"{'=' * 60}\n")

    print(f"Total Parameters: {result['total_params']:,}")
    print(f"  ({result['total_params'] / 1e6:.2f}M parameters)\n")

    print("Memory Breakdown:")
    print(f"  Model Weights:     {format_bytes(result['model_memory'])}")
    print(
        f"  Activations:       {format_bytes(result['activations_memory'])} (per sample)"
    )
    print(f"  Gradients:         {format_bytes(result['gradient_memory'])}")
    print(f"  Optimizer States:  {format_bytes(result['optimizer_memory'])}")

    print(f"\n{'=' * 60}")
    print("Total Memory Requirements:")
    print(f"{'=' * 60}")
    print(f"  Inference Only:    {format_bytes(result['inference_memory'])}")
    print(f"  Training:          {format_bytes(result['training_memory'])}")

    # Add safety margin recommendation
    safety_margin = 1.3  # 30% extra
    print(
        f"\nRecommended GPU Memory (with {int((safety_margin - 1) * 100)}% safety margin):"
    )
    print(
        f"  Inference:         {format_bytes(result['inference_memory'] * safety_margin)}"
    )
    print(
        f"  Training:          {format_bytes(result['training_memory'] * safety_margin)}"
    )

    # Check common GPU sizes
    print(f"\n{'=' * 60}")
    print("GPU Compatibility:")
    print(f"{'=' * 60}")

    gpu_sizes = [
        (8, "8 GB"),
        (12, "12 GB"),
        (16, "16 GB"),
        (24, "24 GB"),
        (40, "40 GB"),
        (48, "48 GB"),
        (80, "80 GB"),
    ]

    training_gb = result["training_memory"] / (1024**3)

    for gb, name in gpu_sizes:
        if training_gb <= gb * 0.85:  # Leave 15% overhead
            status = "[OK] FITS"
        elif training_gb <= gb:
            status = "[!] TIGHT"
        else:
            status = "[X] TOO LARGE"
        print(f"  {name}: {status}")

    print()


if __name__ == "__main__":
    main()
