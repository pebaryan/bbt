# BitByteLM Refactoring

This document describes the modular refactoring of `train_bitbyte.py` into smaller, more manageable components.

## Module Structure

```
bbt/
├── models/
│   ├── __init__.py          # Package exports
│   ├── bitbytelm.py         # Main BitByteLM model class
│   ├── block.py             # Transformer block with attention + MLP
│   ├── attention.py         # Causal self-attention with RoPE
│   ├── mlp.py               # MLP with SwiGLU activation
│   ├── rmsnorm.py           # RMS normalization
│   └── rope.py              # Rotary Position Embedding
├── quantization/
│   ├── __init__.py          # Package exports
│   ├── bitlinear.py         # BitLinear layer with quantization
│   ├── act_quant.py         # Activation quantization functions
│   └── ternary_quant.py     # Ternary quantization with STE
├── data/
│   ├── __init__.py          # Package exports
│   ├── byte_shard_dataset.py   # Byte shard dataset
│   └── byte_stream_dataset.py  # Byte stream dataset
├── training/
│   ├── __init__.py          # Package exports
│   ├── trainer.py           # Trainer class
│   ├── ddp.py               # Distributed training setup
│   ├── lr_scheduler.py      # Learning rate and seq_len schedulers
│   └── loss.py              # Loss metrics
├── utils/
│   ├── __init__.py          # Package exports
│   └── optim.py             # Optimizer and scaler helpers
├── train_bitbyte.py         # Original monolithic script
├── train_bitbyte_new.py     # New modularized training script
└── REFACTORING.md           # This file
```

## Components

### Quantization (`quantization/`)

Contains all quantization-related functionality:

- **`ternary_quant.py`**: Implements `TernaryQuantSTE` - ternary weight quantization with Straight-Through Estimator
- **`act_quant.py`**: Implements `act_quant_per_token` - symmetric per-token activation quantization
- **`bitlinear.py`**: Implements `BitLinear` - linear layer with quantized weights and optional activation quantization

### Models (`models/`)

Contains the neural network architecture:

- **`rope.py`**: Implements RoPE (Rotary Position Embedding) and `rotate_half` helper
- **`rmsnorm.py`**: Implements RMSNorm - simplified layer normalization
- **`attention.py`**: Implements `CausalSelfAttention` with:
  - RoPE embeddings
  - BitLinear projections
  - Optional SDPA (flash attention)
- **`mlp.py`**: Implements `MLP` with:
  - SwiGLU activation function
  - BitLinear projections
- **`block.py`**: Implements `Block` - single transformer block with:
  - RMSNorm
  - Causal self-attention
  - MLP
  - Optional gradient checkpointing
- **`bitbytelm.py`**: Implements `BitByteLM` - the main language model with:
  - Token embeddings
  - Transformer blocks
  - Final normalization and head

### Data (`data/`)

Contains dataset implementations:

- **`byte_shard_dataset.py`**: `ByteShardDataset` - loads from multiple shard files with glob pattern
- **`byte_stream_dataset.py`**: `ByteStreamDataset` - loads from single binary file

### Training (`training/`)

Contains training infrastructure:

- **`ddp.py`**: `setup_ddp` - initializes distributed training
- **`lr_scheduler.py`**: Learning rate and curriculum schedulers:
  - `lr_for_step`: Warmup + cosine decay
  - `seq_len_for_step`: Curriculum learning sequence length
- **`loss.py`**: `bits_per_byte` - converts loss to bits per byte metric
- **`trainer.py`**: `Trainer` class - encapsulates training loop with:
  - Resume functionality
  - Distributed training support
  - Mixed precision training
  - Gradient clipping
  - Logging and checkpointing

### Utils (`utils/`)

Contains utility functions:

- **`optim.py`**: Helper functions for:
  - `create_optimizer`: Creates optimizer with bitsandbytes fallback
  - `create_grad_scaler`: Creates gradient scaler for mixed precision

## Benefits of Refactoring

1. **Separation of Concerns**: Each module has a single responsibility
2. **Reusability**: Components can be easily reused in other projects
3. **Maintainability**: Smaller files are easier to understand and modify
4. **Testability**: Individual components can be tested in isolation
5. **Extensibility**: Easy to add new quantization methods, attention mechanisms, etc.

## Usage

The new modular structure can be used in two ways:

### 1. As a Standalone Script
```bash
python train_bitbyte_new.py --data /path/to/shards --ddp
```

### 2. As Importable Modules
```python
from models.bitbytelm import BitByteLM
from quantization.bitlinear import BitLinear
from data.byte_shard_dataset import ByteShardDataset

# Use components independently
model = BitByteLM(vocab_size=256, n_layer=12, d_model=768, n_head=12)
```

## Migration from Original Script

The original `train_bitbyte.py` has been split into:
- Model architecture → `models/`
- Quantization → `quantization/`
- Dataset → `data/`
- Training loop → `training/`
- Optimizer helpers → `utils/`

The new `train_bitbyte_new.py` is a simplified main script that uses these modules.

## Future Enhancements

Potential improvements:
- Add more quantization methods (e.g., FP8, int4)
- Add more attention mechanisms (e.g., FlashAttention, Linformer)
- Add more normalization methods (e.g., GroupNorm, LayerNorm)
- Add more dataset types (e.g., text-based, pre-tokenized)
- Add more training utilities (e.g., wandb logging, profiling)