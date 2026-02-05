# Mamba Integration with BitByteLM

This document describes the Mamba integration with BitByteLM, creating a Mamba variant that uses BitByte quantization techniques.

## Architecture Overview

### Mamba Block Structure

```
MambaBlock:
  ├── RMSNorm → MambaSSM (SSM layer)
  ├── RMSNorm → MLP (SwiGLU)
  └── Residual connections
```

### Mamba SSM Structure

```
MambaSSM:
  ├── In Projection: BitLinear (d_model → 2*d_inner)
  ├── Conv1D: Local processing (d_inner, kernel=d_conv)
  ├── Delta Projection: BitLinear (d_inner → d_state)
  ├── SSM Kernel: Diagonal state space
  ├── B/C Projections: BitLinear
  └── Out Projection: BitLinear (d_inner → d_model)
```

## Components

### `models/mamba/ssm.py`

- **`S4Kernel`**: Base S4 state space model with diagonal discretization
- **`SelectiveSSM`**: Selective SSM with input-dependent parameters

### `models/mamba/mamba_ssm.py`

- **`MambaSSM`**: Complete Mamba SSM implementation with:
  - Convolution for local processing
  - Selective delta computation
  - Diagonal state space discretization

### `models/mamba/mamba_block.py`

- **`MambaBlock`**: Mamba-based transformer block (replaces `Block`)

### `models/mamba_mlm.py`

- **`MambaMLM`**: Mamba-based language model (replaces `BitByteLM`)

## Usage

### Create a Mamba Model

```python
from models.mamba_mlm import MambaMLM

model = MambaMLM(
    vocab_size=256,
    n_layer=24,
    d_model=1536,
    d_ff=4096,
    d_state=16,
    d_conv=4,
    expand=2,
    act_quant=True,
)
```

### Create a Mamba Block

```python
from models.mamba import MambaBlock

block = MambaBlock(
    d_model=1536,
    d_ff=4096,
    d_state=16,
    d_conv=4,
    expand=2,
    act_quant=True,
)
```

### Use in Training

```python
from models.mamba_mlm import MambaMLM
from training.trainer import Trainer

model = MambaMLM(...)
trainer = Trainer(model, optimizer, scaler, ...)
trainer.train(dataloader)
```

## Key Differences from BitByteLM

| Feature | BitByteLM | MambaMLM |
|---------|-----------|----------|
| Attention | CausalSelfAttention | MambaSSM |
| Block | Block | MambaBlock |
| Complexity | O(n²) attention | O(n) SSM |
| Context | Limited by memory | Long context efficient |

## Quantization Considerations

The Mamba integration uses BitLinear for:
- Input/output projections
- Delta projection
- B/C matrix projections

The SSM parameters (A, B, C) are currently floating-point. Future work could include:
- Quantizing A matrix diagonal
- Quantizing B/C matrices
- Quantizing delta (with special handling for softplus)

## Performance Characteristics

- **Training**: More sequential, may be slower on GPUs with poor sequential performance
- **Inference**: Can be faster with optimized SSM kernels
- **Memory**: O(n) instead of O(n²) for attention
- **Long Context**: Better scalability for very long sequences

## Future Improvements

1. **Quantized SSM**: Quantize A, B, C matrices with STE
2. **Hybrid Model**: Mix Mamba and attention layers
3. **Optimized Kernels**: Use CUDA kernels for SSM
4. **FlashAttention-like**: Develop flash-SSM for faster computation
5. **Mixed Precision**: FP16 for SSM, INT8 for projections