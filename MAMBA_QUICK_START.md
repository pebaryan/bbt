# Mamba Quick Start

## Import Mamba Components

```python
from models.mamba import MambaBlock, MambaSSM, SelectiveSSM
from models.mamba_mlm import MambaMLM
```

## Create a Mamba Model

```python
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

## Mamba Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_state` | 16 | State dimension (N) |
| `d_conv` | 4 | Convolution kernel size |
| `expand` | 2 | Expansion factor for inner dimension |
| `act_quant` | True | Quantize activations |

## Mamba vs Attention

| Aspect | Attention | Mamba |
|--------|-----------|-------|
| Complexity | O(n²) | O(n) |
| Long context | Memory intensive | Efficient |
| Parallelizable | Yes (GPU) | Sequential |

## Hybrid Approach

You can mix Mamba and attention layers:

```python
from models.block import Block
from models.mamba import MambaBlock

# First half Mamba, second half attention
blocks = []
for i in range(n_layer):
    if i < n_layer // 2:
        blocks.append(MambaBlock(d_model, d_ff))
    else:
        blocks.append(Block(d_model, n_head, d_ff))
```

## Training

Use the same training infrastructure:

```python
from training.trainer import Trainer

trainer = Trainer(model, optimizer, scaler, ...)
trainer.train(dataloader)
```

## Checkpointing

Mamba models use the same checkpoint format:

```python
trainer.resume("ckpt.pt")
```