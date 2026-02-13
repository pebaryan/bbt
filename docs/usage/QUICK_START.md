# Quick Start Guide for BitByteLM Refactoring

## Module Overview

### Quantization (`quantization/`)
- `ternary_quant.py` - Ternary weight quantization with STE
- `act_quant.py` - Per-token activation quantization
- `bitlinear.py` - BitLinear layer with quantization

### Models (`models/`)
- `rope.py` - Rotary Position Embedding
- `rmsnorm.py` - RMS normalization
- `attention.py` - Causal self-attention with RoPE
- `mlp.py` - MLP with SwiGLU activation
- `block.py` - Transformer block (norm + attn + mlp)
- `bitbytelm.py` - Main BitByteLM model

### Data (`data/`)
- `byte_shard_dataset.py` - Sharded dataset from glob pattern
- `byte_stream_dataset.py` - Single file dataset

### Training (`training/`)
- `ddp.py` - Distributed training setup
- `lr_scheduler.py` - Learning rate and curriculum schedulers
- `loss.py` - Bits per byte metric
- `trainer.py` - Trainer class

### Utils (`utils/`)
- `optim.py` - Optimizer and scaler creation

## Common Tasks

### Create a Model
```python
from models.bitbytelm import BitByteLM

model = BitByteLM(
    vocab_size=256,
    n_layer=24,
    d_model=1536,
    n_head=12,
    d_ff=4096,
)
```

### Create a Dataset
```python
from data.byte_shard_dataset import ByteShardDataset

ds = ByteShardDataset(
    shard_glob="data/shard_*.bin",
    seq_len=2048,
    rank=0,
    world_size=1,
)
```

### Create a BitLinear Layer
```python
from quantization.bitlinear import BitLinear

layer = BitLinear(
    in_features=1536,
    out_features=1536,
    bias=False,
    act_quant=True,
)
```

### Create an Optimizer
```python
from utils.optim import create_optimizer

opt, use_bnb = create_optimizer(
    model, lr=2e-4, weight_decay=0.1
)
```

### Create a Trainer
```python
from training.trainer import Trainer
from training.ddp import setup_ddp

rank, local_rank, world_size, is_ddp = setup_ddp(use_ddp=True)
device = torch.device("cuda", local_rank)

trainer = Trainer(
    model=model,
    optimizer=opt,
    scaler=scaler,
    rank=rank,
    local_rank=local_rank,
    world_size=world_size,
    is_ddp=is_ddp,
    args=args,
)

trainer.train(dataloader)
```

### Resume from Checkpoint
```python
trainer.resume(
    path="ckpt.pt",
    load_opt_state=True,  # or False to skip optimizer state
)
```

### Get Learning Rate
```python
from training.lr_scheduler import lr_for_step

lr = lr_for_step(
    step=1000,
    warmup_steps=6000,
    total_steps=200000,
    base_lr=2e-4,
)
```

### Get Sequence Length (Curriculum)
```python
from training.lr_scheduler import seq_len_for_step

seq_len = seq_len_for_step(
    step=1000,
    total_steps=200000,
    cap=8192,
)
```

## Command Line Usage

```bash
# Single GPU
python train_bitbyte_new.py --data /path/to/shards

# Distributed (4 GPUs)
python -m torchrun --nproc_per_node=4 train_bitbyte_new.py \
    --data /path/to/shards --ddp

# With custom architecture
python train_bitbyte_new.py \
    --data /path/to/shards \
    --n_layer 12 --d_model 768 --n_head 12 --d_ff 2048

# Resume training
python train_bitbyte_new.py \
    --data /path/to/shards \
    --resume ckpt.pt

# Resume without optimizer state
python train_bitbyte_new.py \
    --data /path/to/shards \
    --resume ckpt.pt --no_opt_state
```

## File Locations

| Original Location | New Location |
|-------------------|--------------|
| `train_bitbyte.py` (inline) | `quantization/ternary_quant.py` |
| `train_bitbyte.py` (inline) | `quantization/act_quant.py` |
| `train_bitbyte.py` (inline) | `quantization/bitlinear.py` |
| `train_bitbyte.py` (inline) | `models/rope.py` |
| `train_bitbyte.py` (inline) | `models/rmsnorm.py` |
| `train_bitbyte.py` (inline) | `models/attention.py` |
| `train_bitbyte.py` (inline) | `models/mlp.py` |
| `train_bitbyte.py` (inline) | `models/block.py` |
| `train_bitbyte.py` (inline) | `models/bitbytelm.py` |
| `train_bitbyte.py` (inline) | `data/byte_stream_dataset.py` |
| `train_bitbyte.py` (inline) | `training/ddp.py` |
| `train_bitbyte.py` (inline) | `training/lr_scheduler.py` |
| `train_bitbyte.py` (inline) | `training/loss.py` |
| `train_bitbyte.py` (main) | `train_bitbyte_new.py` |
| `train_bitbyte.py` (main) | `training/trainer.py` |