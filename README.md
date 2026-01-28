# BitByte Byte‑Level LM

Tiny, byte‑level language model experiments for dataset‑sharded training on a single GPU.

## Repo layout
- `train_bitbyte.py` – main training script (supports curriculum or fixed `--seq_len`, gradient accumulation, SDPA, checkpointing).
- `byte_shard_dataset.py` – mmap‑backed iterable dataset over `.bin` shards.
- `tinystories_json_to_shards.py` – convert TinyStories‑style JSON files to byte shards.
- `pack_to_shards.py` – generic packer for arbitrary text/code/data files into fixed‑size shards.

## Setup
1) Python 3.11+ with CUDA PyTorch. Example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install tqdm
   ```
2) (Optional) bitsandbytes for 8‑bit AdamW; the script falls back to torch AdamW.

## Prepare data
**TinyStories JSON → shards**
```bash
python tinystories_json_to_shards.py --input_dir TinyStories_all_data --out_dir TinyStories_shards --shard_size_gb 1
```

**Generic files → shards**
```bash
python pack_to_shards.py --inputs path/to/data --out_dir shards --shard_size_gb 2 --sep_byte 0
```

Result: `shards/shard_00000.bin`, `shard_00001.bin`, ...

## Train (12 GB GPU‑friendly)
```bash
python train_bitbyte.py --data TinyStories_shards \
  --steps 200000 --batch_size 1 --grad_accum 2 \
  --seq_len 1024 --no_act_quant --lr 1e-4 \
  --use_sdpa --log_every 1 --save_every 100
```
Notes:
- `--seq_len` fixes sequence length; omit to use the built‑in curriculum (2048→4096→8192). `--seq_len_cap` can cap it.
- Checkpointing is on by default; saves every `--save_every` steps (skips step 0) to `--out` (`ckpt.pt` default).
- For more headroom: lower `--grad_accum` or `--seq_len`; for more quality later, raise them or increase model size (`--n_layer`, `--d_model`, `--d_ff`).

## Resume / change save cadence
- To save more often: set `--save_every N` (e.g., 100).
- To resume, point `--out` to an existing checkpoint and load manually (simple load helper not yet wired; use standard PyTorch `torch.load`/`load_state_dict`).

## Tips
- Keep `--use_sdpa` for faster, lower‑memory attention.
- Leave checkpointing enabled unless memory is abundant.
- Watch `nvidia-smi` for memory/power; Task Manager reports dedicated+shared, so ~13.5 GB shown is normal on a 12 GB card if `nvidia-smi` is ~12 GB.
