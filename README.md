# BitByte Byte‑Level LM

Tiny, byte‑level language model experiments for dataset‑sharded training on a single GPU.

## Project Direction

See `docs/strategy/POSITION_PAPER.md` for the unifying narrative, technical thesis, and roadmap.
See `docs/strategy/POSITION_PAPER_PUBLIC.md` for a one-page public version.
See `docs/strategy/EXECUTION_PLAN_6_WEEKS.md` for the concrete execution plan.
See `docs/strategy/ANNOTATED_BIBLIOGRAPHY.md` and `docs/strategy/references.bib` for supporting literature and analysis directions.

## Render Papers with Citations

Requires `pandoc` on `PATH`. For PDF output, install a LaTeX engine (for example `xelatex`).

```bash
# Full position paper -> HTML
pandoc docs/strategy/POSITION_PAPER.md --standalone --citeproc \
  --bibliography docs/strategy/references.bib \
  -o artifacts/reports/POSITION_PAPER.html

# Full position paper -> PDF
pandoc docs/strategy/POSITION_PAPER.md --citeproc \
  --bibliography docs/strategy/references.bib \
  --pdf-engine=xelatex -o artifacts/reports/POSITION_PAPER.pdf

# Public one-pager -> HTML
pandoc docs/strategy/POSITION_PAPER_PUBLIC.md --standalone --citeproc \
  --bibliography docs/strategy/references.bib \
  -o artifacts/reports/POSITION_PAPER_PUBLIC.html

# Public one-pager -> PDF
pandoc docs/strategy/POSITION_PAPER_PUBLIC.md --citeproc \
  --bibliography docs/strategy/references.bib \
  --pdf-engine=xelatex -o artifacts/reports/POSITION_PAPER_PUBLIC.pdf
```

## Repo layout
- `train_bitbyte.py` – main training script (supports curriculum or fixed `--seq_len`, gradient accumulation, SDPA, checkpointing).
- `train_mamba_new.py` – MambaLM training script with checkpoint family safeguards.
- `train_bitbyte_diffusion.py` – masked-diffusion byte denoiser training script (separate from AR path).
- `data/byte_shard_dataset.py` – primary iterable dataset over `.bin` shards.
- `scripts/legacy/byte_shard_dataset.py` – legacy mmap-backed dataset implementation.
- `scripts/tinystories_json_to_shards.py` – convert TinyStories‑style JSON files to byte shards.
- `scripts/pack_to_shards.py` – generic packer for arbitrary text/code/data files into fixed‑size shards.
- `docs/` – architecture notes, usage guides, and strategy papers.

## Setup
1) Python 3.11+ with CUDA PyTorch. Example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install tqdm
   ```
2) (Optional) bitsandbytes for 8‑bit AdamW; the script falls back to torch AdamW.

## Artifact layout
Local datasets, checkpoints, logs, and plots are stored under:
- `artifacts/checkpoints/`
- `artifacts/datasets/`
- `artifacts/logs/`
- `artifacts/plots/`

## Prepare data
**TinyStories JSON → shards**
```bash
python scripts/tinystories_json_to_shards.py \
  --input_dir artifacts/datasets/tinystories/raw_json/data \
  --out_dir artifacts/datasets/tinystories/shards/data \
  --shard_size_gb 1
```

**Generic files → shards**
```bash
python scripts/pack_to_shards.py \
  --inputs path/to/data \
  --out_dir artifacts/datasets/custom/shards \
  --shard_size_gb 2 \
  --sep_byte 0
```

Result: `artifacts/datasets/.../shard_00000.bin`, `shard_00001.bin`, ...

## Train (12 GB GPU‑friendly)
```bash
python train_bitbyte.py --data artifacts/datasets/tinystories/shards/data \
  --steps 200000 --batch_size 1 --grad_accum 2 \
  --seq_len 1024 --no_act_quant --lr 1e-4 \
  --use_sdpa --log_every 1 --save_every 100 \
  --out artifacts/checkpoints/ckpt.pt
```
Notes:
- `--seq_len` fixes sequence length; omit to use the built‑in curriculum (2048→4096→8192). `--seq_len_cap` can cap it.
- Checkpointing is on by default; saves every `--save_every` steps (skips step 0) to `--out`.
- For more headroom: lower `--grad_accum` or `--seq_len`; for more quality later, raise them or increase model size (`--n_layer`, `--d_model`, `--d_ff`).
- Safety guard: if `--out` already exists, training aborts unless you pass `--allow_overwrite`.

## Resume / change save cadence
- To save more often: set `--save_every N` (e.g., 100).
- To resume, point `--out` to an existing checkpoint and load manually (simple load helper not yet wired; use standard PyTorch `torch.load`/`load_state_dict`).

## Tips
- Keep `--use_sdpa` for faster, lower‑memory attention.
- Leave checkpointing enabled unless memory is abundant.
- Watch `nvidia-smi` for memory/power; Task Manager reports dedicated+shared, so ~13.5 GB shown is normal on a 12 GB card if `nvidia-smi` is ~12 GB.

## Train Diffusion Variant
```bash
python train_bitbyte_diffusion.py --data artifacts/datasets/tinystories/shards/data \
  --steps 200000 --batch_size 1 --grad_accum 2 \
  --seq_len 1024 --lr 1e-4 --use_sdpa \
  --diffusion_steps 64 --min_mask_prob 0.05 --max_mask_prob 0.5 \
  --out artifacts/checkpoints/diffusion/ckpt_diffusion.pt
```
Notes:
- This is a separate denoising objective and does not change autoregressive training code.
- Logged `masked_bpb` is bits-per-byte on masked positions only.
- Safety guard: if `--out` already exists, training aborts unless you pass `--allow_overwrite`.

PowerShell launcher with timestamped checkpoint output:
```powershell
pwsh -File scripts/run_diffusion.ps1
```

## Train Mamba Variant
```bash
python train_mamba_new.py --data artifacts/datasets/tinystories/shards/data \
  --steps 200000 --batch_size 1 --grad_accum 2 \
  --seq_len 1024 --lr 1e-4 \
  --out artifacts/checkpoints/mamba/ckpt_mamba.pt
```
Notes:
- Mamba checkpoints are tagged with `model_family: mamba`.
- Resume safety rejects non-Mamba checkpoints.
- Safety guard: if `--out` already exists, training aborts unless you pass `--allow_overwrite`.

PowerShell launcher with timestamped checkpoint output:
```powershell
pwsh -File scripts/run_mamba.ps1
```

## Tag Legacy Checkpoints
Use this for old checkpoints that were saved before variant/family metadata existed.

```bash
# Inspect what tags would be applied
python scripts/tag_checkpoint.py --ckpt artifacts/checkpoints/ckpt.pt --dry_run

# Tag in place as BitByte AR
python scripts/tag_checkpoint.py --ckpt artifacts/checkpoints/ckpt.pt --inplace --variant ar --model_family bitbyte --allow_overwrite

# Tag a legacy Mamba checkpoint
python scripts/tag_checkpoint.py --ckpt artifacts/checkpoints/mamba/old.pt --inplace --variant ar --model_family mamba --allow_overwrite
```

Bulk scan/tag all checkpoints:
```bash
# Dry-run across all .pt files under artifacts/checkpoints
python scripts/tag_all_checkpoints.py

# Apply tags in place
python scripts/tag_all_checkpoints.py --apply
```

Smoke test (CPU-safe, synthetic data, 10 steps):
```bash
python train_bitbyte_diffusion.py --smoke_test
```

## Validation Results

All checkpoints have been validated on `artifacts/datasets/misc_shards/shard_00003.bin`.

### BitByteLM (Autoregressive)
| Checkpoint | Validation BPB | Training Steps | Model Size |
|-----------|----------------|----------------|------------|
| `ckpt.pt` | **0.7005** | 25,400 | 24L/1536D |
| `ckpt-9200-1024.pt` | 0.9094 | 9,200 | 24L/1536D |

### Diffusion Models (Denoising)
| Checkpoint | Masked BPB | Training Steps | Model Size |
|-----------|------------|----------------|------------|
| `ckpt_diffusion_final.pt` | **0.0273** | 500 | 2L/128D |
| `ckpt_diffusion_1k.pt` | 0.0278 | 1,000 | (varies) |
| `ckpt_diffusion_5k_final.pt` | 0.0335 | 5,000 | (varies) |
| `ckpt_diffusion.pt` | 0.0346 | 300 | 6L/320D |

### Mamba Models
| Checkpoint | Validation BPB | Training Steps | Model Size |
|-----------|----------------|----------------|------------|
| `ckpt_mamba.pt` | **4.0401** | 100 | 6L/384D |
| `ckpt_mamba_500.pt` | 4.4795 | 500 | 6L/384D |

**Notes:**
- Diffusion BPB measures denoising loss on masked tokens only (not directly comparable to AR)
- Mamba models were trained for very few steps; longer training needed for competitive performance

## Demo Generation

### BitByteLM (Autoregressive)
```bash
python demo_generate.py \
  --ckpt artifacts/checkpoints/ckpt.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 300 \
  --temperature 0.9 \
  --top_k 50
```
Notes:
- Use `--temperature 0` for greedy decoding.
- Use `--device cpu` if CUDA is unavailable.
- Add `--metrics` to print latency/throughput/memory stats.
- Add `--metrics_json artifacts/reports/generation_metrics.json` to save stats.

### Mamba
```bash
python demo_mamba.py \
  --ckpt artifacts/checkpoints/mamba/ckpt_mamba.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 300 \
  --temperature 0.9 \
  --top_k 50
```
Notes:
- Same interface as `demo_generate.py` for autoregressive generation.
- Mamba models use state-space layers instead of attention.

### Diffusion (Infilling)
```bash
python demo_diffusion.py \
  --ckpt artifacts/checkpoints/diffusion/ckpt_diffusion.pt \
  --prompt "Once upon a time there was a " \
  --max_length 50 \
  --temperature 1.0
```
Notes:
- Diffusion models perform **infilling** rather than left-to-right generation.
- The model predicts masked positions in the input sequence.
- If no mask token (byte 256) is present, random positions are masked.
- Use `--temperature 0` for greedy decoding of masked positions.
- Add `--metrics` to print performance stats.
