# Scripts

- `tinystories_json_to_shards.py`: convert TinyStories JSON files into byte shards.
- `pack_to_shards.py`: pack arbitrary files into fixed-size shard binaries.
- `plot_loss.py`: parse training log and render a loss curve plot.
- `tag_checkpoint.py`: tag legacy checkpoints with `variant` and `model_family` metadata.
- `tag_all_checkpoints.py`: scan and tag all legacy checkpoints under a root folder.
- `migrate_diffusion_steps.py`: resize diffusion `t_emb` for a new `diffusion_steps` value.
- `run_mamba.ps1`: launch a Mamba training run with timestamped checkpoint output.
- `run_diffusion.ps1`: launch a diffusion run with timestamped checkpoint output.
- `legacy/validate.py`: legacy validation helper (now defaults to `artifacts/` paths).
- `legacy/byte_shard_dataset.py`: legacy mmap dataset implementation.

Compatibility wrappers remain in repo root for:
- `tinystories_json_to_shards.py`
- `pack_to_shards.py`
- `plot_loss.py`
- `validate.py`
- `byte_shard_dataset.py`
