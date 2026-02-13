# BitByte: Public Position (One Page)

Date: February 13, 2026

## What BitByte Is

BitByte is a byte-level language modeling project focused on practical training and iteration on commodity hardware [@xue2022byt5tokenfreefuturepretrained; @yu2023megabytepredictingmillionbytesequences]. The goal is not frontier scale. The goal is a strong, reproducible platform for efficient LM experimentation.

## Why This Matters

Many LM workflows assume cluster-scale infrastructure and tokenizer-heavy pipelines [@kaplan2020scalinglawsneurallanguage; @hoffmann2022trainingcomputeoptimallargelanguage]. BitByte takes a simpler path:

- bytes as the native interface [@xue2022byt5tokenfreefuturepretrained; @pagnoni2024bytelatenttransformerpatches],
- sharded binary data for fast local training,
- quantization-aware components for tighter compute budgets [@dettmers20228bitoptimizersblockwisequantization; @dettmers2022llmint88bitmatrixmultiplication],
- one shared training stack for multiple backbones [@gu2024mambalineartimesequencemodeling; @dao2024transformersssmsgeneralizedmodels].

## Technical Position

BitByte treats efficiency, modularity, and reproducibility as first-class design constraints. If those constraints are respected, small teams can still run meaningful LM research loops quickly [@eldan2023tinystoriessmalllanguagemodels; @dao2022flashattentionfastmemoryefficientexact].

## What Exists Today

- Transformer track: `BitByteLM`
- State-space track: `MambaMLM`
- Shared training/data substrate: `training/`, `data/`, `utils/`
- Quantization building blocks: `quantization/`
- Single-GPU-friendly training entrypoints: `train_bitbyte.py`, `train_bitbyte_new.py`

## Direction

- Stabilize reliability first (smoke tests, deterministic baseline runs, resume checks).
- Build a clean Transformer benchmark baseline.
- Bring Mamba to experiment parity and compare quality/speed/memory at matched budgets.
- Publish a default backbone recommendation based on data, not preference.

## What Success Looks Like

- Reproducible baseline and ablations from a fresh checkout.
- Clear quality metric tracking (bits per byte on a fixed validation split).
- Known throughput and memory profiles for both model tracks.
- Faster idea-to-result cycle for contributors.

## Boundaries

BitByte is not currently a product platform, multi-node training system, or frontier-scale benchmark contender. It is an engineering and research platform optimized for disciplined iteration.

Citation format uses Pandoc keys and resolves against `references.bib`.
