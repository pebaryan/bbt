# BitByte Position Paper

Date: February 13, 2026

## One-Sentence Position

BitByte exists to prove that useful, extensible language models can be trained and iterated on commodity hardware by treating bytes as the native interface and efficiency as a first-class architectural constraint [@xue2022byt5tokenfreefuturepretrained; @yu2023megabytepredictingmillionbytesequences; @pagnoni2024bytelatenttransformerpatches].

## Why This Project Exists

Most LM workflows assume large clusters, heavyweight tokenization pipelines, and infrastructure debt that blocks fast iteration for small teams [@kaplan2020scalinglawsneurallanguage; @hoffmann2022trainingcomputeoptimallargelanguage].  
BitByte takes the opposite path:

- Byte-native modeling to remove tokenizer coupling and simplify multimodal/text ingestion paths [@xue2022byt5tokenfreefuturepretrained; @clark2022caninepretrainingefficienttokenizationfree; @tay2022charformerfastcharactertransformers; @schmidt2024tokenizationcompression].
- Quantization-aware model components to stretch practical compute [@dettmers20228bitoptimizersblockwisequantization; @dettmers2022llmint88bitmatrixmultiplication; @ma2024era1bitllmslarge].
- Sharded binary data pipelines for low-friction, high-throughput training.
- A modular codebase where architecture experiments (Transformer vs Mamba) share one training spine [@gu2022efficientlymodelinglongsequences; @gu2024mambalineartimesequencemodeling; @dao2024transformersssmsgeneralizedmodels].

## Core Thesis

If we optimize for `clarity + efficiency + modularity`, then a single-GPU-first stack can still deliver meaningful model research velocity and usable outcomes [@eldan2023tinystoriessmalllanguagemodels; @dao2022flashattentionfastmemoryefficientexact].

This is not a claim that BitByte will beat frontier models.  
This is a claim that BitByte can become a strong engineering and research platform for:

- byte-level language modeling,
- long-context efficiency experiments,
- quantization-native training patterns,
- practical local-first iteration loops.

## Principles

- Efficiency is a product feature, not just an optimization pass.
- Byte-level simplicity beats tokenizer complexity unless quality data shows otherwise [@xue2022byt5tokenfreefuturepretrained; @pagnoni2024bytelatenttransformerpatches; @schmidt2024tokenizationcompression].
- One training interface should support multiple backbones.
- Experiments must be reproducible and comparable.
- Prefer measurable progress over architectural novelty for its own sake.

## What We Are Building

- A shared training/runtime substrate (`data/`, `training/`, `utils/`).
- Two model tracks on top of that substrate: `BitByteLM` (attention baseline) and `MambaMLM` (state-space long-context track).
- Quantized linear infrastructure (`quantization/`) used across tracks.
- Dataset ingestion and packing tools for byte shard generation.

## What We Are Not Building (For Now)

- A frontier-scale model training stack.
- Multi-node distributed orchestration.
- Product UX or deployment platform.
- Benchmark-chasing across many datasets before core reliability exists.

## Strategic Direction (Next 2 Quarters)

### Phase 1: Reliability and Reproducibility

- Establish a minimal test suite: import smoke, forward pass, one-step train loop.
- Lock a canonical baseline run config and expected metrics window.
- Add a validation/eval path with consistent bits-per-byte reporting.
- Ensure checkpoint resume behavior is verified and documented.

Exit criteria:

- Any commit can run a 1-5 minute smoke workflow and produce deterministic sanity metrics.

### Phase 2: Baseline Quality Track (Transformer)

- Produce a clean BitByteLM baseline on TinyStories shards.
- Run targeted ablations for `act_quant` on/off.
- Run targeted ablations for sequence curriculum vs fixed sequence length.
- Run targeted ablations for optimizer path (`torch AdamW` vs `bitsandbytes`).

Exit criteria:

- A short benchmark table with quality/speed/memory tradeoffs is part of repo docs.

### Phase 3: Long-Context Efficiency Track (Mamba)

- Bring Mamba training/eval to parity with Transformer experiment hygiene.
- Compare Transformer vs Mamba on bpb at matched parameter budget.
- Compare Transformer vs Mamba on throughput at long sequence lengths.
- Compare Transformer vs Mamba on memory scaling behavior.

Exit criteria:

- Decision memo: where Mamba is better, where it is not, and the default backbone recommendation for this repo.

### Phase 4: Project Productization (Internal)

- Promote stable commands and configs into a simple CLI or runbook.
- Standardize experiment artifact layout and naming.
- Document "new contributor path" from data prep to first result.

Exit criteria:

- A new contributor can reproduce baseline and one ablation in under one day.

## Success Metrics

- Quality: bits-per-byte on a fixed validation split.
- Stability: percent of runs that complete without manual intervention.
- Efficiency: tokens/sec (or bytes/sec), VRAM footprint, checkpoint size.
- Velocity: time from idea to comparable experiment result.

## Decision Framework

Adopt a change when at least one is true and none regress critically:

- Improves quality at same or lower compute.
- Improves throughput/memory at similar quality.
- Improves reliability/reproducibility.
- Simplifies architecture with neutral performance.

Reject or pause a direction when:

- It adds complexity without measurable gain in two consecutive evaluation cycles.
- It breaks compatibility with the shared training substrate for marginal benefit.

## Canonical Near-Term Objective

By end of next milestone cycle, BitByte should have:

- a reproducible Transformer baseline,
- a parity Mamba baseline,
- an explicit recommendation for when to use each,
- and a small but trustworthy evaluation harness for ongoing iteration.

That is the threshold where BitByte becomes a real platform, not just a set of promising scripts.

Citation format uses Pandoc keys (for example `[@gu2024mambalineartimesequencemodeling]`) and resolves against `references.bib`.
