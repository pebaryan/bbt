# Annotated Bibliography for BitByte

Date: February 13, 2026  
Companion file: `references.bib`

## How to Use This

This list is organized to support two goals:

- stronger project positioning (what BitByte is and why it matters),
- concrete experiment design (what analyses to run next).

Citation keys below match `references.bib`.

## A) Byte-Level and Tokenization-Free Modeling

`xue2022byt5tokenfreefuturepretrained`  
ByT5 is a core token-free baseline for byte-level pretraining. It is the most direct early evidence that byte-only modeling can be competitive with enough scale and architecture care.

`clark2022caninepretrainingefficienttokenizationfree`  
CANINE shows tokenization-free language representation with efficiency-oriented design choices. Useful as precedent that byte/character paths need explicit efficiency mechanisms.

`tay2022charformerfastcharactertransformers`  
Charformer demonstrates learned subword structure from characters/bytes. Relevant for the question: when should BitByte stay pure-byte versus adding learned compression stages.

`yu2023megabytepredictingmillionbytesequences`  
MEGABYTE provides a strong multiscale byte modeling direction for very long sequences. This supports BitByte's long-context positioning and motivates hierarchical extensions.

`pagnoni2024bytelatenttransformerpatches`  
Byte Latent Transformer argues that byte patches can scale better than token-based interfaces in some settings. This is high-value support for a byte-native roadmap.

`kallini2025mrt5dynamictokenmerging`  
MrT5 studies dynamic token merging for byte-level efficiency. Useful reference if BitByte explores adaptive sequence compression inside training.

`schmidt2024tokenizationcompression`  
This paper argues tokenization is not only compression. It is a useful counterweight for positioning: byte-native is a strategy with tradeoffs, not a universal win.

## B) Long-Context and Backbone Direction (Transformer vs SSM)

`gu2022efficientlymodelinglongsequences`  
S4 establishes the modern structured state space foundation for long-sequence modeling.

`gu2024mambalineartimesequencemodeling`  
Mamba is the primary selective SSM reference and strongest support for your Mamba track.

`dao2024transformersssmsgeneralizedmodels`  
Mamba-2 / SSM-Transformer duality helps frame BitByte as a comparative systems project rather than a binary architecture bet.

`dao2022flashattentionfastmemoryefficientexact`  
FlashAttention is a key efficiency baseline for attention models and should be included in any fairness discussion when comparing Transformer and SSM speed/memory.

`tay2020longrangearenabenchmark`  
LRA is the classic long-range efficiency benchmark reference.

`pmlr-v202-zhang23r`  
CAB broadens efficient-attention evaluation beyond basic self-attention patterns. Useful for arguing evaluation breadth, not only single-task wins.

## C) Quantization, Compute Budgeting, and Practical Training

`dettmers20228bitoptimizersblockwisequantization`  
Justifies low-memory optimizer choices and supports your optional bitsandbytes path.

`dettmers2022llmint88bitmatrixmultiplication`  
A core reference for practical low-bit matrix computation in LMs.

`xiao2024smoothquantaccurateefficientposttraining`  
Strong post-training quantization reference; good for future inference-focused branch if BitByte adds eval-serving loops.

`dettmers2023qloraefficientfinetuningquantized`  
QLoRA is the dominant memory-efficient finetuning reference; useful for future adaptation workflows.

`ma2024era1bitllmslarge`  
BitNet 1.58-bit gives a provocative low-bit direction aligned with BitByte's efficiency-first identity.

`kaplan2020scalinglawsneurallanguage`  
Scaling laws provide the baseline framing for compute/data/parameter tradeoffs.

`hoffmann2022trainingcomputeoptimallargelanguage`  
Chinchilla-style compute-optimal guidance is important for deciding model size versus data size in your baseline plans.

`eldan2023tinystoriessmalllanguagemodels`  
TinyStories supports your current dataset choice as a legitimate small-model capability testbed.

## Positioning Claims You Can Defend with This Literature

1. BitByte is byte-native by design, not by omission.  
Support: ByT5, MEGABYTE, BLT, MrT5, Tokenization Is More Than Compression.

2. BitByte is architecture-plural and evidence-driven.  
Support: S4, Mamba, Mamba-2, FlashAttention, LRA, CAB.

3. BitByte is compute-aware and practical.  
Support: 8-bit Optimizers, LLM.int8, QLoRA, BitNet, Scaling Laws, Chinchilla.

## High-Value Analyses to Run Next

1. Backbone frontier map (Transformer vs Mamba)  
Measure bpb, throughput, and peak VRAM across matched parameter budgets and sequence lengths (`1k`, `2k`, `4k`, `8k`).  
References: `gu2024mambalineartimesequencemodeling`, `dao2024transformersssmsgeneralizedmodels`, `dao2022flashattentionfastmemoryefficientexact`.

2. Byte-efficiency interventions  
Compare pure-byte baseline vs lightweight learned compression/merging variants (if added), with fixed compute budget.  
References: `tay2022charformerfastcharactertransformers`, `kallini2025mrt5dynamictokenmerging`, `pagnoni2024bytelatenttransformerpatches`.

3. Quantization impact matrix  
Ablate activation quantization and optimizer precision paths; track stability and quality degradation thresholds.  
References: `dettmers20228bitoptimizersblockwisequantization`, `dettmers2022llmint88bitmatrixmultiplication`, `ma2024era1bitllmslarge`.

4. Compute-optimal sweep for this repo scale  
Run small grid over model size and effective data exposure to approximate local "Chinchilla point" for BitByte.  
References: `kaplan2020scalinglawsneurallanguage`, `hoffmann2022trainingcomputeoptimallargelanguage`.

5. Evaluation robustness pass  
Keep TinyStories as base, then add at least one long-context stress task with consistent reporting protocol.  
References: `eldan2023tinystoriessmalllanguagemodels`, `tay2020longrangearenabenchmark`, `pmlr-v202-zhang23r`.
