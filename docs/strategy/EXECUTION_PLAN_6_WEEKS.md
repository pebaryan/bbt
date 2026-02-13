# BitByte Execution Plan (6 Weeks)

Date: February 13, 2026
Scope: reliability baseline, Transformer baseline, Mamba parity, and decision output.

## Mission for This Cycle

Convert BitByte from promising scripts into a reproducible experimentation platform with clear model recommendations.

## Cycle Outcomes

By end of week 6, this repo should provide:

- a reproducible Transformer baseline run,
- a parity Mamba baseline run,
- a compact comparison table (quality, throughput, memory),
- a short decision memo on default backbone choice.

## Primary Metrics

- Quality: bits per byte on fixed validation split.
- Throughput: bytes/sec or tokens/sec at fixed config.
- Memory: peak GPU VRAM during train loop.
- Stability: percent of runs that finish without manual intervention.

## Workstreams

## 1) Reliability and Harness

- Add smoke tests:
- import smoke (`models`, `training`, `data`).
- one forward pass test.
- one-step train loop test on tiny shard.
- Add a canonical baseline config and command docs.
- Validate checkpoint resume behavior with and without optimizer state.

Acceptance criteria:

- Fresh clone can run smoke checks in under 5 minutes.
- Resume behavior is documented and verified once per release cycle.

## 2) Transformer Baseline (BitByteLM)

- Lock a baseline config for `train_bitbyte_new.py`.
- Run baseline on TinyStories shards with fixed seed and log cadence.
- Run three focused ablations:
- `act_quant` on/off.
- sequence curriculum vs fixed `--seq_len`.
- optimizer path (`torch AdamW` vs `bitsandbytes` when available).

Acceptance criteria:

- Results table includes bpb, throughput, peak VRAM, and checkpoint size.

## 3) Mamba Parity Track

- Create a matching Mamba config at similar parameter budget.
- Run same measurement protocol as Transformer baseline.
- Identify regimes where Mamba wins or loses (longer sequence lengths are priority).

Acceptance criteria:

- Side-by-side table with matched-budget comparison and notes on tradeoffs.

## 4) Decision and Packaging

- Publish one short decision memo:
- default backbone recommendation,
- exceptions and known caveats,
- next experiment priorities.
- Update top-level docs with "start here" flow.

Acceptance criteria:

- New contributor can reproduce one baseline and one ablation in one day.

## Week-by-Week Plan

## Week 1

- Finalize baseline config files and smoke tests.
- Document canonical commands and artifact locations.

Deliverables:

- test skeleton,
- baseline command block,
- run artifact naming convention.

## Week 2

- Run Transformer baseline end to end.
- Capture first metrics snapshot and sanity check reproducibility.

Deliverables:

- baseline metrics row,
- reproducibility notes.

## Week 3

- Run Transformer ablations (`act_quant`, sequence schedule, optimizer path).
- Consolidate interim findings.

Deliverables:

- ablation table for Transformer track.

## Week 4

- Execute Mamba baseline with parity logging protocol.
- Resolve obvious measurement or stability gaps.

Deliverables:

- first Mamba baseline metrics row.

## Week 5

- Run Mamba comparison at long sequence settings.
- Finalize cross-model quality/speed/memory table.

Deliverables:

- matched-budget comparison table,
- long-context behavior notes.

## Week 6

- Write decision memo and update docs.
- Lock next-cycle priorities based on findings.

Deliverables:

- decision memo,
- updated contributor runbook.

## Experiment Logging Standard

For each run, record:

- commit hash,
- command line,
- model family,
- parameter count,
- dataset shard spec,
- seed,
- bpb,
- throughput,
- peak VRAM,
- wall-clock time,
- checkpoint path and size.

## Risks and Mitigations

- Risk: unstable metrics due to changing data splits.
- Mitigation: fixed validation split and seed policy.

- Risk: comparison bias from unmatched parameter budgets.
- Mitigation: publish budget matching assumptions in each table.

- Risk: environment drift across machines.
- Mitigation: document exact dependency and CUDA/PyTorch setup in run artifacts.

## Definition of Done for This Cycle

- At least one reproducible baseline per backbone.
- At least three meaningful ablations completed.
- A clear default-backbone decision backed by repo-local evidence.
