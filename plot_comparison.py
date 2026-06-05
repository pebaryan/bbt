#!/usr/bin/env python3
"""Generate loss comparison plot for the writeup — all runs."""
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            m = re.match(r'step\s+(\d+).*clean_ce\s+([\d.]+).*block_ce\s+([\d.]+).*masked\s+(\d+)', line)
            if m:
                entries.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return entries

def split_run(entries):
    runs, cur = [], []
    for s in entries:
        if cur and s[0] < cur[-1][0]:
            runs.append(cur); cur = [s]
        else:
            cur.append(s)
    if cur: runs.append(cur)
    if len(runs) > 1:
        return runs[-1]  # last run is most recent
    return runs[-1] if runs else []

# ── Runs ────────────────────────────────────────────────────────────────────
# Each: (label, path, color, linestyle, marker)
runs_2L = [
    ('GA dim=4', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim4.log', '#e74c3c', '-', ''),
    ('GA dim=8', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim8.log', '#2ecc71', '-', ''),
    ('GA dim=16', '/home/peb/data/bbt_checkpoints/blockwise_2L_ga_dim16.log', '#27ae60', '-', ''),
    ('Vanilla', '/home/peb/data/bbt_checkpoints/blockwise_2L_vanilla.log', '#3498db', '-', ''),
]

runs_16L = [
    ('GA dim=8 (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.log', '#27ae60', '--', ''),
    ('GA dim=16 (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.log', '#1b9e4e', '-', ''),
    ('GA dim=24 (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim24.log', '#f39c12', '-', ''),
    ('GA dim=32 (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim32.log', '#e67e22', '-', ''),
    ('Vanilla 16L (AMP)', '/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.log', '#3498db', '--', ''),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: 2L clean CE
ax = axes[0, 0]
for label, path, color, ls, marker in runs_2L:
    if not os.path.exists(path): continue
    run = split_run(parse_log(path))
    steps = [s[0] for s in run]
    vals = [s[1] for s in run]  # clean_ce
    ax.plot(steps, vals, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)
ax.set_title('2L/128d — Clean CE')
ax.set_xlabel('Step'); ax.set_ylabel('Clean CE (nats)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: 2L block CE
ax = axes[0, 1]
for label, path, color, ls, marker in runs_2L:
    if not os.path.exists(path): continue
    run = split_run(parse_log(path))
    steps = [s[0] for s in run]
    vals = [s[2] for s in run]  # block_ce
    ax.plot(steps, vals, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)
ax.set_title('2L/128d — Block CE')
ax.set_xlabel('Step'); ax.set_ylabel('Block CE (nats)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: 16L clean CE
ax = axes[1, 0]
for label, path, color, ls, marker in runs_16L:
    if not os.path.exists(path): continue
    run = split_run(parse_log(path))
    steps = [s[0] for s in run]
    vals = [s[1] for s in run]
    ax.plot(steps, vals, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)
ax.set_title('16L/768d — Clean CE')
ax.set_xlabel('Step'); ax.set_ylabel('Clean CE (nats)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 4: 16L block CE
ax = axes[1, 1]
for label, path, color, ls, marker in runs_16L:
    if not os.path.exists(path): continue
    run = split_run(parse_log(path))
    steps = [s[0] for s in run]
    vals = [s[2] for s in run]
    ax.plot(steps, vals, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)
ax.set_title('16L/768d — Block CE')
ax.set_xlabel('Step'); ax.set_ylabel('Block CE (nats)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/peb/code/bbt/figures/loss_comparison.png', dpi=150)
print("Saved loss_comparison.png to figures/")

# ── Also print a summary table ──────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

def summarize(path, label):
    if not os.path.exists(path):
        print(f"{label:25s}: no log")
        return
    run = split_run(parse_log(path))
    clean = [s[1] for s in run]
    block = [s[2] for s in run]
    print(f"{label:25s}: final_clean={clean[-1]:.3f}  best_clean={min(clean):.3f}  "
          f"final_block={block[-1]:.3f}  best_block={min(block):.3f}  "
          f"steps={len(run)}  total_steps={run[-1][0]}")

for label, path, *_ in runs_2L + runs_16L:
    summarize(path, label)
