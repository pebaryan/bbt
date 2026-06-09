#!/usr/bin/env python3
"""Generate writeup figures for GA vs vanilla blockwise runs."""
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path('/home/peb/code/bbt/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('/home/peb/data/bbt_checkpoints')


def parse_log(path):
    entries = []
    evals = []
    if not os.path.exists(path):
        return entries, evals
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'step\s+(\d+).*clean_ce\s+([\d.]+).*block_ce\s+([\d.]+).*masked\s+(\d+)', line)
            if m:
                step = int(m.group(1))
                entries.append((step, step * 16384, float(m.group(2)), float(m.group(3))))
            m = re.search(r'EVAL step=(\d+) tokens=(\d+)\s+→\s+PPL=([\d.]+)\s+BPB=([\d.]+)', line)
            if m:
                evals.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return entries, evals


def split_run(entries):
    runs, cur = [], []
    for s in entries:
        if cur and s[0] < cur[-1][0]:
            runs.append(cur)
            cur = [s]
        else:
            cur.append(s)
    if cur:
        runs.append(cur)
    return runs[-1] if runs else []


def plot_metric(ax, runs, metric_idx, title, ylabel, max_tokens=None):
    for label, path, color, ls in runs:
        entries, _ = parse_log(path)
        run = split_run(entries)
        if not run:
            continue
        xs = np.array([r[1] / 1e6 for r in run])
        ys = np.array([r[metric_idx] for r in run])
        if max_tokens is not None:
            keep = xs <= max_tokens / 1e6
            xs, ys = xs[keep], ys[keep]
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=1.7, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel('Tokens seen (M)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def make_loss_comparison():
    runs_2l = [
        ('GA dim=4', CKPT_DIR / 'blockwise_2L_ga_dim4.log', '#e74c3c', '-'),
        ('GA dim=8', CKPT_DIR / 'blockwise_2L_ga_dim8.log', '#2ecc71', '-'),
        ('GA dim=16', CKPT_DIR / 'blockwise_2L_ga_dim16.log', '#27ae60', '-'),
        ('Vanilla', CKPT_DIR / 'blockwise_2L_vanilla.log', '#3498db', '-'),
    ]
    runs_16l = [
        ('GA dim=8', CKPT_DIR / 'blockwise_16L_ga.log', '#27ae60', '--'),
        ('GA dim=16', CKPT_DIR / 'blockwise_16L_ga_dim16.log', '#1b9e4e', '-'),
        ('GA dim=24', CKPT_DIR / 'blockwise_16L_ga_dim24.log', '#f39c12', '-'),
        ('GA dim=32', CKPT_DIR / 'blockwise_16L_ga_dim32.log', '#e67e22', '-'),
        ('Vanilla', CKPT_DIR / 'blockwise_16L_vanilla.log', '#3498db', '-'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_metric(axes[0, 0], runs_2l, 2, '2L/128d — Clean CE', 'Clean CE (nats)')
    plot_metric(axes[0, 1], runs_2l, 3, '2L/128d — Block CE', 'Block CE (nats)')
    plot_metric(axes[1, 0], runs_16l, 2, '16L/768d — Clean CE', 'Clean CE (nats)')
    plot_metric(axes[1, 1], runs_16l, 3, '16L/768d — Block CE', 'Block CE (nats)')
    fig.suptitle('Training curves: GA embeddings vs standard byte embeddings', y=1.01)
    plt.tight_layout()
    out = OUT_DIR / 'loss_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def make_16l_controlled():
    runs = [
        ('GA dim=8', CKPT_DIR / 'blockwise_16L_ga.log', '#27ae60', '--'),
        ('GA dim=16', CKPT_DIR / 'blockwise_16L_ga_dim16.log', '#1b9e4e', '-'),
        ('GA dim=24', CKPT_DIR / 'blockwise_16L_ga_dim24.log', '#f39c12', '-'),
        ('GA dim=32', CKPT_DIR / 'blockwise_16L_ga_dim32.log', '#e67e22', '-'),
        ('Vanilla', CKPT_DIR / 'blockwise_16L_vanilla_82M_backup.log', '#3498db', '-'),
    ]
    # fallback if backup log not present: current vanilla log begins with same 82M sweep before resume
    runs = [(l, p if p.exists() else CKPT_DIR / 'blockwise_16L_vanilla.log', c, s) for l, p, c, s in runs]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_metric(axes[0], runs, 2, '16L controlled sweep @82M — Clean CE', 'Clean CE (nats)', max_tokens=82_000_000)
    plot_metric(axes[1], runs, 3, '16L controlled sweep @82M — Block CE', 'Block CE (nats)', max_tokens=82_000_000)
    plt.tight_layout()
    out = OUT_DIR / '16l_82m_controlled_sweep.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def make_1b_scaling():
    runs = [
        ('GA dim=16 @1B', CKPT_DIR / 'ga16l_1B_dim16_resume_20260607_v2.log', '#1b9e4e', '-'),
        ('Vanilla @1B', CKPT_DIR / 'vanilla16l_1B_resume_20260607.log', '#3498db', '-'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_metric(axes[0], runs, 2, '16L 82M→1B continuation — Clean CE', 'Clean CE (nats)')
    plot_metric(axes[1], runs, 3, '16L 82M→1B continuation — Block CE', 'Block CE (nats)')

    for label, path, color, ls in runs:
        _, evals = parse_log(path)
        if not evals:
            continue
        xs = [e[1] / 1e6 for e in evals]
        ys = [e[2] for e in evals]
        axes[2].plot(xs, ys, marker='o', label=label, color=color, linestyle=ls, linewidth=1.7)
    axes[2].set_title('16L 1B continuation — held-out PPL snapshots')
    axes[2].set_xlabel('Tokens seen (M)')
    axes[2].set_ylabel('Held-out PPL')
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / '16l_1b_scaling.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def summarize(path, label):
    run, evals = parse_log(path)
    run = split_run(run)
    if not run:
        return f'{label}: no log'
    best_clean = min(r[2] for r in run)
    best_block = min(r[3] for r in run)
    final = run[-1]
    msg = (
        f'{label}: final_step={final[0]} tokens={final[1]} '
        f'final_clean={final[2]:.3f} best_clean={best_clean:.3f} '
        f'final_block={final[3]:.3f} best_block={best_block:.3f}'
    )
    if evals:
        e = evals[-1]
        msg += f' last_eval_ppl={e[2]:.3f} last_eval_bpb={e[3]:.4f}'
    return msg


if __name__ == '__main__':
    make_loss_comparison()
    make_16l_controlled()
    make_1b_scaling()
    print('\nSUMMARY')
    for label, path in [
        ('GA dim16 1B', CKPT_DIR / 'ga16l_1B_dim16_resume_20260607_v2.log'),
        ('Vanilla 1B', CKPT_DIR / 'vanilla16l_1B_resume_20260607.log'),
        ('GA dim16 82M/full log', CKPT_DIR / 'blockwise_16L_ga_dim16.log'),
        ('Vanilla 82M/full log', CKPT_DIR / 'blockwise_16L_vanilla.log'),
    ]:
        print(summarize(path, label))
