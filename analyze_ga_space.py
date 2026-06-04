#!/usr/bin/env python3
"""Analyze the GA decoder embedding space."""
import sys, os
sys.path.insert(0, '/home/peb/code/bbt')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load GA 16L checkpoint and extract decoder weights
ckpt = '/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt'
raw = torch.load(ckpt, map_location='cpu', weights_only=False)

# The GADecoder weight is a key: 'ga_decode.weight' in the GA_Blockwise state_dict
# GA_Blockwise stores it as 'ga_decode.weight'
decoder_weight = raw['model']['ga_decode.weight']  # [257, 8]
print(f"Decoder weight shape: {decoder_weight.shape}")  # [257, 8]

# Normalize for cosine similarity
w_norm = decoder_weight / (decoder_weight.norm(dim=-1, keepdim=True) + 1e-8)

# Cosine similarity matrix
cos_sim = w_norm @ w_norm.T  # [257, 257]

# Print byte groups to analyze structure
byte_categories = {
    'uppercase': list(range(65, 91)),
    'lowercase': list(range(97, 123)),
    'digits': list(range(48, 58)),
    'punctuation': [33, 34, 39, 44, 46, 58, 59, 63, 33, 45],
    'whitespace': [9, 10, 13, 32],
    'control': list(range(0, 32)) + [127],
    'extended': list(range(128, 256)),
}

# Average intra-category and inter-category similarities
def avg_sim(indices_a, indices_b):
    sub = cos_sim[np.array(indices_a)[:, None], np.array(indices_b)[None, :]]
    return sub.mean().item()

print("\nAverage cosine similarity within/across byte categories:")
print("-" * 60)
cats = ['lowercase', 'uppercase', 'digits', 'punctuation', 'whitespace']
print(f"{'Category':>12} | " + " | ".join(f"{c:>10}" for c in cats))
print("-" * 60)
for c1 in cats:
    row = f"{c1:>12} | "
    for c2 in cats:
        sim = avg_sim(byte_categories[c1], byte_categories[c2])
        row += f"{sim:>10.4f}"
    print(row)

# Also check: do the same character in upper/lower case have high similarity?
print("\nUpper-lowercase similarity for each letter:")
ul_sims = []
for i in range(26):
    upper_idx = 65 + i  # 'A'
    lower_idx = 97 + i  # 'a'
    sim = cos_sim[upper_idx, lower_idx].item()
    ul_sims.append(sim)
    if i < 5:  # just show first 5
        print(f"  {chr(upper_idx)}-{chr(lower_idx)}: {sim:.4f}")
print(f"  Average upper-lower similarity: {np.mean(ul_sims):.4f}")
print(f"  Average upper-upper similarity: {np.mean([cos_sim[65+i, 65+j].item() for i in range(26) for j in range(26) if i!=j]):.4f}")
print(f"  Average lower-lower similarity: {np.mean([cos_sim[97+i, 97+j].item() for i in range(26) for j in range(26) if i!=j]):.4f}")

# Visualize the full similarity matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Full matrix
ax = axes[0]
im = ax.imshow(cos_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_title('GA Decoder: Byte Cosine Similarity Matrix (257×257)')
ax.set_xlabel('Byte ID')
ax.set_ylabel('Byte ID')
plt.colorbar(im, ax=ax, shrink=0.8)

# Annotate byte regions
for label, start, end, color in [
    ('Control', 0, 32, 'gray'),
    ('Printable', 32, 127, 'lightgray'),
    ('Extended', 127, 257, 'darkgray'),
]:
    ax.axvline(start, color=color, linestyle=':', linewidth=0.5)
    ax.axhline(start, color=color, linestyle=':', linewidth=0.5)

# Just ASCII region (32-127)
ax = axes[1]
ascii_sim = cos_sim[32:127, 32:127]
im = ax.imshow(ascii_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_title('ASCII Printable Region (bytes 32-126)')
ax.set_xlabel('Byte ID')
ax.set_ylabel('Byte ID')
plt.colorbar(im, ax=ax, shrink=0.8)

# Highlight regions in ASCII
for label, start, end in [('Punct', 33, 48), ('Digits', 48, 58), ('Uppercase', 65, 91), ('Lowercase', 97, 123)]:
    rel_start = start - 32
    rel_end = end - 32
    ax.axvline(rel_start, color='blue', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(rel_start, color='blue', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('/home/peb/data/bbt_checkpoints/ga_decoder_analysis.png', dpi=150)
print(f"\nSaved /home/peb/data/bbt_checkpoints/ga_decoder_analysis.png")
