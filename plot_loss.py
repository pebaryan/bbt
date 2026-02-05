import re, matplotlib.pyplot as plt
import pathlib
log_path = pathlib.Path('log.txt')
step=[]; loss=[]; bpb=[]
pat=re.compile(r"step\s+(\d+)\s+seq\s+\d+\s+loss\s+([0-9.]+)\s+bpb\s+([0-9.]+)")
for line in log_path.read_text().splitlines():
    m=pat.search(line)
    if m:
        step.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        bpb.append(float(m.group(3)))

plt.figure(figsize=(8,4))
plt.plot(step, loss, label='loss (nats)')
plt.plot(step, bpb, label='bpb', linestyle='--')
plt.xlabel('step')
plt.ylabel('value')
plt.title('Training loss / bits-per-byte')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out='loss_curve.png'
plt.savefig(out, dpi=150)
print('saved', out, 'points', len(step))