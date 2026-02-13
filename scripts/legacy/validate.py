import argparse
import math
from contextlib import nullcontext

import torch

from byte_shard_dataset import ByteShardDataset
from train_bitbyte import BitByteLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/checkpoints/ckpt.pt")
    ap.add_argument(
        "--data",
        type=str,
        default="artifacts/datasets/tinystories/shards/data/shard_*.bin",
        help="Shard glob or single shard path",
    )
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["args"]

    ds = ByteShardDataset(args.data, args.seq_len, seed=0, rank=0, world_size=1)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)

    model = BitByteLM(
        vocab_size=256,
        n_layer=cfg["n_layer"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        d_ff=cfg["d_ff"],
        act_quant=not cfg["no_act_quant"],
        use_sdpa=cfg["use_sdpa"],
        ckpt=not cfg["no_ckpt"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tot_loss = 0.0
    tot_tokens = 0
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx:
        for i, (x, y) in enumerate(dl):
            if i >= args.batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 256),
                y.view(-1),
                reduction="sum",
            )
            tot_loss += float(loss.item())
            tot_tokens += int(y.numel())

    bpb = (tot_loss / max(1, tot_tokens)) / math.log(2.0)
    print(f"val bpb {bpb:.4f} on {tot_tokens} tokens")


if __name__ == "__main__":
    main()
