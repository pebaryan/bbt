# pack_to_shards.py
import os
import argparse
from pathlib import Path

def iter_files(inputs, exts=None):
    exts = set(e.lower() for e in exts) if exts else None
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            yield p
        else:
            for fp in p.rglob("*"):
                if fp.is_file():
                    if exts is None:
                        yield fp
                    else:
                        if fp.suffix.lower().lstrip(".") in exts:
                            yield fp

def pack(inputs, out_dir, shard_size_gb=2.0, buf_mb=8, sep_byte=None, exts=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = int(shard_size_gb * (1024**3))
    buf_size = int(buf_mb * (1024**2))

    shard_idx = 0
    shard_path = out_dir / f"shard_{shard_idx:05d}.bin"
    out = open(shard_path, "wb")
    written = 0

    total_files = 0
    total_bytes = 0

    def rotate():
        nonlocal shard_idx, shard_path, out, written
        out.flush()
        out.close()
        shard_idx += 1
        shard_path = out_dir / f"shard_{shard_idx:05d}.bin"
        out = open(shard_path, "wb")
        written = 0

    for fp in iter_files(inputs, exts=exts):
        try:
            with open(fp, "rb") as f:
                while True:
                    chunk = f.read(buf_size)
                    if not chunk:
                        break
                    # rotate if needed
                    if written + len(chunk) > shard_size and written > 0:
                        rotate()
                    out.write(chunk)
                    written += len(chunk)
                    total_bytes += len(chunk)

            total_files += 1

            # Optional separator to reduce “cross-file continuation” artifacts
            if sep_byte is not None:
                b = bytes([sep_byte])
                if written + 1 > shard_size and written > 0:
                    rotate()
                out.write(b)
                written += 1
                total_bytes += 1

            if total_files % 1000 == 0:
                print(f"Packed {total_files} files, total {total_bytes/1e9:.2f} GB -> {shard_idx+1} shards")

        except Exception as e:
            print(f"Skip {fp}: {e}")

    out.flush()
    out.close()
    print(f"Done. Files: {total_files}, bytes: {total_bytes} (~{total_bytes/1e9:.2f} GB), shards: {shard_idx+1}")
    print(f"Output dir: {out_dir.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Folders or files")
    ap.add_argument("--out_dir", required=True, help="Where to write shards")
    ap.add_argument("--shard_size_gb", type=float, default=2.0)
    ap.add_argument("--buf_mb", type=int, default=8)
    ap.add_argument("--sep_byte", type=int, default=0, help="Insert separator byte between files (0-255). Set -1 to disable.")
    ap.add_argument("--exts", nargs="*", default=None, help="Optional whitelist like txt json code md")
    args = ap.parse_args()

    sep = None if args.sep_byte < 0 else int(args.sep_byte)
    pack(args.inputs, args.out_dir, args.shard_size_gb, args.buf_mb, sep, args.exts)
