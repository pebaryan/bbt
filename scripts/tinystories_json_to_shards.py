# tinystories_json_to_shards.py
import os, json, argparse
from pathlib import Path

CANDIDATE_KEYS = ["text", "story", "completion", "content"]

def extract_text(obj):
    # obj can be dict or list
    if isinstance(obj, dict):
        for k in CANDIDATE_KEYS:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                return obj[k]
        # sometimes nested
        for v in obj.values():
            if isinstance(v, (dict, list)):
                t = extract_text(v)
                if t:
                    return t
        return None
    if isinstance(obj, list):
        for it in obj:
            t = extract_text(it)
            if t:
                return t
    return None

def iter_json_files(root):
    root = Path(root)
    for p in root.rglob("*.json"):
        if p.is_file():
            yield p

def pack(input_dir, out_dir, shard_size_gb=1.0, sep_byte=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = int(shard_size_gb * (1024**3))
    shard_idx = 0
    out_path = out_dir / f"shard_{shard_idx:05d}.bin"
    out = open(out_path, "wb")
    written = 0

    n_files = 0
    n_items = 0
    n_bytes = 0

    def rotate():
        nonlocal shard_idx, out_path, out, written
        out.flush()
        out.close()
        shard_idx += 1
        out_path = out_dir / f"shard_{shard_idx:05d}.bin"
        out = open(out_path, "wb")
        written = 0

    for fp in iter_json_files(input_dir):
        try:
            raw = fp.read_text(encoding="utf-8")
            obj = json.loads(raw)

            # TinyStories HF files are often a list of records
            items = obj if isinstance(obj, list) else [obj]
            n_files += 1

            for rec in items:
                txt = extract_text(rec)
                if not txt:
                    continue
                b = txt.encode("utf-8", errors="ignore")

                # rotate if needed
                if written + len(b) + 1 > shard_size and written > 0:
                    rotate()

                out.write(b)
                out.write(bytes([sep_byte]))
                written += len(b) + 1

                n_items += 1
                n_bytes += len(b) + 1

            if n_files % 200 == 0:
                print(f"files={n_files} stories={n_items} bytes={n_bytes/1e9:.3f}GB shards={shard_idx+1}")

        except Exception as e:
            print(f"skip {fp}: {e}")

    out.flush()
    out.close()
    print(f"Done. files={n_files}, stories={n_items}, bytes={n_bytes} (~{n_bytes/1e9:.3f}GB), shards={shard_idx+1}")
    print(f"Output: {out_dir.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size_gb", type=float, default=1.0)
    ap.add_argument("--sep_byte", type=int, default=0)
    args = ap.parse_args()
    pack(args.input_dir, args.out_dir, args.shard_size_gb, args.sep_byte)
