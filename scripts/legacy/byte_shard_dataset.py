# byte_shard_dataset.py
import os
import glob
import random
import torch

# byte_shard_dataset.py
import mmap
from dataclasses import dataclass

@dataclass
class Shard:
    path: str
    f: object          # file handle
    mm: mmap.mmap
    size: int

def open_mmap(path: str) -> Shard:
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    size = mm.size()
    return Shard(path=path, f=f, mm=mm, size=size)


class ByteShardDataset(torch.utils.data.IterableDataset):
    def __init__(self, shard_glob: str, seq_len: int, seed: int, rank: int, world_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        paths = sorted(glob.glob(shard_glob))
        if not paths:
            raise FileNotFoundError(f"No shards match: {shard_glob}")

        self.shards = [open_mmap(p) for p in paths]
        self.total_bytes = sum(s.size for s in self.shards)

        # Precompute cumulative sizes for fast weighted sampling by bytes
        self.cum = []
        c = 0
        for s in self.shards:
            c += s.size
            self.cum.append(c)

    def set_seq_len(self, seq_len: int):
        self.seq_len = seq_len

    def _pick_shard(self, r: int):
        # r in [0, total_bytes)
        # binary search cum
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r < self.cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard = self.shards[lo]
        base = 0 if lo == 0 else self.cum[lo - 1]
        return shard, r - base

    def __iter__(self):
        rng = random.Random(self.seed + self.rank * 100003)

        # Simple striding to reduce overlap across ranks
        # (not perfect, but good enough; real diversity comes from random offsets)
        step = self.world_size

        i = self.rank
        while True:
            # sample a byte position in the concatenated shard space
            r = rng.randrange(0, self.total_bytes)
            shard, off_in_shard = self._pick_shard(r)

            # shift by rank stride to decorrelate
            off_in_shard = (off_in_shard + i) % shard.size

            need = self.seq_len + 1
            if shard.size <= need + 1:
                i += step
                continue

            # ensure we have room
            if off_in_shard + need >= shard.size:
                off_in_shard = shard.size - need - 1

            chunk = shard.mm[off_in_shard : off_in_shard + need]  # bytes
            # Convert to tensors
            x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)

            yield x, y
            i += step
