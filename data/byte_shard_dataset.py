import os
import glob
import torch
import torch.utils.data


class ByteShardDataset(torch.utils.data.IterableDataset):
    """Byte shard dataset for training on multiple binary files.
    
    Streams from a glob pattern of shard files with sharding support
    for distributed training.
    """
    
    def __init__(
        self,
        shard_glob: str,
        seq_len: int,
        seed: int = 1234,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize byte shard dataset.
        
        Args:
            shard_glob: Glob pattern for shard files (e.g., "data/shard_*.bin")
            seq_len: Sequence length
            seed: Random seed
            rank: Process rank for sharding
            world_size: Total number of processes
        """
        super().__init__()
        self.shard_glob = shard_glob
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        
        # Find all shards
        self.shards = sorted(glob.glob(shard_glob))
        if not self.shards:
            raise ValueError(f"No shard files found matching: {shard_glob}")
        self.total_shards = len(self.shards)
        
        # Precompute total data size
        self.shard_sizes = []
        for shard_path in self.shards:
            self.shard_sizes.append(os.path.getsize(shard_path))
        self.total_size = sum(self.shard_sizes)
        
    def set_seq_len(self, seq_len: int):
        """Update sequence length dynamically."""
        self.seq_len = seq_len

    def __iter__(self):
        """Infinite stream of (x, y) pairs from shuffled shards."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank)

        # Create deterministic shard order per rank
        shard_order = list(range(self.total_shards))
        
        # infinite stream
        i = self.rank
        shard_idx = 0
        
        while True:
            # Select shard (round-robin across ranks)
            current_shard = self.shards[shard_idx % self.total_shards]
            
            with open(current_shard, "rb") as f:
                data = f.read()
            
            n = len(data)
            
            # pseudo-random-ish offset
            off = int(torch.randint(
                0, n - (self.seq_len + 1), (1,), generator=g).item())
            off = (off + i) % (n - (self.seq_len + 1))
            
            chunk = data[off: off + self.seq_len + 1]
            x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)
            
            yield x, y
            
            i += self.world_size
            shard_idx += 1