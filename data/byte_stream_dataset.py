import torch
import torch.utils.data


class ByteStreamDataset(torch.utils.data.IterableDataset):
    """Byte stream dataset for training on raw binary files.
    
    Streams from a single binary file with sharding support
    for distributed training.
    """
    
    def __init__(
        self,
        path: str,
        seq_len: int,
        seed: int = 1234,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize byte stream dataset.
        
        Args:
            path: Path to binary file
            seq_len: Sequence length
            seed: Random seed
            rank: Process rank for sharding
            world_size: Total number of processes
        """
        super().__init__()
        self.path = path
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def set_seq_len(self, seq_len: int):
        """Update sequence length dynamically."""
        self.seq_len = seq_len

    def __iter__(self):
        """Infinite stream of (x, y) pairs."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank)

        with open(self.path, "rb") as f:
            data = f.read()

        n = len(data)
        # shard by rank: each rank samples offsets in a disjoint strided pattern
        stride = self.world_size

        # infinite stream
        i = self.rank
        while True:
            # pseudo-random-ish offset
            off = int(torch.randint(
                0, n - (self.seq_len + 1), (1,), generator=g).item())
            off = (off + i) % (n - (self.seq_len + 1))
            chunk = data[off: off + self.seq_len + 1]
            x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)
            yield x, y
            i += stride