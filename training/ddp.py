import os
import torch
import torch.distributed as dist


def setup_ddp(use_ddp: bool):
    """Setup distributed data parallel training.
    
    Args:
        use_ddp: Whether to use DDP
        
    Returns:
        Tuple of (rank, local_rank, world_size, is_ddp)
    """
    if not use_ddp:
        # Single-process (no torchrun)
        return 0, 0, 1, False

    backend = "nccl" if os.name != "nt" else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size(), True