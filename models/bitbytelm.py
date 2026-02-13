import math
import torch
import torch.nn as nn

from .block import Block
from .rmsnorm import RMSNorm


class BitByteLM(nn.Module):
    """BitByte language model.
    
    Implements a transformer-based language model with:
    - BitLinear layers for quantized weights
    - Ternary quantization with STE
    - RoPE embeddings
    - RMSNorm
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        n_layer: int = 24,
        d_model: int = 1536,
        n_head: int = 12,
        d_ff: int = 4096,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        ckpt: bool = True,
    ):
        """Initialize BitByteLM.
        
        Args:
            vocab_size: Size of vocabulary (default 256 for bytes)
            n_layer: Number of transformer blocks
            d_model: Model dimension
            n_head: Number of attention heads
            d_ff: Feed-forward dimension
            act_quant: Whether to quantize activations
            rope_base: Base for RoPE calculation
            use_sdpa: Whether to use SDPA
            ckpt: Whether to use gradient checkpointing
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Lower init scale keeps logits in a sane range for fp16.
        nn.init.normal_(self.tok_emb.weight, mean=0.0,
                        std=1.0 / math.sqrt(d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff, act_quant=act_quant,
                  rope_base=rope_base, use_sdpa=use_sdpa, ckpt=ckpt)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            idx: Input token indices [B, T]
            
        Returns:
            Logits [B, T, vocab_size]
        """
        # idx: [B,T] bytes
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B,T,256]
        return logits
