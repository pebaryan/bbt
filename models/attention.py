import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE
from quantization.bitlinear import BitLinear


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE, quantization, and GQA support.
    
    Implements multi-head self-attention with:
    - Rotary position embeddings
    - BitLinear projections
    - Optional SDPA (flash attention)
    - Grouped-Query Attention (GQA) for efficient inference
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int | None = None,
        rope_base: float = 10000.0,
        act_quant: bool = True,
        use_sdpa: bool = True,
    ):
        """Initialize causal self-attention.
        
        Args:
            d_model: Model dimension
            n_head: Number of attention heads
            n_kv_head: Number of key/value heads (GQA). If None, defaults to n_head (MHA).
            rope_base: Base for RoPE calculation
            act_quant: Whether to quantize activations
            use_sdpa: Whether to use PyTorch SDPA (flash attention)
        """
        super().__init__()
        assert d_model % n_head == 0
        if n_kv_head is None:
            n_kv_head = n_head
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.head_dim = d_model // n_head
        self.use_sdpa = use_sdpa

        self.q_proj = BitLinear(
            d_model, d_model, bias=False, act_quant=act_quant)
        # K/V projections project to n_kv_head * head_dim (smaller than d_model for GQA)
        self.k_proj = BitLinear(
            d_model, n_kv_head * self.head_dim, bias=False, act_quant=act_quant)
        self.v_proj = BitLinear(
            d_model, n_kv_head * self.head_dim, bias=False, act_quant=act_quant)
        self.o_proj = BitLinear(
            d_model, d_model, bias=False, act_quant=act_quant)

        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal attention.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            Output tensor [B, T, C]
        """
        # x: [B,T,C]
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head,
                                self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_kv_head,
                                self.head_dim).transpose(1, 2)  # [B,KVH,T,D]
        v = self.v_proj(x).view(B, T, self.n_kv_head,
                                self.head_dim).transpose(1, 2)

        pos = torch.arange(T, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        # Repeat K/V heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_head, T, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_head, T, self.head_dim)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch SDPA (may use flash on supported configs)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True)  # [B,H,T,D]
        else:
            # fallback attention
            att = (q @ k.transpose(-2, -1)) / \
                math.sqrt(self.head_dim)  # [B,H,T,T]
            att = att.masked_fill(torch.triu(torch.ones(
                T, T, device=x.device), diagonal=1).bool(), float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)
