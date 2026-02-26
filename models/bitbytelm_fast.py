import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm
from .rope import RoPE


class FastLinear(nn.Module):
    """Inference-only linear with frozen pre-quantized weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if w.dtype != x.dtype:
            w = w.to(dtype=x.dtype)
        if b is not None and b.dtype != x.dtype:
            b = b.to(dtype=x.dtype)
        return F.linear(x, w, b)


class FastCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.use_sdpa = use_sdpa

        self.q_proj = FastLinear(d_model, d_model, bias=False)
        self.k_proj = FastLinear(d_model, d_model, bias=False)
        self.v_proj = FastLinear(d_model, d_model, bias=False)
        self.o_proj = FastLinear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        start_pos: int = 0,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, t, d_model = x.shape
        q = self.q_proj(x).view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)

        pos = torch.arange(start_pos, start_pos + t, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        if max_cache_len is not None:
            if past_k is None or past_v is None:
                k_cache = torch.empty(
                    bsz, self.n_head, max_cache_len, self.head_dim, device=k.device, dtype=k.dtype
                )
                v_cache = torch.empty(
                    bsz, self.n_head, max_cache_len, self.head_dim, device=v.device, dtype=v.dtype
                )
            else:
                k_cache = past_k
                v_cache = past_v

            k_cache[:, :, start_pos : start_pos + t, :] = k
            v_cache[:, :, start_pos : start_pos + t, :] = v
            k = k_cache[:, :, : start_pos + t, :]
            v = v_cache[:, :, : start_pos + t, :]
            next_k = k_cache
            next_v = v_cache
        else:
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            next_k = k
            next_v = v

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            # For decode (t=1 with cache), no causal mask is needed.
            use_causal = start_pos == 0
            y = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)
        else:
            kt = k.transpose(-2, -1)
            att = (q @ kt) / math.sqrt(self.head_dim)
            if start_pos == 0:
                att = att.masked_fill(
                    torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool(),
                    float("-inf"),
                )
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, t, d_model)
        return self.o_proj(y), next_k, next_v


class FastSwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class FastMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = FastLinear(d_model, 2 * d_ff, bias=False)
        self.down = FastLinear(d_ff, d_model, bias=False)
        self.act = FastSwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate_up(x)))


class FastBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = FastCausalSelfAttention(
            d_model=d_model,
            n_head=n_head,
            rope_base=rope_base,
            use_sdpa=use_sdpa,
        )
        self.n2 = RMSNorm(d_model)
        self.mlp = FastMLP(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        start_pos: int = 0,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, k, v = self.attn(
            self.n1(x),
            past_k=past_k,
            past_v=past_v,
            start_pos=start_pos,
            max_cache_len=max_cache_len,
        )
        x = x + attn_out
        x = x + self.mlp(self.n2(x))
        return x, k, v


class FastBitByteLM(nn.Module):
    """Inference model that expects already ternarized linear weights."""

    def __init__(
        self,
        vocab_size: int = 256,
        n_layer: int = 24,
        d_model: int = 1536,
        n_head: int = 12,
        d_ff: int = 4096,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        self.blocks = nn.ModuleList(
            [
                FastBlock(
                    d_model=d_model,
                    n_head=n_head,
                    d_ff=d_ff,
                    rope_base=rope_base,
                    use_sdpa=use_sdpa,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x, _, _ = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        start_pos: int = 0,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.tok_emb(idx)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, blk in enumerate(self.blocks):
            past_k = None
            past_v = None
            if kv_cache is not None and kv_cache[i] is not None:
                past_k, past_v = kv_cache[i]

            x, k, v = blk(
                x,
                past_k=past_k,
                past_v=past_v,
                start_pos=start_pos,
                max_cache_len=max_cache_len,
            )
            new_cache.append((k, v))

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, new_cache
