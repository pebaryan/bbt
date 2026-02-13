import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .rope import RoPE
from .rmsnorm import RMSNorm
from .mlp import MLP
from quantization.bitlinear import BitLinear


class BidirectionalSelfAttention(nn.Module):
    """Self-attention without causal masking for denoising models."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        rope_base: float = 10000.0,
        act_quant: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.use_sdpa = use_sdpa

        self.q_proj = BitLinear(d_model, d_model, bias=False, act_quant=act_quant)
        self.k_proj = BitLinear(d_model, d_model, bias=False, act_quant=act_quant)
        self.v_proj = BitLinear(d_model, d_model, bias=False, act_quant=act_quant)
        self.o_proj = BitLinear(d_model, d_model, bias=False, act_quant=act_quant)
        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        pos = torch.arange(seqlen, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.o_proj(y)


class DiffusionBlock(nn.Module):
    """Transformer block for denoising diffusion."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        ckpt: bool = True,
    ):
        super().__init__()
        self.ckpt = ckpt
        self.n1 = RMSNorm(d_model)
        self.attn = BidirectionalSelfAttention(
            d_model,
            n_head,
            rope_base=rope_base,
            act_quant=act_quant,
            use_sdpa=use_sdpa,
        )
        self.n2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, act_quant=act_quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_fn(inp: torch.Tensor) -> torch.Tensor:
            return inp + self.attn(self.n1(inp))

        def mlp_fn(inp: torch.Tensor) -> torch.Tensor:
            return inp + self.mlp(self.n2(inp))

        if self.ckpt and self.training:
            x = checkpoint(attn_fn, x, use_reentrant=False)
            x = checkpoint(mlp_fn, x, use_reentrant=False)
        else:
            x = attn_fn(x)
            x = mlp_fn(x)
        return x


class BitByteDiffusionLM(nn.Module):
    """Byte denoising model conditioned on diffusion timestep."""

    def __init__(
        self,
        vocab_size: int = 256,
        mask_token_id: int = 256,
        num_diffusion_steps: int = 64,
        n_layer: int = 24,
        d_model: int = 1536,
        n_head: int = 12,
        d_ff: int = 4096,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        ckpt: bool = True,
    ):
        super().__init__()
        if mask_token_id < vocab_size:
            raise ValueError("mask_token_id must be >= vocab_size")

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_diffusion_steps = num_diffusion_steps

        in_vocab_size = mask_token_id + 1
        self.tok_emb = nn.Embedding(in_vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        # t in [1..num_diffusion_steps], keep index 0 available for convenience.
        self.t_emb = nn.Embedding(num_diffusion_steps + 1, d_model)
        nn.init.normal_(self.t_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    d_model=d_model,
                    n_head=n_head,
                    d_ff=d_ff,
                    act_quant=act_quant,
                    rope_base=rope_base,
                    use_sdpa=use_sdpa,
                    ckpt=ckpt,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean byte logits from corrupted sequence and timestep.

        Args:
            x_t: Corrupted byte tokens [B, T]
            t: Diffusion step [B], expected range [1, num_diffusion_steps]

        Returns:
            Logits over clean bytes [B, T, vocab_size]
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1:
            raise ValueError("t must be 1D [B]")
        if x_t.size(0) != t.size(0):
            raise ValueError("batch size mismatch between x_t and t")

        x = self.tok_emb(x_t) + self.t_emb(t)[:, None, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)
