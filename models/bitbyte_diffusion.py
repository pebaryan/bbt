import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .rope import RoPE
from .rmsnorm import RMSNorm
from .mlp import MLP
from quantization.bitlinear import BitLinear


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding (continuous).

    Args:
        t: Float tensor [B] with values in (0, 1].
        dim: Output dimension (should be even).
        max_period: Maximum period for the sinusoidal frequencies.

    Returns:
        [B, dim] sinusoidal embedding.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNModulation(nn.Module):
    """Predicts scale, shift, and gate from noise embedding for AdaLN.

    For each of the two sub-layers (attention, MLP), produces and returns
    three modulation parameters: shift, scale, and gate (residual scaling).

    The gate is zero-initialized so the block starts as identity,
    providing stable training from initialization (DiT-style).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),  # γ₁, β₁, γ₂, β₂, α₁, α₂
        )
        # Zero-init for stable training start (block acts as identity)
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Produce 6 modulation tensors from conditioning vector c.

        Args:
            c: [B, d_model] conditioning vector (timestep embedding).

        Returns:
            Tuple of 6 tensors, each [B, d_model]:
            (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        """
        return self.net(c).chunk(6, dim=-1)


class BidirectionalSelfAttention(nn.Module):
    """Self-attention without causal masking for denoising models."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int | None = None,
        rope_base: float = 10000.0,
        act_quant: bool = True,
        use_sdpa: bool = True,
    ):
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

        self.q_proj = BitLinear(d_model, d_model, bias=False, act_quant=act_quant)
        # K/V projections project to n_kv_head * head_dim (smaller than d_model for GQA)
        self.k_proj = BitLinear(d_model, n_kv_head * self.head_dim, bias=False, act_quant=act_quant)
        self.v_proj = BitLinear(d_model, n_kv_head * self.head_dim, bias=False, act_quant=act_quant)
        self.o_proj = BitLinear(d_model, d_model, bias=False, act_quant=act_quant)
        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)

        pos = torch.arange(seqlen, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        # Repeat K/V heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1).reshape(bsz, self.n_head, seqlen, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1).reshape(bsz, self.n_head, seqlen, self.head_dim)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.o_proj(y)


class DiffusionBlock(nn.Module):
    """Transformer block with AdaLN modulation for diffusion conditioning.

    Uses adaptive layer norm (AdaLN) with gate scaling, following DiT.
    Each block receives a noise-encoding vector c that modulates the
    normalisation and residual scaling.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        n_kv_head: int | None = None,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        ckpt: bool = True,
    ):
        super().__init__()
        self.ckpt = ckpt
        self.n1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = BidirectionalSelfAttention(
            d_model,
            n_head,
            n_kv_head=n_kv_head,
            rope_base=rope_base,
            act_quant=act_quant,
            use_sdpa=use_sdpa,
        )
        self.n2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = MLP(d_model, d_ff, act_quant=act_quant)
        self.adaLN = AdaLNModulation(d_model)

    def _forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Core forward with AdaLN modulation.

        Args:
            x: Token embeddings [B, T, d_model].
            c: Conditioning vector [B, d_model].

        Returns:
            Output [B, T, d_model].
        """
        # Compute modulation in fp32 for numerical stability.
        # AdaLN gates are zero-initialized and produce small values early
        # in training; fp16 rounding causes NaN cascades with BitLinear.
        orig_dtype = x.dtype
        (
            shift_attn, scale_attn, gate_attn,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN(c.float())

        # Pre-modulate: elementwise affine on the normalised input
        # Shape: x [B,T,D], modulation [B,D] → unsqueeze(1) → [B,1,D]
        mod_attn = self.n1(x).float() * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        x = x + gate_attn.unsqueeze(1).to(orig_dtype) * self.attn(mod_attn.to(orig_dtype))

        mod_mlp = self.n2(x).float() * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1).to(orig_dtype) * self.mlp(mod_mlp.to(orig_dtype))

        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.ckpt and self.training:
            return checkpoint(self._forward, x, c, use_reentrant=False)
        return self._forward(x, c)


class BitByteDiffusionLM(nn.Module):
    """Byte denoising model conditioned on continuous diffusion timestep.

    Architecture changes vs original bbt diffusion:
    - Sinusoidal timestep embedding (continuous) instead of learned discrete Embedding
    - Per-block AdaLN modulation (scale, shift, gate) in every DiffusionBlock
      instead of single additive t_emb at input
    - LayerNorm (elementwise_affine=False) + AdaLN instead of RMSNorm
    - Cosine noise schedule with weighted loss in training loop
    """

    def __init__(
        self,
        vocab_size: int = 256,
        num_diffusion_steps: int = 64,
        n_layer: int = 24,
        d_model: int = 1536,
        n_head: int = 12,
        n_kv_head: int | None = None,
        d_ff: int = 4096,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        ckpt: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_diffusion_steps = num_diffusion_steps

        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token at index = vocab_size
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        # Sinusoidal timestep embedding + MLP projection
        # Produces a conditioning vector c for all AdaLN blocks
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    d_model=d_model,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    d_ff=d_ff,
                    act_quant=act_quant,
                    rope_base=rope_base,
                    use_sdpa=use_sdpa,
                    ckpt=ckpt,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean byte logits from corrupted sequence and timestep.

        Args:
            x_t: Corrupted byte tokens [B, T].
            t: Diffusion step [B], expected range [1, num_diffusion_steps].

        Returns:
            Logits over clean bytes [B, T, vocab_size].
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1:
            raise ValueError("t must be 1D [B]")
        if x_t.size(0) != t.size(0):
            raise ValueError("batch size mismatch between x_t and t")

        # Convert discrete timestep to continuous in (0, 1]
        t_cont = t.float() / float(max(1, self.num_diffusion_steps))
        # Build sinusoidal embedding → MLP → conditioning vector (fp32 for stability)
        t_sin = timestep_embedding(t_cont, self.t_embed[0].in_features)
        c = self.t_embed(t_sin)  # [B, d_model]

        x = self.tok_emb(x_t)  # [B, T, d_model]
        for blk in self.blocks:
            x = blk(x, c)
        x = self.norm_f(x)
        return self.lm_head(x)
