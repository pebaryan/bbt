"""Deep-equilibrium (DEQ) byte language model — an experimental variant.

Instead of a deep stack of distinct transformer blocks, this model uses:

    prelude (a few standard blocks)  -> context c(x)
    weight-tied core block, iterated to a fixed point y* = f(y*, c)
    coda (optional standard blocks)  -> unembed(LN(y*))

The attractor map is f(y, c) = core(y + c) - y, so the fixed point satisfies
core(y*+c) = 2y* (equivalently y* = c + gated_sublayers(y*+c)). When the core's
per-channel LayerScale gates are initialized near zero the core is ~identity, so
f(.,c) ≈ c is *constant in y* -> Jacobian ≈ 0 -> the forward solve is contractive
from the first step (DiT / attractor-style stable init). As the gates grow the
head does real work.

Forward solve runs under no_grad (FPI or Anderson). Backward uses Jacobian-Free
Backpropagation (Fung et al. 2022): one extra application of f at the detached
fixed point, so memory is O(1) in solver iterations. Full implicit-function-
theorem gradients can be swapped in later if JFB proves insufficient.

`quantize` toggles BitNet ternary BitLinear vs. plain nn.Linear so the DEQ
dynamics can be de-risked in full precision before adding ternary quantization.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.utils.checkpoint import checkpoint as grad_ckpt
from quantization.bitlinear import BitLinear
from .rmsnorm import RMSNorm
from .rope import RoPE
from .deq_solver import get_solver


def make_linear(in_f, out_f, *, quantize, act_quant, bias=False):
    if quantize:
        return BitLinear(in_f, out_f, bias=bias, act_quant=act_quant)
    return nn.Linear(in_f, out_f, bias=bias)


class Attention(nn.Module):
    def __init__(self, d_model, n_head, *, quantize, act_quant, rope_base, use_sdpa,
                 causal=True, n_kv_head=None):
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
        self.causal = causal
        self.q_proj = make_linear(d_model, d_model, quantize=quantize, act_quant=act_quant)
        self.k_proj = make_linear(d_model, n_kv_head * self.head_dim, quantize=quantize, act_quant=act_quant)
        self.v_proj = make_linear(d_model, n_kv_head * self.head_dim, quantize=quantize, act_quant=act_quant)
        self.o_proj = make_linear(d_model, d_model, quantize=quantize, act_quant=act_quant)
        self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        pos = torch.arange(T, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        if self.n_rep > 1:
            k = k[:, :, None].expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_head, T, self.head_dim)
            v = v[:, :, None].expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_head, T, self.head_dim)
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.causal:
                att = att.masked_fill(
                    torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, *, quantize, act_quant):
        super().__init__()
        self.gate_up = make_linear(d_model, 2 * d_ff, quantize=quantize, act_quant=act_quant)
        self.down = make_linear(d_ff, d_model, quantize=quantize, act_quant=act_quant)

    def forward(self, x):
        a, b = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class DEQBlock(nn.Module):
    """Pre-norm transformer block, optionally with per-channel LayerScale gates.

    Gated blocks (the iterated core) start near-identity for solver contraction;
    ungated blocks (prelude/coda) behave like a standard residual block.
    """

    def __init__(self, d_model, n_head, d_ff, *, quantize, act_quant, rope_base,
                 use_sdpa, causal=True, n_kv_head=None, gated=False,
                 layer_scale_init=0.1, gamma_max=1.0, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.n1 = RMSNorm(d_model)
        self.attn = Attention(
            d_model, n_head, quantize=quantize, act_quant=act_quant,
            rope_base=rope_base, use_sdpa=use_sdpa, causal=causal, n_kv_head=n_kv_head)
        self.n2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, quantize=quantize, act_quant=act_quant)
        if gated:
            p = min(max(layer_scale_init / gamma_max, 1e-6), 1.0 - 1e-6)
            raw = math.log(p / (1.0 - p))
            self.raw_gamma_attn = nn.Parameter(torch.full((d_model,), raw))
            self.raw_gamma_mlp = nn.Parameter(torch.full((d_model,), raw))
            self.gamma_max = gamma_max
        else:
            self.register_parameter("raw_gamma_attn", None)
            self.register_parameter("raw_gamma_mlp", None)
            self.gamma_max = 1.0

    def _forward_body(self, x):
        a = self.attn(self.n1(x))
        if self.raw_gamma_attn is not None:
            a = a * (torch.sigmoid(self.raw_gamma_attn) * self.gamma_max)
        x = x + a
        m = self.mlp(self.n2(x))
        if self.raw_gamma_mlp is not None:
            m = m * (torch.sigmoid(self.raw_gamma_mlp) * self.gamma_max)
        return x + m

    def forward(self, x):
        if self.ckpt and self.training:
            return grad_ckpt(self._forward_body, x, use_reentrant=False)
        return self._forward_body(x)


class BitByteDEQ(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        d_model=512,
        n_head=8,
        n_kv_head=None,
        d_ff=1024,
        n_prelude=2,
        n_core=1,
        n_coda=0,
        quantize=False,
        act_quant=True,
        rope_base=10000.0,
        use_sdpa=True,
        solver="anderson",
        max_iter=24,
        tol=None,
        solver_beta=1.0,
        anderson_m=5,
        layer_scale_init=0.1,
        gamma_max=1.0,
        ckpt=True,
    ):
        super().__init__()
        self.solver_name = solver
        self.max_iter = max_iter
        # Auto-select tol based on quantize if not specified explicitly.
        self.tol = tol if tol is not None else (3e-3 if quantize else 1e-3)
        self.solver_beta = solver_beta
        self.anderson_m = anderson_m

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        common = dict(quantize=quantize, act_quant=act_quant, rope_base=rope_base,
                      use_sdpa=use_sdpa, causal=True, n_kv_head=n_kv_head)
        # Prelude/coda support grad-ckpt; core runs under no_grad so ckpt has no effect there.
        self.prelude = nn.ModuleList(
            DEQBlock(d_model, n_head, d_ff, gated=False, ckpt=ckpt, **common)
            for _ in range(n_prelude))
        self.core = nn.ModuleList(
            DEQBlock(d_model, n_head, d_ff, gated=True,
                     layer_scale_init=layer_scale_init, gamma_max=gamma_max, **common)
            for _ in range(n_core))
        self.coda = nn.ModuleList(
            DEQBlock(d_model, n_head, d_ff, gated=False, ckpt=ckpt, **common)
            for _ in range(n_coda))

        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.last_info = {}
        self.last_sigma = float("nan")
        self.last_sigma_max = float("nan")

    def _core(self, h):
        for blk in self.core:
            h = blk(h)
        return h

    def _f(self, y, c):
        # Attractor map; constant in y when the core is ~identity at init.
        return self._core(y + c) - y

    def _solver_kwargs(self):
        if self.solver_name == "anderson":
            return dict(max_iter=self.max_iter, tol=self.tol,
                        beta=self.solver_beta, m=self.anderson_m)
        return dict(max_iter=self.max_iter, tol=self.tol, beta=self.solver_beta)

    @staticmethod
    def _unit(v):
        n = v.flatten(1).norm(dim=1).clamp_min(1e-12)
        return v / n.reshape(-1, *([1] * (v.dim() - 1)))

    def _vjp(self, y_out, y_in, u, create_graph=False):
        (g,) = torch.autograd.grad(
            y_out, y_in, grad_outputs=u,
            create_graph=create_graph, retain_graph=True)
        return g

    def _jvp(self, y_out, y_in, v, create_graph=False):
        # Jacobian-vector product J v via the double-vjp trick.
        w = torch.zeros_like(y_out, requires_grad=True)
        (jt_w,) = torch.autograd.grad(
            y_out, y_in, grad_outputs=w, create_graph=True, retain_graph=True)
        (jv,) = torch.autograd.grad(
            jt_w, w, grad_outputs=v, create_graph=create_graph, retain_graph=True)
        return jv

    def _spectral_sigma(self, y_out, y_in, n_iters):
        """Per-sample estimate of sigma_max(J_f) via power iteration on J^T J."""
        v = self._unit(torch.randn_like(y_in))
        for _ in range(max(0, n_iters - 1)):
            jv = self._jvp(y_out, y_in, v)
            v = self._unit(self._vjp(y_out, y_in, jv)).detach()
        jv = self._jvp(y_out, y_in, v, create_graph=True)
        return jv.flatten(1).norm(dim=1)  # [B], differentiable

    def forward(self, idx, return_info=False, reg=None, reg_margin=0.9, reg_iters=2):
        c = self.tok_emb(idx)
        for blk in self.prelude:
            c = blk(c)

        c_det = c.detach()
        solver = get_solver(self.solver_name)
        with torch.no_grad():
            y_star, info = solver(lambda y: self._f(y, c_det), c_det, **self._solver_kwargs())
        self.last_info = info

        # Jacobian-free (1-step) gradient at the detached fixed point.
        reg_loss = None
        if reg and self.training:
            # The math SDPA backend is required: efficient/flash backward kernels
            # have no double-backward, which the vjp/jvp create_graph needs.
            y_in = y_star.detach().requires_grad_(True)
            with sdpa_kernel([SDPBackend.MATH]):
                y_out = self._f(y_in, c)
                if reg == "jac":
                    # Hutchinson estimate of ||J_f||_F^2 (Bai et al. 2021).
                    eps = torch.randn_like(y_out)
                    vjp = self._vjp(y_out, y_in, eps, create_graph=True)
                    reg_loss = vjp.pow(2).sum() / y_out.shape[0]
                    self.last_sigma = float("nan")
                elif reg == "spec":
                    # Penalize the spectral norm above a contraction margin so
                    # the fixed-point map stays a contraction (sigma_max < 1).
                    sigma = self._spectral_sigma(y_out, y_in, reg_iters)
                    reg_loss = (sigma - reg_margin).clamp_min(0).pow(2).mean()
                    self.last_sigma = float(sigma.mean().item())
                    self.last_sigma_max = float(sigma.max().item())
                else:
                    raise ValueError(f"unknown reg {reg!r}")
            y_star = y_out
        else:
            y_star = self._f(y_star.detach(), c)

        h = y_star
        for blk in self.coda:
            h = blk(h)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        if reg:
            return logits, reg_loss, info
        if return_info:
            return logits, info
        return logits
