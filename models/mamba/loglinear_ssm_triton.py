"""LogLinear SSM — Triton per-level parallel scan.

Drop-in for ``_ssm_scan_multilevel``.  Each Fenwick level runs as an
independent Triton kernel; levels are then weighted-summed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.mamba.loglinear_ssm import LogLinearSSM as _L

try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except ImportError:
    _USE_TRITON = False


if _USE_TRITON:

    @triton.jit
    def _level_scan(
        x_ptr, del_ptr, b_ptr, c_ptr,
        a_ptr, mask_ptr,
        y_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_db, stride_dt, stride_dd,
        stride_bb, stride_bt, stride_bn,
        stride_cb, stride_ct, stride_cn,
        stride_ad, stride_an,
        B: tl.constexpr, T: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
        BLOCK_D: tl.constexpr, lambda_w: tl.constexpr,
    ):
        """One instance = (batch, d_tile).  Single-pass SSM + output."""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_msk = d_off < D

        a_row = tl.load(
            a_ptr + d_off[:, None] * stride_ad + tl.arange(0, N)[None, :] * stride_an,
            mask=d_msk[:, None], other=0.0,
        ).to(tl.float32)

        h = tl.zeros((BLOCK_D, N), dtype=tl.float32)
        n_arr = tl.arange(0, N)

        for t in range(T):
            active = tl.load(mask_ptr + t) > 0

            x_t   = tl.load(x_ptr  + pid_b*stride_xb + t*stride_xt + d_off*stride_xd,
                            mask=d_msk, other=0.0).to(tl.float32)
            del_t = tl.load(del_ptr + pid_b*stride_db + t*stride_dt + d_off*stride_dd,
                            mask=d_msk, other=0.0).to(tl.float32)
            b_t   = tl.load(b_ptr  + pid_b*stride_bb + t*stride_bt + n_arr*stride_bn)
            c_t   = tl.load(c_ptr  + pid_b*stride_cb + t*stride_ct + n_arr*stride_cn)

            a_bar = tl.exp(del_t[:, None] * a_row)
            if active:
                h = a_bar * h + del_t[:, None] * b_t[None, :].to(tl.float32) * x_t[:, None]
            else:
                h = a_bar * h

            y_t = tl.sum(c_t[None, :].to(tl.float32) * h, axis=1) * lambda_w
            tl.store(y_ptr + pid_b*(T*D) + t*D + d_off, y_t, mask=d_msk)


def _build_mask_tensor(L: int, num_levels: int, device) -> torch.Tensor:
    levels_per_t = _L._fenwick_mask(L, num_levels)
    out = torch.zeros(L, num_levels, dtype=torch.int32, device=device)
    for t_idx, active_list in enumerate(levels_per_t):
        for ℓ in active_list:
            if ℓ < num_levels:
                out[t_idx, ℓ] = 1
    return out


def triton_level_scan_forward(
    x_inner, delta, B_mat, C_mat,
    A, D, level_mask, num_levels: int,
    level_logits, device,
):
    """Returns ``y_inner`` [B, T, D].  Matches ``_ssm_scan_multilevel``."""
    if not _USE_TRITON:
        raise RuntimeError("Triton not installed")
    Bsz, T, d_inner = x_inner.shape
    N = A.shape[1]
    out_dtype = x_inner.dtype

    mask_t = _build_mask_tensor(T, num_levels, device)
    lambda_vec = torch.nn.functional.softmax(level_logits, dim=0).float()

    y_acc = torch.zeros(Bsz, T, d_inner, device=device, dtype=torch.float32)
    A_fp32 = -torch.exp(A.float())

    BLOCK_D = 64
    assert d_inner % BLOCK_D == 0
    grid = (Bsz, d_inner // BLOCK_D)

    for ℓ in range(num_levels):
        lam = float(lambda_vec[ℓ])
        if lam < 1e-8:
            continue
        _level_scan[grid](
            x_inner, delta[ℓ], B_mat[ℓ], C_mat[ℓ],
            A_fp32, mask_t[:, ℓ], y_acc,
            stride_xb=1, stride_xt=d_inner, stride_xd=1,
            stride_db=1, stride_dt=d_inner, stride_dd=1,
            stride_bb=1, stride_bt=N,      stride_bn=1,
            stride_cb=1, stride_ct=N,      stride_cn=1,
            stride_ad=N, stride_an=1,
            B=Bsz, T=T, D=d_inner, N=N, BLOCK_D=BLOCK_D,
            lambda_w=lam,
            num_warps=4, num_stages=2,
        )

    # D * x (added once, matching the PyTorch ref)
    # lambda_vec here is NOT used — D is shared across levels, no per-level weight.
    y_acc = y_acc + D.view(1, 1, -1).to(torch.float32) * x_inner
    return y_acc.to(out_dtype)
