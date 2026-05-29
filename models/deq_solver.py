"""Fixed-point solvers for the deep-equilibrium (DEQ) byte LM.

Both solvers find y* such that f(y*) == y* for a batched map f, operating on
tensors shaped [B, ...] (the batch dim is leading; the rest is flattened
internally). They return (y_star, info) where info carries iteration count,
final relative residual, and a converged flag. Intended to run under no_grad in
fp32 — the gradient is attached separately (Jacobian-free) by the model.
"""
import torch


def _rel_residual(fx, x):
    num = (fx - x).reshape(fx.shape[0], -1).norm(dim=1)
    den = fx.reshape(fx.shape[0], -1).norm(dim=1).clamp_min(1e-5)
    return (num / den).max().item()


def fpi_solve(f, x0, *, max_iter=50, tol=1e-3, beta=1.0, min_iter=0):
    """Damped fixed-point iteration: x <- (1-beta) x + beta f(x)."""
    x = x0
    res = float("inf")
    it = 0
    for k in range(max_iter):
        it = k + 1
        fx = f(x)
        new_x = fx if beta == 1.0 else (beta * fx + (1.0 - beta) * x)
        res = _rel_residual(new_x, x)
        x = new_x
        if it >= min_iter and res < tol:
            break
    return x, {"iters": it, "rel_residual": res, "converged": res < tol}


def anderson_solve(f, x0, *, max_iter=50, tol=1e-3, m=5, beta=1.0, lam=1e-4, min_iter=0):
    """Anderson acceleration (Bai et al. DEQ formulation).

    Keeps a window of the last m iterates/residuals and solves a small
    least-squares problem each step to extrapolate the next iterate.
    """
    if max_iter < 3:
        return fpi_solve(f, x0, max_iter=max_iter, tol=tol, beta=beta, min_iter=min_iter)

    bsz = x0.shape[0]
    d = x0[0].numel()
    dev, dtype = x0.device, x0.dtype

    X = torch.zeros(bsz, m, d, device=dev, dtype=dtype)
    Fm = torch.zeros(bsz, m, d, device=dev, dtype=dtype)
    X[:, 0] = x0.reshape(bsz, -1)
    Fm[:, 0] = f(x0).reshape(bsz, -1)
    X[:, 1] = Fm[:, 0]
    Fm[:, 1] = f(Fm[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, device=dev, dtype=dtype)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    yv = torch.zeros(bsz, m + 1, 1, device=dev, dtype=dtype)
    yv[:, 0] = 1.0

    res = float("inf")
    it = 2
    for k in range(2, max_iter):
        it = k + 1
        n = min(k, m)
        G = Fm[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, device=dev, dtype=dtype)[None]
        )
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], yv[:, :n + 1])[:, 1:n + 1, 0]
        idx = k % m
        X[:, idx] = (
            beta * (alpha[:, None] @ Fm[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        Fm[:, idx] = f(X[:, idx].reshape_as(x0)).reshape(bsz, -1)
        res = _rel_residual(Fm[:, idx], X[:, idx])
        if it >= min_iter and res < tol:
            break

    y_star = X[:, (it - 1) % m].reshape_as(x0)
    return y_star, {"iters": it, "rel_residual": res, "converged": res < tol}


_SOLVERS = {"fpi": fpi_solve, "anderson": anderson_solve}


def get_solver(name):
    if name not in _SOLVERS:
        raise ValueError(f"Unknown solver {name!r}; choose from {sorted(_SOLVERS)}")
    return _SOLVERS[name]
