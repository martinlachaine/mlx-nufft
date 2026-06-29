"""mlx-nufft quickstart — runnable end-to-end, no dev dependencies.

Run it after installing the package:

    python examples/quickstart.py

It does three things:
  1. a 2-D type-1 transform via the drop-in finufft API,
  2. a plan-reuse example (plan once, execute many),
  3. a self-check of accuracy against an exact direct DFT on a small problem,
     so you can confirm the library is correct on *your* machine without
     needing CPU finufft installed.

Only requires `mlx_nufft` and `numpy`.
"""

import numpy as np
import mlx_nufft as finufft


def demo_basic():
    """2-D type-1: M nonuniform points -> (N1 x N2) uniform Fourier modes."""
    rng = np.random.default_rng(0)
    M, N1, N2 = 100_000, 256, 256
    x = rng.uniform(-np.pi, np.pi, M)              # coords in [-pi, pi)
    y = rng.uniform(-np.pi, np.pi, M)
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)   # strengths

    fk = finufft.nufft2d1(x, y, c, (N1, N2), eps=1e-6)
    print(f"[basic]   nufft2d1 -> shape {fk.shape}, dtype {fk.dtype}")


def demo_plan_reuse():
    """Plan once (caches geometry-dependent setup), execute per call.

    This is the fast path for fixed-geometry workloads: you pay kernel/grid
    setup once, then each execute() is just the transform.
    """
    rng = np.random.default_rng(1)
    M, N1, N2 = 200_000, 512, 512
    x = rng.uniform(-np.pi, np.pi, M)
    y = rng.uniform(-np.pi, np.pi, M)

    plan = finufft.Plan(1, (N1, N2), eps=1e-6)
    plan.setpts(x, y)
    for k in range(3):
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        fk = plan.execute(c)
    print(f"[reuse]   3x execute on one plan -> shape {fk.shape}")


def demo_accuracy_selfcheck():
    """Verify against an exact direct DFT on a small problem.

    type-1, 1-D:  f[k] = sum_j c[j] * exp(i * k * x_j),  k = -N/2 .. N/2-1
    The direct sum is O(M*N) but exact, so it is the ground truth.
    """
    rng = np.random.default_rng(2)
    M, N = 2_000, 64
    x = rng.uniform(-np.pi, np.pi, M)
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    eps = 1e-6

    fk = np.asarray(finufft.nufft1d1(x, c, N, eps=eps, isign=+1))

    k = np.arange(-(N // 2), N - N // 2)              # finufft "CMCL" mode order
    f_exact = (c[None, :] * np.exp(1j * k[:, None] * x[None, :])).sum(axis=1)

    rel_l2 = np.linalg.norm(fk - f_exact) / np.linalg.norm(f_exact)
    ok = rel_l2 < 10 * eps
    print(f"[verify]  rel-L2 vs exact DFT = {rel_l2:.2e} "
          f"(eps={eps:g})  ->  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"mlx-nufft {finufft.__version__}")
    demo_basic()
    demo_plan_reuse()
    ok = demo_accuracy_selfcheck()
    raise SystemExit(0 if ok else 1)
