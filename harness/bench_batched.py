"""Multi-vector / batched efficiency: realized speedup vs independent.

Two levers, both measured here on the active machine:

  (1) n_trans plan-sharing — one plan, many strength-vectors, vs INDEPENDENT
      transforms each building their own plan. The shared plan amortizes the
      geometry setup (sort, kernel compile, deconvolution, twiddles) across
      vectors; the per-vector execute is bandwidth-bound so batching the
      stages themselves (execute_batch) adds ~nothing beyond plan-sharing.

  (2) disjoint-support — G disjoint subsets of ONE point set, each transformed
      independently (coefficients outside the subset zeroed). execute_disjoint
      spreads each point once total (into its subset's grid) instead of once
      per subset, sharing the spread stage. Win vs independent-shared-plan is
      largest where spreading dominates (high density); FFT/crop don't share.

Usage: bench_batched.py
"""
import sys
import time
import platform
import subprocess
import pathlib

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
import mlx_nufft as af                                   # noqa: E402
import mlx_nufft.gpu_t3 as g                             # noqa: E402
from harness.gen import gen_anisotropic                          # noqa: E402


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, {platform.platform()}"


def tmin(fn, n=3):
    fn()
    mx.synchronize()
    ts = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000


if __name__ == "__main__":
    print(f"machine: {machine()}")
    rng = np.random.default_rng(11)
    G = 4

    print("\n=== (1) n_trans plan-sharing: G separate plans vs 1 shared plan "
          f"+ G executes (G={G}) ===")

    # type-3 3D, largest anisotropic case (heaviest demand)
    prob = gen_anisotropic(N=1024, P=100_000, lat=1.5)
    x3, s3 = prob["x"], prob["s"]
    cs3 = [(rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000)
            ).astype(np.complex64) for _ in range(G)]
    sh3 = g.GpuT3Plan(x3, s3, eps=1e-5, isign=+1, prec="crit64")
    sh3.execute(cs3[0])

    def sep3():
        for c in cs3:
            g.GpuT3Plan(x3, s3, eps=1e-5, isign=+1, prec="crit64").execute(c)
    tA = tmin(sep3, n=2)
    tB = tmin(lambda: [sh3.execute(c) for c in cs3], n=2)
    print(f"  type-3 3D anisotrop: separate {tA:6.0f} ms | shared {tB:6.0f} ms"
          f"  -> {tA/tB:5.2f}x")

    # type-1 3D high density
    M = 8 * 64 ** 3
    x1 = [rng.uniform(-np.pi, np.pi, M) for _ in range(3)]
    cs1 = [(rng.standard_normal(M) + 1j * rng.standard_normal(M)
            ).astype(np.complex64) for _ in range(G)]
    sh1 = af.Type1PlanND(tuple(x1), (64, 64, 64), eps=1e-5, isign=+1)
    sh1.execute(cs1[0])

    def sep1():
        for c in cs1:
            af.Type1PlanND(tuple(x1), (64, 64, 64), eps=1e-5,
                           isign=+1).execute(c)
    tA1 = tmin(sep1, n=2)
    tB1 = tmin(lambda: [sh1.execute(c) for c in cs1], n=2)
    print(f"  type-1 3D high-dens: separate {tA1:6.0f} ms | shared {tB1:6.0f} ms"
          f"  -> {tA1/tB1:5.2f}x")

    print(f"\n=== (2) disjoint-support: independent (shared plan, zeroed) vs "
          f"execute_disjoint (G={G}) ===")
    for label, dim, N, Md in [("type-1 3D high-dens", 3, (64, 64, 64),
                               8 * 64 ** 3),
                              ("type-1 2D", 2, (1024, 1024), 8 * 1024 ** 2),
                              ("type-1 1D", 1, (1 << 20,), 8 * (1 << 20))]:
        xs = [rng.uniform(-np.pi, np.pi, Md) for _ in range(dim)]
        c = (rng.standard_normal(Md) + 1j * rng.standard_normal(Md)
             ).astype(np.complex64)
        groups = rng.integers(0, G, Md)
        p = af.Type1PlanND(tuple(xs), N, eps=1e-5, isign=+1)
        p.execute_disjoint(c, groups, return_np=False)

        def indep():
            for gg in range(G):
                cg = c.copy()
                cg[groups != gg] = 0
                p.execute(cg, return_np=False)
        indep()
        t_ind = tmin(indep)
        t_dis = tmin(lambda: p.execute_disjoint(c, groups, return_np=False))
        print(f"  {label:20}: independent {t_ind:6.1f} ms | disjoint "
              f"{t_dis:6.1f} ms  -> {t_ind/t_dis:5.2f}x")

    print("\nnote: (1) and (2) compound — disjoint over a shared plan also "
          "amortizes the plan, so vs fully-independent (separate-plan) "
          "transforms the realized speedup is the product.")
