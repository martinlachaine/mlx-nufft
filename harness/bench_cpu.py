"""CPU FINUFFT baseline timings + accuracy at given problem size.

Usage: bench_cpu.py [lat] [N] [P] [eps] [reps]
Runs fp64 and fp32 CPU FINUFFT type-3, reports wall-clock (averaged) and
rel-L2 vs the fp64 run (and vs direct-sum subset when small enough).
"""

import sys
import time
import pathlib

import numpy as np
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum, rel_l2   # noqa: E402


def run(fn, reps):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, min(ts), float(np.mean(ts))


if __name__ == "__main__":
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    prob = gen_anisotropic(N=N, P=P, lat=lat)
    x, c, s = prob["x"], prob["c"], prob["s"]
    M = s[0].size
    print(f"lat={lat} N={N} (M={M}) P={P} eps={eps:.0e}")

    f64, t64min, t64avg = run(
        lambda: finufft.nufft3d3(*x, c, *s, isign=+1, eps=eps), reps)
    print(f"[cpu fp64] min {t64min:.3f}s avg {t64avg:.3f}s")

    x32 = [v.astype(np.float32) for v in x]
    s32 = [v.astype(np.float32) for v in s]
    c32 = c.astype(np.complex64)
    f32, t32min, t32avg = run(
        lambda: finufft.nufft3d3(*x32, c32, *s32, isign=+1, eps=eps), reps)
    print(f"[cpu fp32] min {t32min:.3f}s avg {t32avg:.3f}s  "
          f"rel_l2(fp32 vs fp64) = {rel_l2(f32, f64):.3e}")

    # subset direct-sum oracle cross-check
    rng = np.random.default_rng(1)
    nsub = min(20_000, M)
    idx = rng.choice(M, nsub, replace=False)
    t0 = time.perf_counter()
    fd = direct_sum(x, c, s, isign=+1, idx=idx)
    td = time.perf_counter() - t0
    print(f"[oracle subset n={nsub}] direct sum {td:.1f}s  "
          f"rel_l2(fp64 finufft vs direct) = {rel_l2(f64[idx], fd):.3e}  "
          f"rel_l2(fp32 finufft vs direct) = {rel_l2(f32[idx], fd):.3e}")
