"""Public Plan n_trans batching: looped single-vector executes vs one batched
execute, min-of-5 warm timing, four regimes (dense OD 2D/3D, sparse FFT-bound,
small GM-path). Batching applies to GM-path type-1 plans only; OD-path plans
loop by design (see api._execute_stack)."""

import sys
import time
import pathlib

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import mlx_nufft as nf  # noqa: E402

REGIMES = [("dense 2D OD", 2, (1024, 1024), 4_000_000, 8),
           ("dense 3D OD", 3, (128, 128, 128), 4_000_000, 8),
           ("sparse FFT-bound", 2, (4096, 4096), 400_000, 8),
           ("GM-path small", 2, (256, 256), 10_000, 8)]


def tmin(fn, n=5):
    fn(); mx.synchronize()
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); fn(); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


if __name__ == "__main__":
    print("mlx_nufft from:", nf.__file__)
    for name, dim, N, M, B in REGIMES:
        rng = np.random.default_rng(0)
        x = [rng.uniform(-np.pi, np.pi, M) for _ in range(dim)]
        cs = (rng.standard_normal((B, M)) + 1j * rng.standard_normal((B, M))).astype(np.complex64)
        pb = nf.Plan(1, N, n_trans=B, eps=1e-5, dtype="complex64"); pb.setpts(*x)
        p1 = nf.Plan(1, N, n_trans=1, eps=1e-5, dtype="complex64"); p1.setpts(*x)
        tb = tmin(lambda: pb.execute(cs))
        tl = tmin(lambda: [p1.execute(cs[b]) for b in range(B)])
        ref = np.stack([p1.execute(cs[b]) for b in range(B)])
        rel = np.linalg.norm(pb.execute(cs) - ref) / np.linalg.norm(ref)
        print(f"{name:17s} {dim}d N={N} M={M:.0e} B={B}: loop {tl:7.1f} ms  "
              f"batched {tb:7.1f} ms  -> {tl/tb:5.2f}x  (rel diff {rel:.1e})")
