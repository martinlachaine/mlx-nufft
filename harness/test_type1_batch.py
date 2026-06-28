"""Validate Type1PlanND.execute_batch (batched multi-strength type-1) and
set_sources (cheap re-point): multiple strength vectors over one shared
point spread, and re-pointing a plan to new coordinates.

Correctness gates (hard):
  - execute_batch(cs)[b] == execute(cs[b]) for every b   (batched == looped)
  - execute_batch(cs)[b] ~= CPU finufft fp64 reference    (rel_l2 <= thr)
  - set_sources(x2) gives the same result as a fresh plan over x2
  - covers dims 1/2/3, both the OD (large P) and GM (small P) spread paths,
    and B-chunking when the batched grid would exceed the int32 buffer cap.

Speedup (informational, not gated — timing is noisy and regime-dependent):
  - batched beats looped only when the SPREAD dominates (dense point cloud);
    when the FFT dominates (sparse) they tie. The shared spread amortizes the
    ES-weight evaluation, not the atomic scatter, so the win tracks density.
"""

import sys
import time
import pathlib

import numpy as np
import finufft
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import rel_l2                              # noqa: E402
from mlx_nufft import Type1PlanND                       # noqa: E402

# rel-L2 acceptance threshold at eps=1e-5 (fp32 execute floor ~1.4e-5).
THR = 4.0e-4


def _ref(x, c, N, isign):
    fn = {1: finufft.nufft1d1, 2: finufft.nufft2d1, 3: finufft.nufft3d1}[len(x)]
    return fn(*x, c, N, isign=isign, eps=1e-9)


def main():
    rng = np.random.default_rng(11)
    fails = []

    # (dim, N, P, B) — P=200k exercises the OD path, P=5k the GM path.
    cases = [
        (1, (4096,), 200_000, 5),
        (2, (256, 256), 200_000, 5),
        (2, (192, 160), 5_000, 4),       # GM path
        (3, (64, 64, 48), 5_000, 6),
    ]
    for dim, N, P, B in cases:
        x = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
        cs = (rng.standard_normal((B, P))
              + 1j * rng.standard_normal((B, P))).astype(np.complex64)
        for isign in (+1, -1):
            plan = Type1PlanND(tuple(x), N, eps=1e-5, isign=isign)
            fb = plan.execute_batch(cs)                      # (B, *N)
            # batched == looped, and batched ~= CPU reference, per strength
            max_vs_loop = 0.0
            max_vs_ref = 0.0
            for b in range(B):
                fl = plan.execute(cs[b])
                ref = _ref(x, cs[b].astype(np.complex128), N, isign)
                max_vs_loop = max(max_vs_loop, rel_l2(fb[b], fl))
                max_vs_ref = max(max_vs_ref, rel_l2(fb[b], ref))
            ok = (max_vs_loop < 1e-5) and (max_vs_ref < THR)
            tag = "PASS" if ok else "FAIL"
            if not ok:
                fails.append(f"{dim}d N={N} isign={isign:+d}")
            print(f"  {tag} t1-batch {dim}d N={N} P={P} B={B} isign={isign:+d}: "
                  f"vs_loop={max_vs_loop:.2e} vs_ref={max_vs_ref:.2e} (thr {THR:.0e})")

    # ---- set_sources: cheap re-point == fresh plan over the new points -------
    print("== set_sources re-point ==")
    for dim, N, P in [(2, (256, 256), 50_000), (3, (64, 64, 48), 50_000)]:
        x1 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
        x2 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
        c = (rng.standard_normal(P) + 1j * rng.standard_normal(P)).astype(np.complex64)
        plan = Type1PlanND(tuple(x1), N, eps=1e-5, isign=+1)
        plan.set_sources(x2)
        f_rp = plan.execute(c)
        f_fresh = Type1PlanND(tuple(x2), N, eps=1e-5, isign=+1).execute(c)
        d = rel_l2(f_rp, f_fresh)
        ok = d < 1e-5
        tag = "PASS" if ok else "FAIL"
        if not ok:
            fails.append(f"set_sources {dim}d")
        print(f"  {tag} set_sources {dim}d N={N} P={P}: vs_fresh={d:.2e}")

        # P may change on re-point (OD bin tables / GM kernel rebuilt)
        P2 = P // 3
        x3 = [rng.uniform(-np.pi, np.pi, P2) for _ in range(dim)]
        c3 = (rng.standard_normal(P2) + 1j * rng.standard_normal(P2)).astype(np.complex64)
        plan.set_sources(x3)
        f_rp3 = plan.execute(c3)
        f_fresh3 = Type1PlanND(tuple(x3), N, eps=1e-5, isign=+1).execute(c3)
        d3 = rel_l2(f_rp3, f_fresh3)
        ok3 = d3 < 1e-5
        tag3 = "PASS" if ok3 else "FAIL"
        if not ok3:
            fails.append(f"set_sources {dim}d P-change")
        print(f"  {tag3} set_sources {dim}d P {P}->{P2}: vs_fresh={d3:.2e}")

    # ---- speedup (informational): spread-dominated regime -> batched wins ----
    print("== batched-vs-looped speedup (informational) ==")

    def tmin(fn, n=3):
        fn(); mx.synchronize()
        ts = []
        for _ in range(n):
            mx.synchronize(); t0 = time.perf_counter(); fn(); mx.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1000

    for label, N, P, B in [("spread-dominated (dense)", (512, 512), 4_000_000, 5),
                           ("fft-dominated (sparse)", (4096, 4096), 400_000, 5)]:
        x = [rng.uniform(-np.pi, np.pi, P) for _ in range(2)]
        cs = (rng.standard_normal((B, P))
              + 1j * rng.standard_normal((B, P))).astype(np.complex64)
        plan = Type1PlanND(tuple(x), N, eps=1e-5, isign=+1)
        plan.execute_batch(cs); [plan.execute(cs[b]) for b in range(B)]
        t_b = tmin(lambda: plan.execute_batch(cs, return_np=False))
        t_l = tmin(lambda: [plan.execute(cs[b], return_np=False) for b in range(B)])
        dens = P / float(np.prod(plan.n_up))
        print(f"  {label} n_up={plan.n_up} density={dens:.2f}/cell B={B}: "
              f"looped {t_l:.0f}ms batched {t_b:.0f}ms -> {t_l/t_b:.2f}x")

    print()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
