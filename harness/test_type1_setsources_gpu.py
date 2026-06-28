"""Validate Type1PlanND.set_sources(backend="gpu") — the df64 GPU re-point
the type-1 analogue of Type3Plan.set_sources(backend="gpu").

Motivation: workloads that re-point a fixed-mode-box type-1 plan to new
coordinates on every call. The host path is CPU-argsort bound (~hundreds of ms
at P~2e6); the GPU path runs a df64 (double-single) Metal rescale + mx.argsort
sort entirely on-GPU, AND skips the redundant lateral-key sort the OD host path
computes-then-discards.

Hard gates:
  - df64 GPU cell index i1 BIT-EXACT vs the host fp64 path (i1 is a discrete
    cliff — an off-by-one flips a w-tap stencil), including coords spanning many
    multiples of 2*pi (the df64 periodic mod(x,2*pi) reduction is the one
    genuinely new numerical piece vs the validated type-3 df64 setup).
  - df64 GPU fraction fr within 1e-6 of host fp64.
  - mx_perm a valid bijection over 0..P-1.
  - end-to-end execute after a GPU re-point == host re-point == a fresh plan
    (<=1e-5), and ~= CPU finufft fp64 (<= the crit64 gate), for OD + GM and
    dims 1/2/3, including a P change and execute_disjoint.

Speedup is informational (timing is noisy): GPU should be ~tens of x over host.
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

THR = 4.0e-4        # crit64 gate margin at eps=1e-5 (fp32 floor ~1.4e-5)


def _ref(x, c, N, isign):
    fn = {1: finufft.nufft1d1, 2: finufft.nufft2d1, 3: finufft.nufft3d1}[len(x)]
    return fn(*x, c, N, isign=isign, eps=1e-9)


def main():
    rng = np.random.default_rng(11)
    fails = []

    # ---- (1) df64 i1/fr exactness vs host fp64, incl. large |x| ----------
    print("== df64 GPU cells vs host fp64 (i1 exact, fr<=1e-6) ==")
    for dim, N in [(1, (4096,)), (2, (2048, 2048)), (3, (256, 256, 128))]:
        P = 60_000
        # half in-band [-pi,pi), half spanning +-1000*2pi (stress the reduction)
        x = [np.concatenate([rng.uniform(-np.pi, np.pi, P // 2),
                             rng.uniform(-1000 * 2 * np.pi, 1000 * 2 * np.pi,
                                         P - P // 2)]) for _ in range(dim)]
        x = [v.astype(np.float64) for v in x]
        p = Type1PlanND(tuple(x), N, eps=1e-5, isign=+1)
        p._compute_cells(x)
        i1_host = [v.copy() for v in p.i1]
        fr_host = [v.copy() for v in p.fr]
        i1_g, fr_g = p._compute_cells_gpu(x)
        for d in range(dim):
            n_bad = int(np.sum(np.array(i1_g[d]) != i1_host[d]))
            max_dfr = float(np.max(np.abs(np.array(fr_g[d]) - fr_host[d])))
            ok = (n_bad == 0) and (max_dfr <= 1e-6)
            if not ok:
                fails.append(f"df64 cells {dim}d d{d} (i1 bad={n_bad} dfr={max_dfr:.1e})")
            print(f"  {'PASS' if ok else 'FAIL'} {dim}d d{d}: i1 mismatches={n_bad}/{P} "
                  f"max|dfr|={max_dfr:.2e}")

    # ---- (2) end-to-end gpu==host==fresh ~= CPU, OD + GM, dims 1/2/3 ------
    print("== end-to-end execute after set_sources(gpu) ==")
    cases = [
        (1, (4096,), 200_000),       # OD
        (2, (1024, 1024), 200_000),  # OD
        (2, (192, 160), 5_000),      # GM
        (3, (128, 128, 96), 200_000),  # OD
        (3, (64, 64, 48), 5_000),    # GM
    ]
    for dim, N, P in cases:
        x1 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
        x2 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
        c = (rng.standard_normal(P) + 1j * rng.standard_normal(P)).astype(np.complex64)
        for isign in (+1, -1):
            pg = Type1PlanND(tuple(x1), N, eps=1e-5, isign=isign)
            pg.set_sources(x2, backend="gpu")
            f_gpu = pg.execute(c)
            ph = Type1PlanND(tuple(x1), N, eps=1e-5, isign=isign)
            ph.set_sources(x2, backend="host")
            f_host = ph.execute(c)
            f_fresh = Type1PlanND(tuple(x2), N, eps=1e-5, isign=isign).execute(c)
            ref = _ref(x2, c.astype(np.complex128), N, isign)
            perm = np.array(pg.mx_perm)
            bij = np.array_equal(np.sort(perm), np.arange(pg.P))
            d_host = rel_l2(f_gpu, f_host)
            d_fresh = rel_l2(f_gpu, f_fresh)
            d_cpu = rel_l2(f_gpu, ref)
            ok = bij and d_host < 1e-5 and d_fresh < 1e-5 and d_cpu < THR
            if not ok:
                fails.append(f"e2e {dim}d N={N} isign={isign:+d}")
            print(f"  {'PASS' if ok else 'FAIL'} {dim}d N={N} P={P} _od={pg._od} "
                  f"isign={isign:+d}: vs_host={d_host:.1e} vs_fresh={d_fresh:.1e} "
                  f"vs_CPU={d_cpu:.1e} bijection={bij}")

    # ---- (3) P-change re-point on the GPU path ---------------------------
    print("== P-change on gpu re-point ==")
    for dim, N, P in [(2, (512, 512), 200_000), (3, (128, 128, 96), 200_000)]:
        x1 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
        p = Type1PlanND(tuple(x1), N, eps=1e-5, isign=+1)
        P2 = P // 3
        x3 = [rng.uniform(-np.pi, np.pi, P2) for _ in range(dim)]
        c3 = (rng.standard_normal(P2) + 1j * rng.standard_normal(P2)).astype(np.complex64)
        p.set_sources(x3, backend="gpu")
        d = rel_l2(p.execute(c3), _ref(x3, c3.astype(np.complex128), N, +1))
        ok = (p.P == P2) and d < THR
        if not ok:
            fails.append(f"P-change {dim}d")
        print(f"  {'PASS' if ok else 'FAIL'} {dim}d P {P}->{P2}: vs_CPU={d:.1e} "
              f"P={p.P} _od={p._od}")

    # ---- (4) execute_disjoint on a GPU re-point (uses numpy self.perm) ----
    print("== execute_disjoint on gpu re-point ==")
    P, dim, N, G = 200_000, 2, (512, 512), 4
    x1 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
    x2 = [rng.uniform(-np.pi, np.pi, P) for _ in range(dim)]
    c = (rng.standard_normal(P) + 1j * rng.standard_normal(P)).astype(np.complex64)
    groups = rng.integers(0, G, P)
    p = Type1PlanND(tuple(x1), N, eps=1e-5, isign=+1)
    p.set_sources(x2, backend="gpu")
    fd = p.execute_disjoint(c, groups)
    ref = np.stack([_ref(x2, np.where(groups == g, c, 0).astype(np.complex128), N, +1)
                    for g in range(G)])
    d = rel_l2(fd, ref)
    ok = d < THR
    if not ok:
        fails.append("execute_disjoint gpu re-point")
    print(f"  {'PASS' if ok else 'FAIL'} execute_disjoint vs CPU per-group: {d:.2e}")

    # ---- (5) bad backend raises ------------------------------------------
    print("== backend validation ==")
    try:
        Type1PlanND((rng.uniform(-np.pi, np.pi, 100),), (64,)).set_sources(
            (rng.uniform(-np.pi, np.pi, 100),), backend="cuda")
        fails.append("bad backend did not raise")
        print("  FAIL bad backend did not raise")
    except ValueError:
        print("  PASS bad backend raises ValueError")

    # ---- (6) speedup (informational) -------------------------------------
    print("== set_sources gpu-vs-host speedup (informational, P=2e6 OD) ==")

    def tmin(fn, n=5):
        fn(); mx.synchronize()
        ts = []
        for _ in range(n):
            mx.synchronize(); t0 = time.perf_counter(); fn(); mx.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1000

    for N in [(2048, 2048), (4096, 4096)]:
        Pb = 2_000_000
        x1 = [rng.uniform(-np.pi, np.pi, Pb) for _ in range(2)]
        p = Type1PlanND(tuple(x1), N, eps=1e-5, isign=+1)
        xs = [[rng.uniform(-np.pi, np.pi, Pb) for _ in range(2)] for _ in range(6)]
        it = [0]

        def host():
            i = it[0] % 6; it[0] += 1; p.set_sources(xs[i], backend="host")

        def gpu():
            i = it[0] % 6; it[0] += 1; p.set_sources(xs[i], backend="gpu")

        th, tg = tmin(host), tmin(gpu)
        print(f"  N={N} _od={p._od}: host {th:.0f}ms  gpu {tg:.1f}ms  -> {th / tg:.0f}x")

    print()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
