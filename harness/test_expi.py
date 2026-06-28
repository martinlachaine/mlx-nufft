"""Oracle-gate mlx_nufft.expi — the general df64 phase -> e^{i*phi}
primitive (df64 reduction mod 2pi + f32 cos/sin on the GPU).

The problem it solves: an fp64 phase of order 1e8 radians cannot be reduced mod
2pi in f32 (f32 loses all fractional bits by |phi| ~ 1e3), but the cos/sin
should run on the GPU. expi forms/sums the phase in df64, reduces mod 2pi in
df64 (k*2pi exact via two_prod), then does f32 cos/sin of the O(1) residual.

Reference is a straight fp64 host computation (np.exp), which evaluates cos/sin
of the exact fp64-represented phase to ~1e-16 — a valid gold oracle here.

Gate: for |phi| up to ~1e8 rad, max abs error on e^{i*phi} <= 2e-6 (typ ~1e-6
at 1e8, ~2e-7 below); beyond the f32-exact-quotient ceiling the error grows
gracefully (checked, informational).
"""

import sys
import time
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from mlx_nufft import expi, EXPI_MAX_PHASE          # noqa: E402

GATE_1E8 = 3e-6        # band edge (typ ~1e-6; worst-case ~1.9e-6 near ceiling)
GATE_SMALL = 5e-7      # |phi| <= 1e7


def main():
    rng = np.random.default_rng(7)
    fails = []

    def maxerr(phi, isign=1):
        e = np.asarray(expi(phi, isign=isign, return_np=True))
        ref = np.exp(1j * isign * phi.astype(np.float64))
        return float(np.max(np.abs(e - ref)))

    print(f"EXPI_MAX_PHASE = {EXPI_MAX_PHASE:.4e} rad")

    # ---- magnitude sweep (single array) ----------------------------------
    print("== magnitude sweep (single fp64 array, P=200k) ==")
    P = 200_000
    for mag, gate in [(1e0, GATE_SMALL), (1e3, GATE_SMALL), (1e5, GATE_SMALL),
                      (1e6, GATE_SMALL), (1e7, GATE_SMALL),
                      (5e7, GATE_1E8), (1e8, GATE_1E8)]:
        phi = rng.uniform(-mag, mag, P)
        e = maxerr(phi)
        ok = e <= gate
        if not ok:
            fails.append(f"|phi|<={mag:.0e} err={e:.2e}>{gate:.0e}")
        print(f"  {'PASS' if ok else 'FAIL'} |phi|<={mag:.0e}: "
              f"max|Δ|={e:.2e} (gate {gate:.0e})")

    # ---- both signs ------------------------------------------------------
    print("== isign ==")
    phi = rng.uniform(-1e8, 1e8, P)
    for isign in (+1, -1):
        e = maxerr(phi, isign=isign)
        ok = e <= GATE_1E8
        if not ok:
            fails.append(f"isign={isign} err={e:.2e}")
        print(f"  {'PASS' if ok else 'FAIL'} isign={isign:+d}: max|Δ|={e:.2e}")

    # ---- summable components (phi = a+b+c, df64 sum) ---------------------
    print("== summable fp64 components (phi = a+b+c, |phi|~1e8) ==")
    comps = [rng.uniform(-3.3e7, 3.3e7, P) for _ in range(3)]
    e = np.asarray(expi(comps, return_np=True))
    ref = np.exp(1j * sum(c.astype(np.float64) for c in comps))
    err_c = float(np.max(np.abs(e - ref)))
    ok = err_c <= GATE_1E8
    if not ok:
        fails.append(f"multi-component err={err_c:.2e}")
    print(f"  {'PASS' if ok else 'FAIL'} 3 components: max|Δ|={err_c:.2e}")
    # one component must equal passing a bare array
    one = rng.uniform(-1e8, 1e8, 5000)
    d = float(np.max(np.abs(np.asarray(expi([one], return_np=True))
                            - np.asarray(expi(one, return_np=True)))))
    ok1 = d == 0.0
    if not ok1:
        fails.append(f"[one]!=one ({d:.1e})")
    print(f"  {'PASS' if ok1 else 'FAIL'} expi([a]) == expi(a): Δ={d:.1e}")
    # a flat list of scalars is the footgun -> must raise, not silently sum
    try:
        expi([0.3, 1.7, -2.2])
        fails.append("scalar-list did not raise")
        print("  FAIL expi([scalars]) did not raise")
    except ValueError:
        print("  PASS expi([0.3, 1.7, -2.2]) raises (scalar-list guard)")

    # ---- shape preservation + device return ------------------------------
    print("== shape / dtype ==")
    phi2 = rng.uniform(-1e8, 1e8, (128, 97))
    dev = expi(phi2)                       # device mx.array
    ok = (str(dev.dtype).endswith("complex64") and tuple(dev.shape) == (128, 97))
    err2 = float(np.max(np.abs(np.array(dev) - np.exp(1j * phi2))))
    ok = ok and err2 <= GATE_1E8
    if not ok:
        fails.append("shape/dtype")
    print(f"  {'PASS' if ok else 'FAIL'} 2D->({dev.shape}) dtype={dev.dtype} "
          f"max|Δ|={err2:.2e}")

    # ---- graceful degradation beyond the ceiling (informational) ---------
    print("== beyond EXPI_MAX_PHASE (informational — degrades, not gated) ==")
    for mag in [3e8, 1e9]:
        phi = rng.uniform(-mag, mag, 50_000)
        print(f"  |phi|<={mag:.0e}: max|Δ|={maxerr(phi):.2e}")

    # ---- offload context: GPU expi vs fp64 host np.exp (informational) ---
    print("== expi vs host np.exp(1j*phi) (informational, P=2e6) ==")
    phi = rng.uniform(-1e8, 1e8, 2_000_000)

    def tmin(fn, n=5):
        fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1000

    import mlx.core as mx
    t_gpu = tmin(lambda: mx.eval(expi(phi)))
    t_host = tmin(lambda: np.exp(1j * phi))
    print(f"  host np.exp {t_host:.0f}ms   gpu expi {t_gpu:.0f}ms   "
          f"-> {t_host / t_gpu:.1f}x")

    print()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
