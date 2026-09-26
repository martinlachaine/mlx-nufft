"""Gate the kernel-width and spread-path rules that the paper's harness relies on.

(a) The auto spread path of Type1PlanND takes the tile (OD) kernels at every
    kernel width once a plan has at least 20000 points, in dims 1, 2 and 3,
    and plans under 20000 points keep the direct global-atomic (GM) spread.
(b) At the narrow widths (w = 2 at eps 1e-1, w = 3 at eps 1e-2) the tile
    path agrees with the forced GM path on random and clustered points to
    within fp32 accumulation-order noise, and at eps 1e-2 both stay within
    the harness gate 4*eps + 2e-4 of CPU FINUFFT fp64 (eps=1e-9).
(c) The reference pipeline behind harness/diagnose_ref.py (RefT3Plan) picks
    the same kernel width as the library's type-3 plan, which counts the
    axes that carry a full band, for slab, generic and rod geometries.
(d) kernel_params is the fp32 low-sigma cap applied to finufft_width, and
    run_acceptance.cpu_pred_gib models the double-precision CPU reference
    with FINUFFT's own width (15.4 GiB for the anisotropic acceptance case
    at eps 1e-5).

Prints PASS/FAIL per check; exits non-zero on any failure.

    .venv/bin/python harness/test_widths_and_paths.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import finufft                                                   # noqa: E402
from gen import gen_anisotropic, gen_generic, uniform_points     # noqa: E402
from mlx_nufft import Type1PlanND, Type3Plan, sizing              # noqa: E402
from mlx_nufft.ref_t3 import RefT3Plan                            # noqa: E402

FAILS = []
NOISE = 1e-4          # tile vs GM rel-L2: atomics order, up to ~2e-5 clustered
SIZES = {1: 2 ** 16, 2: 256, 3: 48}
CPU = {1: finufft.nufft1d1, 2: finufft.nufft2d1, 3: finufft.nufft3d1}


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def path(plan):
    return "od" if (plan._od or plan._od_ex) else "gm"


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_paths():
    print("== (a) auto spread path by width and point count ==")
    rng = np.random.default_rng(3)
    for dim in (1, 2, 3):
        N = (SIZES[dim],) * dim
        for P, want in ((100_000, "od"), (5_000, "gm")):
            x = tuple(rng.uniform(-np.pi, np.pi, P) for _ in range(dim))
            for eps in (1e-1, 1e-2, 1e-3):
                p = Type1PlanND(x, N, eps=eps, isign=1, upsampfac=2.0)
                check(f"{dim}D P={P} eps={eps:.0e} w={p.w}: auto path {path(p)}",
                      path(p) == want, f"want {want}")


def check_narrow_widths():
    print("== (b) tile vs direct spread at w = 2 and 3, random and clustered ==")
    for dim in (1, 2, 3):
        for dist in ("rand", "cluster"):
            x, N, M = uniform_points(dim, SIZES[dim], rho=1.0, dist=dist,
                                     sigma=2.0, seed=5)
            rng = np.random.default_rng(6)
            c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
            ref = None
            for eps in (1e-1, 1e-2):
                pa = Type1PlanND(tuple(x), N, eps=eps, isign=1, upsampfac=2.0)
                pg = Type1PlanND(tuple(x), N, eps=eps, isign=1, upsampfac=2.0,
                                 spread_method="gm")
                fa = pa.execute(c.astype(np.complex64))
                fg = pg.execute(c.astype(np.complex64))
                d = rel(fa, fg)
                check(f"{dim}D {dist} eps={eps:.0e} w={pa.w}: tile ({path(pa)}) vs GM",
                      path(pa) == "od" and d <= NOISE, f"rel diff {d:.1e}")
                if eps == 1e-2:
                    if ref is None:
                        ref = CPU[dim](*x, c, N, eps=1e-9, isign=1)
                    thr = 4 * eps + 2e-4
                    ea, eg = rel(fa, ref), rel(fg, ref)
                    check(f"{dim}D {dist} eps=1e-02 vs cpu", ea <= thr and eg <= thr,
                          f"tile {ea:.1e} GM {eg:.1e} (gate {thr:.1e})")


def check_reference_width():
    print("== (c) reference pipeline width == library width ==")
    rng = np.random.default_rng(7)
    rod_x = tuple(rng.uniform(-np.pi, np.pi, 2000) for _ in range(3))
    rod_s = (rng.uniform(-40, 40, 2000), rng.uniform(-1e-3, 1e-3, 2000),
             rng.uniform(-1e-3, 1e-3, 2000))
    geoms = (("slab", gen_anisotropic(N=64, P=2000, lat=0.02)),
             ("generic", gen_generic(N=64, P=2000, seed=0)),
             ("rod", dict(x=rod_x, s=rod_s)))
    for name, g in geoms:
        for eps in (1e-4, 1e-5, 1e-6):
            r = RefT3Plan(g["x"], g["s"], eps=eps)
            lib = Type3Plan(g["x"], g["s"], eps=eps, isign=1)
            check(f"{name} eps={eps:.0e}: reference w={r.w} library w={lib.w}",
                  r.w == lib.w and r.nfull == lib.nfull,
                  f"nfull {r.nfull} vs {lib.nfull}")


def check_width_helpers():
    print("== (d) width helpers and the CPU memory model ==")
    ok = all(sizing.kernel_params(e, s, nf)[0]
             == sizing.cap_kernel_width(sizing.finufft_width(e, s), s, nf)
             for s in (2.0, 1.25) for nf in (None, 1, 2, 3)
             for e in np.logspace(-9, -1, 33))
    check("kernel_params == cap(finufft_width) over 33 eps x 2 sigma x 4 nfull", ok)
    check("finufft_width(1e-5, 1.25) == 9 (double-precision FINUFFT)",
          sizing.finufft_width(1e-5, 1.25) == 9)
    import run_acceptance as ra
    g = gen_anisotropic(N=512, P=10_000, lat=1.5, seed=0)
    gib = ra.cpu_pred_gib(g["x"], g["s"], 1e-5)
    check("cpu_pred_gib anisotropic acceptance case eps=1e-5", abs(gib - 15.4) < 0.1,
          f"{gib:.2f} GiB")


if __name__ == "__main__":
    check_paths()
    check_narrow_widths()
    check_reference_width()
    check_width_helpers()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
