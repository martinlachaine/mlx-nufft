"""Accuracy study for the sigma=1.25 kernel-width cap.

FINUFFT's setup_spreadinterp caps the ES kernel width at ns=8 in single
precision at upsampfac=1.25 ("ns reducing from N to 8 to prevent
r_{dyn}-related catastrophic cancellation"). The deconvolution divides by
the kernel Fourier transform, whose dynamic range across the band,
r_dyn = phihat(0) / phihat(pi/sigma), is 34 at w=8, 93 at w=10 and 1863 at
w=16 for sigma=1.25 (vs 2 to 8 at every width for sigma=2); it multiplies
the fp32 grid's rounding error, so past w=8 a wider kernel makes the result
worse, not better. mlx-nufft's grid is always fp32, so sizing.kernel_params
applies the same cap (sizing.W_MAX_FP32_LOWSIGMA) for every upsampfac < 2.

This script measures what the cap buys, case by case: rel-L2 against a CPU
FINUFFT fp64 reference for the OLD (uncapped) and NEW (capped) width, with
CPU FINUFFT in single precision (float32 / complex64 inputs, same sigma) as
the yardstick. Cases:
  - ND types 1 and 2, sigma=1.25, 2D 512^2 and 3D 128^3, M=1e6, eps
    1e-4 / 1e-5 / 1e-6; the same at sigma=2 as a control (cap inert);
  - type 3 (sigma=1.25 hard-wired), 3D, three geometries at the same eps:
    iso20 (x in [-pi,pi]^3, s in [-20,20]^3, P=M=1e5, the case that
    motivated the cap), iso60 (s in [-60,60]^3, P=M=5e4, a wider band) and
    aniso (harness.gen.gen_anisotropic: the acceptance suite's thin-slab
    geometry with 1e7-scale coordinates, N=256, P=5e4). The isotropic
    references are CPU FINUFFT fp64 at eps=1e-8; the anisotropic one is the
    exact direct sum on a 4096-target subset (its fp64 FINUFFT grid would
    need tens of GB).

FINUFFT 2.5.1 single precision picks ns = 7 / 8 / 8 at sigma=1.25 for eps
1e-4 / 1e-5 / 1e-6 (the last through its "ns reducing from 10 to 8"
warning) and 5 / 6 / 7 at sigma=2. The OLD width is obtained by lifting the
cap to kernel_params' hard maximum (16) for the duration of a run.
test_sigma125_cap.py imports the case builders below, so the gate and the
study see the same problems.

Measured (M5 Max, mlx 0.31.2, finufft 2.5.1; rel-L2 vs fp64 at eps 1e-5 /
1e-6, uncapped -> capped, FINUFFT fp32 at the same sigma in brackets):
    3D 128^3 t1   8.6e-5 / 3.2e-4 -> 4.6e-5   [5.0e-5]
    3D 128^3 t2   3.7e-5 / 1.2e-4 -> 4.4e-5   [4.5e-5]
    2D 512^2 t1   1.8e-5 / 3.4e-5 -> 3.5e-5   [1.4e-4]
    2D 512^2 t2   1.4e-5 / 1.7e-5 -> 3.5e-5   [1.4e-4]
    t3 iso20      9.5e-5 / 3.7e-4 -> 5.2e-5   [4.6e-5]
    t3 iso60      1.8e-4 / 8.0e-4 -> 6.6e-5   [6.7e-5]
    t3 aniso      1.6e-5 / 2.6e-5 -> 3.7e-5   [3.4e-4: fp32 coordinates]
The cap is a clear win in 3D, where a corner target sees the amplification
cubed; the 2D grids and the thin-slab geometry (its third axis carries no
band, so it is effectively 2D) were better off at w=9, though they stay
4x to 9x ahead of FINUFFT fp32 at the same sigma. sigma=2 is inert.

Usage (accuracy only, no timing):

    .venv/bin/python harness/study_sigma125_cap.py [--quick]
"""

import contextlib
import pathlib
import sys
import time

import numpy as np
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum_mp, rel_l2    # noqa: E402
from mlx_nufft import sizing                                       # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND                  # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan                             # noqa: E402

EPS_LIST = (1e-4, 1e-5, 1e-6)
ND_CASES = [(2, 512, 1_000_000), (3, 128, 1_000_000)]   # (dim, N/axis, M)
ND_CASES_QUICK = [(2, 128, 100_000), (3, 48, 100_000)]
T3_GEOMS = ("iso20", "iso60", "aniso")
T3_NSUB = 4096              # direct-sum target subset (anisotropic case)
REF_EPS_ND, REF_EPS_T3 = 1e-9, 1e-8
W_UNCAPPED = 16             # kernel_params' hard maximum: lifting the cap
                            # there restores the pre-cap rule


@contextlib.contextmanager
def width_cap(value):
    """Temporarily set sizing.W_MAX_FP32_LOWSIGMA (W_UNCAPPED: old rule)."""
    old = sizing.W_MAX_FP32_LOWSIGMA
    sizing.W_MAX_FP32_LOWSIGMA = value
    try:
        yield
    finally:
        sizing.W_MAX_FP32_LOWSIGMA = old


def widths(eps, sigma):
    """(old, new) kernel widths for eps at sigma."""
    with width_cap(W_UNCAPPED):
        w_old = sizing.kernel_params(eps, sigma)[0]
    return w_old, sizing.kernel_params(eps, sigma)[0]


# ---- ND types 1 and 2 ----------------------------------------------------
def nd_problem(dim, N, M, seed=0):
    rng = np.random.default_rng(seed)
    x = [rng.uniform(-np.pi, np.pi, M) for _ in range(dim)]
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    fk = (rng.standard_normal((N,) * dim)
          + 1j * rng.standard_normal((N,) * dim))
    return x, c, fk


def nd_reference(dim, N, x, c, fk):
    """CPU FINUFFT fp64 (eps=1e-9) type-1 and type-2 references."""
    f1 = getattr(finufft, f"nufft{dim}d1")(*x, c, (N,) * dim,
                                           eps=REF_EPS_ND, isign=+1)
    c2 = getattr(finufft, f"nufft{dim}d2")(*x, fk, eps=REF_EPS_ND, isign=-1)
    return f1, c2


def mlx_nd(dim, N, x, c, fk, eps, sigma):
    """mlx-nufft type-1 / type-2 outputs at (eps, sigma), current cap."""
    p1 = Type1PlanND(tuple(x), (N,) * dim, eps=eps, isign=+1,
                     upsampfac=sigma)
    f1 = np.asarray(p1.execute(c.astype(np.complex64)))
    p2 = Type2PlanND(tuple(x), (N,) * dim, eps=eps, isign=-1,
                     upsampfac=sigma)
    c2 = np.asarray(p2.execute(fk.astype(np.complex64)))
    return f1, c2


def fin32_nd(dim, N, x, c, fk, eps, sigma):
    """CPU FINUFFT single precision (float32 / complex64) at the same sigma."""
    x32 = [v.astype(np.float32) for v in x]
    f1 = getattr(finufft, f"nufft{dim}d1")(*x32, c.astype(np.complex64),
                                           (N,) * dim, eps=eps, isign=+1,
                                           upsampfac=sigma)
    c2 = getattr(finufft, f"nufft{dim}d2")(*x32, fk.astype(np.complex64),
                                           eps=eps, isign=-1, upsampfac=sigma)
    return f1, c2


# ---- type 3 ---------------------------------------------------------------
def t3_problem(name, seed=0, quick=False):
    """One of the study's 3D type-3 geometries; idx is the target subset the
    error is measured on (None: every target)."""
    rng = np.random.default_rng(seed)
    if name in ("iso20", "iso60"):
        P = M = 20_000 if quick else (100_000 if name == "iso20" else 50_000)
        band = 20.0 if name == "iso20" else 60.0
        x = tuple(rng.uniform(-np.pi, np.pi, P) for _ in range(3))
        s = tuple(rng.uniform(-band, band, M) for _ in range(3))
        c = rng.standard_normal(P) + 1j * rng.standard_normal(P)
        idx = None
    elif name == "aniso":
        prob = gen_anisotropic(N=64 if quick else 256,
                               P=20_000 if quick else 50_000, lat=1.5,
                               seed=seed)
        x, c, s = prob["x"], prob["c"], prob["s"]
        nsub = min(T3_NSUB, s[0].size)
        idx = np.sort(rng.choice(s[0].size, nsub, replace=False))
    else:
        raise ValueError(f"unknown type-3 geometry {name!r}")
    return dict(name=name, x=x, c=c, s=s, idx=idx)


def t3_reference(prob):
    """CPU FINUFFT fp64 (eps=1e-8), or the exact direct sum on prob['idx']."""
    if prob["idx"] is None:
        return finufft.nufft3d3(*prob["x"], prob["c"], *prob["s"],
                                eps=REF_EPS_T3, isign=+1)
    return direct_sum_mp(prob["x"], prob["c"], prob["s"], isign=+1,
                         idx=prob["idx"])


def t3_err(f, prob, ref):
    f = np.asarray(f)
    if prob["idx"] is not None:
        f = f[prob["idx"]]
    return rel_l2(f, ref)


def mlx_t3(prob, eps):
    plan = GpuT3Plan(prob["x"], prob["s"], eps=eps, isign=+1, prec="crit64")
    return np.asarray(plan.execute(prob["c"].astype(np.complex64)))


def fin32_t3(prob, eps):
    x32 = [np.asarray(v, dtype=np.float32) for v in prob["x"]]
    s32 = [np.asarray(v, dtype=np.float32) for v in prob["s"]]
    return finufft.nufft3d3(*x32, prob["c"].astype(np.complex64), *s32,
                            eps=eps, isign=+1, upsampfac=1.25)


# ---- study ----------------------------------------------------------------
HEADER = ("| case | type | sigma | eps | w old -> new | err old | err new "
          "| finufft fp32 | new / fp32 |\n"
          "|---|---|---|---|---|---|---|---|---|")


def row_line(r):
    return (f"| {r['case']} | {r['typ']} | {r['sigma']:g} | {r['eps']:.0e} "
            f"| {r['w_old']} -> {r['w_new']} | {r['err_old']:.2e} "
            f"| {r['err_new']:.2e} | {r['err_fin']:.2e} "
            f"| {r['err_new'] / r['err_fin']:.2f} |")


def main(quick=False):
    rows = []
    nd_cases = ND_CASES_QUICK if quick else ND_CASES
    print(HEADER, flush=True)
    for dim, N, M in nd_cases:
        x, c, fk = nd_problem(dim, N, M)
        t0 = time.perf_counter()
        ref1, ref2 = nd_reference(dim, N, x, c, fk)
        case = f"{dim}D {N}^{dim} M={M:.0e}"
        print(f"# [{case}] fp64 references (eps={REF_EPS_ND:g}) in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        for sigma in (1.25, 2.0):
            for eps in EPS_LIST:
                w_old, w_new = widths(eps, sigma)
                f1n, c2n = mlx_nd(dim, N, x, c, fk, eps, sigma)
                if w_old != w_new:
                    with width_cap(W_UNCAPPED):
                        f1o, c2o = mlx_nd(dim, N, x, c, fk, eps, sigma)
                else:
                    f1o, c2o = f1n, c2n
                f1f, c2f = fin32_nd(dim, N, x, c, fk, eps, sigma)
                for typ, ref, o, n, f in ((1, ref1, f1o, f1n, f1f),
                                          (2, ref2, c2o, c2n, c2f)):
                    rows.append(dict(case=case, typ=f"t{typ}", sigma=sigma,
                                     eps=eps, w_old=w_old, w_new=w_new,
                                     err_old=rel_l2(o, ref),
                                     err_new=rel_l2(n, ref),
                                     err_fin=rel_l2(f, ref)))
                    print(row_line(rows[-1]), flush=True)
    for name in T3_GEOMS:
        prob = t3_problem(name, quick=quick)
        t0 = time.perf_counter()
        ref = t3_reference(prob)
        P, M = prob["x"][0].size, prob["s"][0].size
        sub = "" if prob["idx"] is None else f" (err on {prob['idx'].size} targets)"
        case = f"t3 {name} P={P:.0e} M={M:.0e}"
        print(f"# [{case}] reference in {time.perf_counter() - t0:.1f}s{sub}",
              flush=True)
        for eps in EPS_LIST:
            w_old, w_new = widths(eps, 1.25)
            fn = mlx_t3(prob, eps)
            if w_old != w_new:
                with width_cap(W_UNCAPPED):
                    fo = mlx_t3(prob, eps)
            else:
                fo = fn
            ff = fin32_t3(prob, eps)
            rows.append(dict(case=case, typ="t3", sigma=1.25, eps=eps,
                             w_old=w_old, w_new=w_new,
                             err_old=t3_err(fo, prob, ref),
                             err_new=t3_err(fn, prob, ref),
                             err_fin=t3_err(ff, prob, ref)))
            print(row_line(rows[-1]), flush=True)
    print("\n" + HEADER)
    for r in rows:
        print(row_line(r))
    return rows


if __name__ == "__main__":
    main(quick="--quick" in sys.argv[1:])
