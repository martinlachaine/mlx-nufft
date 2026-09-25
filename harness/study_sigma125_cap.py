"""Accuracy study for the sigma=1.25 kernel-width cap.

FINUFFT's setup_spreadinterp caps the ES kernel width at ns=8 in single
precision at upsampfac=1.25 ("ns reducing from N to 8 to prevent
r_{dyn}-related catastrophic cancellation"). The deconvolution divides by
the kernel Fourier transform, whose dynamic range across the band,
r_dyn = phihat(0) / phihat(pi/sigma), is 34 at w=8, 56 at w=9 and 93 at
w=10 for sigma=1.25 (2 to 8 at every width for sigma=2); it multiplies
the fp32 grid's rounding error, and a target at a corner of the band sees
the product over the axes that reach the band edge. mlx-nufft's grid is
always fp32, so sizing.kernel_params caps the width for every
upsampfac < 2 by the number of full-band axes: 8 with three
(W_MAX_FP32_LOWSIGMA_3D, FINUFFT's rule), 10 otherwise
(W_MAX_FP32_LOWSIGMA); ND plans pass their dimension, type 3 counts its
axes with sizing.full_band_axes (an axis whose targets reach at most
THIN_AXIS_BAND = 0.5 of the band is thin).

This script measures what the rule buys, case by case: rel-L2 against a
CPU FINUFFT fp64 reference for the OLD (uncapped, pre-change) and NEW
(shipped) width, the error at each forced width 8 / 9 / 10 (classic
beta), the new width with the other beta variant (see below), and CPU
FINUFFT in single precision (float32 / complex64 inputs, same sigma) as
the yardstick. The per-row gate is new <= min(old, 1.15 x FINUFFT fp32).
Cases:
  - ND types 1 and 2 at sigma=1.25: 1D 2^20 (eps 1e-5 / 1e-6), 2D 512^2
    and 3D 128^3 (eps 1e-4 / 1e-5 / 1e-6), M=1e6; 2D and 3D again at
    sigma=2 as a control (the cap is inert there);
  - type 3 (sigma=1.25 hard-wired), 3D, four geometries at eps 1e-4 /
    1e-5 / 1e-6: iso20 (x in [-pi,pi]^3, s in [-20,20]^3, P=M=1e5, the
    case that motivated the cap; three full-band axes), iso60 (s in
    [-60,60]^3, P=M=5e4), aniso (harness.gen.gen_anisotropic, the
    acceptance suite's thin-slab geometry with 1e7-scale coordinates,
    N=256, P=5e4; z band fraction 0.26, so two full-band axes) and rod
    (x2, x3 in [-1e-3, 1e-3], s2, s3 in [-200, 200], band fraction 0.2 on
    both, so one full-band axis; P=M=1e5). The iso and rod references are
    CPU FINUFFT fp64 at eps=1e-8; the anisotropic one is the exact direct
    sum on a 4096-target subset (its fp64 FINUFFT grid would need tens of
    GB).

Beta variants at sigma=1.25: "classic" is the gamma=0.97 formula of
FINUFFT <= 2.3 (beta = 0.97 pi (1 - 1/(2 sigma)) w, 14.63 at w=8);
"finufft251" is the table finufft 2.5.1 prints in spread_debug (15.0 at
w=8; identical in 1D/2D/3D and in single/double). The one sizing.py ships
is detected and labelled; the gate is evaluated on it.

FINUFFT 2.5.1 single precision picks ns 7 / 8 / 8 (3D: 7 / 9 / 11 before
its own reduction to 8) at sigma=1.25 for eps 1e-4 / 1e-5 / 1e-6 and
5 / 6 / 7 at sigma=2. Widths are forced by setting both W_MAX constants
for the duration of a run (16, kernel_params' hard maximum, restores the
pre-cap rule). test_sigma125_cap.py imports the case builders below, so
the gate and the study see the same problems.

Measured (M5 Max, mlx 0.31.2, finufft 2.5.1; rel-L2 vs fp64 at eps 1e-5 /
1e-6, old (uncapped) -> new (this rule), FINUFFT fp32 at the same sigma
in brackets, w=8 in parentheses where the rule keeps a wider kernel):
    1D 2^20  t1/t2   8.9e-6 / 8.1e-6 -> same, w 9/10   [fp32 unusable: 3e2]  (2.4e-5)
    2D 512^2 t1      1.8e-5 / 3.4e-5 -> same, w 9/10   [1.4e-4]  (3.5e-5)
    2D 512^2 t2      1.4e-5 / 1.7e-5 -> same, w 9/10   [1.4e-4]  (3.5e-5)
    3D 128^3 t1      8.5e-5 / 3.2e-4 -> 4.6e-5, w 8    [5.0e-5]
    3D 128^3 t2      3.7e-5 / 1.2e-4 -> 4.4e-5, w 8    [4.5e-5]  w=9: 3.7e-5 at both
    t3 iso20         9.9e-5 / 3.3e-4 -> 5.2e-5, w 8    [4.6e-5]
    t3 iso60         1.8e-4 / 7.8e-4 -> 6.4e-5, w 8    [6.7e-5]
    t3 aniso (slab)  1.6e-5 / 2.7e-5 -> same, w 9/10   [3.4e-4: fp32 coordinates]  (3.7e-5)
    t3 rod           5.1e-6 / 7.6e-6 -> same, w 9/10   [9.7e-6]  (1.1e-5)
Gate misses (new <= min(old, 1.15 fp32)): 3D t2 at eps=1e-5, where the
cap costs 19% (w=9 is type 2's own optimum in 3D); t3 iso20 at eps=1e-6,
on the 1.15 line within run-to-run noise (1.147 at eps=1e-5); and every
eps=1e-4 row in 3D and for type 3 (w=7, no cap involved), where this
kernel sits 1.5x to 2.4x above FINUFFT fp32 at equal width, the regime
its aliasing error dominates. finufft 2.5.1's printed beta table (15.0 at
w=8 against the classic 14.63) does not transfer to this ES kernel: worse
on 23 of 28 rows (2.1x at w=7 and 8, 0.8x to 0.9x only at w=10), and the
kernel's own fp64 aliasing bound puts the optimum at gamma=0.97 for
w=7..9 (6.4e-4 vs 1.8e-3 at w=7), so it is not adopted.

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
import mlx_nufft.nd as _nd                                         # noqa: E402
import mlx_nufft.gpu_t3 as _t3                                     # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND                  # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan                             # noqa: E402

EPS_LIST = (1e-4, 1e-5, 1e-6)
# (dim, N per axis, M, eps values)
ND_CASES = [(1, 2 ** 20, 1_000_000, (1e-5, 1e-6)),
            (2, 512, 1_000_000, EPS_LIST),
            (3, 128, 1_000_000, EPS_LIST)]
ND_CASES_QUICK = [(1, 2 ** 14, 100_000, (1e-5, 1e-6)),
                  (2, 128, 100_000, EPS_LIST),
                  (3, 48, 100_000, EPS_LIST)]
ND_CONTROL_DIMS = (2, 3)    # sigma=2 control rows
T3_GEOMS = ("iso20", "iso60", "aniso", "rod")
T3_NSUB = 4096              # direct-sum target subset (anisotropic case)
REF_EPS_ND, REF_EPS_T3 = 1e-9, 1e-8
W_UNCAPPED = 16             # kernel_params' hard maximum: forcing the cap
                            # there restores the pre-cap rule
FP32_RATIO = 1.15           # gate: new <= min(old, FP32_RATIO x fp32)
NOISE = 1.03                # run-to-run fp32 atomics noise allowance

# finufft 2.5.1's ES beta at upsampfac=1.25 by width, as spread_debug
# prints it (3 significant digits)
FINUFFT251_BETA_125 = {2: 3.72, 3: 5.6, 4: 7.49, 5: 9.37, 6: 11.3, 7: 13.1,
                       8: 15.0, 9: 16.9, 10: 18.8, 11: 20.7, 12: 22.6,
                       13: 24.5, 14: 26.3, 15: 28.2, 16: 30.1}
BETA_VARIANTS = ("classic", "finufft251")


def beta_of(variant, w, sigma=1.25):
    if variant == "finufft251":
        return FINUFFT251_BETA_125[w]
    return 0.97 * np.pi * (1.0 - 1.0 / (2.0 * sigma)) * w


_KP_MODULES = (sizing, _nd, _t3)      # every module binding kernel_params
_KP_ORIG = sizing.kernel_params
SHIPPED_BETA = ("finufft251"
                if abs(_KP_ORIG(3e-5, 1.25)[1] - FINUFFT251_BETA_125[8]) < 1e-6
                else "classic")
OTHER_BETA = [v for v in BETA_VARIANTS if v != SHIPPED_BETA][0]


@contextlib.contextmanager
def kernel_variant(cap=None, beta=None):
    """Run plans with the width cap forced to `cap` (both W_MAX constants;
    W_UNCAPPED restores the pre-cap rule) and/or the sigma=1.25 beta
    variant `beta` ('classic' or 'finufft251')."""
    saved = (sizing.W_MAX_FP32_LOWSIGMA_3D, sizing.W_MAX_FP32_LOWSIGMA,
             [m.kernel_params for m in _KP_MODULES])
    if cap is not None:
        sizing.W_MAX_FP32_LOWSIGMA_3D = sizing.W_MAX_FP32_LOWSIGMA = cap
    if beta is not None:
        def kp(eps, upsampfac, nfull=None):
            w, b = _KP_ORIG(eps, upsampfac, nfull)
            return (w, beta_of(beta, w)) if upsampfac == 1.25 else (w, b)
        for m in _KP_MODULES:
            m.kernel_params = kp
    try:
        yield
    finally:
        sizing.W_MAX_FP32_LOWSIGMA_3D, sizing.W_MAX_FP32_LOWSIGMA = saved[:2]
        for m, f in zip(_KP_MODULES, saved[2]):
            m.kernel_params = f


def widths(eps, sigma, nfull):
    """(old, new, forced): the uncapped width, the shipped width for nfull
    full-band axes, and the widths worth measuring (old, new, 8..10)."""
    with kernel_variant(cap=W_UNCAPPED):
        w_old = sizing.kernel_params(eps, sigma)[0]
    w_new = sizing.kernel_params(eps, sigma, nfull)[0]
    forced = sorted({w_old, w_new} | {w for w in (8, 9, 10) if w <= w_old}) \
        if sigma < 2.0 else [w_new]
    return w_old, w_new, forced


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
    """mlx-nufft type-1 / type-2 outputs at (eps, sigma), current rule."""
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
    idx = None
    if name in ("iso20", "iso60", "rod"):
        P = M = 20_000 if quick else (50_000 if name == "iso60" else 100_000)
        band = 60.0 if name == "iso60" else 20.0
        if name == "rod":
            # sources within 1e-3 of the axis on x2, x3; targets spread to
            # 200 there: band fraction S*X = 0.2 on both thin axes
            x = (rng.uniform(-np.pi, np.pi, P), rng.uniform(-1e-3, 1e-3, P),
                 rng.uniform(-1e-3, 1e-3, P))
            s = (rng.uniform(-band, band, M), rng.uniform(-200.0, 200.0, M),
                 rng.uniform(-200.0, 200.0, M))
        else:
            x = tuple(rng.uniform(-np.pi, np.pi, P) for _ in range(3))
            s = tuple(rng.uniform(-band, band, M) for _ in range(3))
        c = rng.standard_normal(P) + 1j * rng.standard_normal(P)
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


def t3_nfull(prob):
    """Full-band axes of a type-3 problem, as GpuT3Plan counts them."""
    S = [0.5 * (float(v.max()) - float(v.min())) for v in prob["s"]]
    X = [0.5 * (float(v.max()) - float(v.min())) for v in prob["x"]]
    return sizing.full_band_axes(S, X)


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
def _e(v):
    return "" if v is None else f"{v:.2e}"


def header():
    return (f"| case | type | sigma | eps | w old->new | err old | err new "
            f"({SHIPPED_BETA}) | finufft fp32 | gate | e@8 | e@9 | e@10 "
            f"(classic beta) | new@{OTHER_BETA} |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def row_line(r):
    gate = "ok" if r["gate"] else "MISS"
    return (f"| {r['case']} | {r['typ']} | {r['sigma']:g} | {r['eps']:.0e} "
            f"| {r['w_old']} -> {r['w_new']} | {_e(r['err_old'])} "
            f"| {_e(r['err_new'])} | {_e(r['err_fin'])} | {gate} "
            f"| {_e(r['e'].get(8))} | {_e(r['e'].get(9))} | {_e(r['e'].get(10))} "
            f"| {_e(r['err_other'])} |")


def make_row(case, typ, sigma, eps, w_old, w_new, e_classic, e_other,
             err_fin):
    """e_classic: {width: err} with the classic beta; e_other: err at w_new
    with the other beta variant (None at sigma=2)."""
    err_old = e_classic[w_old]
    if SHIPPED_BETA == "classic" or sigma == 2.0:
        err_new, err_other = e_classic[w_new], e_other
    else:
        err_new, err_other = e_other, e_classic[w_new]
    gate = err_new <= min(err_old * NOISE, FP32_RATIO * err_fin)
    return dict(case=case, typ=typ, sigma=sigma, eps=eps, w_old=w_old,
                w_new=w_new, err_old=err_old, err_new=err_new,
                err_fin=err_fin, gate=gate, e=e_classic, err_other=err_other)


def main(quick=False):
    rows = []
    print(f"# shipped beta variant at sigma=1.25: {SHIPPED_BETA}")
    print(header(), flush=True)
    nd_cases = ND_CASES_QUICK if quick else ND_CASES
    for dim, N, M, eps_list in nd_cases:
        x, c, fk = nd_problem(dim, N, M)
        t0 = time.perf_counter()
        ref1, ref2 = nd_reference(dim, N, x, c, fk)
        case = f"{dim}D {N}^{dim} M={M:.0e}"
        print(f"# [{case}] fp64 references (eps={REF_EPS_ND:g}) in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        sigmas = (1.25, 2.0) if dim in ND_CONTROL_DIMS else (1.25,)
        for sigma in sigmas:
            for eps in eps_list:
                w_old, w_new, forced = widths(eps, sigma, dim)
                e1, e2 = {}, {}
                for w in forced:
                    with kernel_variant(cap=w, beta="classic"):
                        f1, c2 = mlx_nd(dim, N, x, c, fk, eps, sigma)
                    e1[w], e2[w] = rel_l2(f1, ref1), rel_l2(c2, ref2)
                o1 = o2 = None
                if sigma < 2.0:
                    with kernel_variant(cap=w_new, beta="finufft251"):
                        f1, c2 = mlx_nd(dim, N, x, c, fk, eps, sigma)
                    o1, o2 = rel_l2(f1, ref1), rel_l2(c2, ref2)
                f1f, c2f = fin32_nd(dim, N, x, c, fk, eps, sigma)
                rows.append(make_row(case, "t1", sigma, eps, w_old, w_new,
                                     e1, o1, rel_l2(f1f, ref1)))
                print(row_line(rows[-1]), flush=True)
                rows.append(make_row(case, "t2", sigma, eps, w_old, w_new,
                                     e2, o2, rel_l2(c2f, ref2)))
                print(row_line(rows[-1]), flush=True)
    for name in T3_GEOMS:
        prob = t3_problem(name, quick=quick)
        nfull = t3_nfull(prob)
        t0 = time.perf_counter()
        ref = t3_reference(prob)
        P, M = prob["x"][0].size, prob["s"][0].size
        sub = ("" if prob["idx"] is None
               else f" (err on {prob['idx'].size} targets)")
        case = f"t3 {name} P={P:.0e} M={M:.0e} nfull={nfull}"
        print(f"# [{case}] reference in {time.perf_counter() - t0:.1f}s{sub}",
              flush=True)
        for eps in EPS_LIST:
            w_old, w_new, forced = widths(eps, 1.25, nfull)
            e = {}
            for w in forced:
                with kernel_variant(cap=w, beta="classic"):
                    e[w] = t3_err(mlx_t3(prob, eps), prob, ref)
            with kernel_variant(cap=w_new, beta="finufft251"):
                o = t3_err(mlx_t3(prob, eps), prob, ref)
            ef = t3_err(fin32_t3(prob, eps), prob, ref)
            rows.append(make_row(case, "t3", 1.25, eps, w_old, w_new, e, o,
                                 ef))
            print(row_line(rows[-1]), flush=True)
    print("\n" + header())
    for r in rows:
        print(row_line(r))
    miss = [r for r in rows if not r["gate"]]
    print(f"\n# gate new <= min(old x {NOISE}, {FP32_RATIO} x fp32): "
          f"{len(rows) - len(miss)} of {len(rows)} rows ok"
          + ("" if not miss else "; MISS: " + "; ".join(
              f"{r['case']} {r['typ']} eps={r['eps']:.0e}" for r in miss)))
    # beta decision: the new width with each variant, sigma=1.25 rows
    print(f"\n# beta at sigma=1.25, new width, classic vs finufft251 "
          f"(ratio finufft251/classic; within noise = {NOISE}):")
    worse = 0
    for r in rows:
        if r["sigma"] == 2.0:
            continue
        ec, ef = ((r["err_new"], r["err_other"]) if SHIPPED_BETA == "classic"
                  else (r["err_other"], r["err_new"]))
        ratio = ef / ec
        worse += ratio > NOISE
        print(f"  {r['case']} {r['typ']} eps={r['eps']:.0e} w={r['w_new']}: "
              f"classic {ec:.2e}  finufft251 {ef:.2e}  ratio {ratio:.2f}"
              + ("  WORSE" if ratio > NOISE else ""))
    print(f"# finufft251 beta worse than classic beyond noise on {worse} rows")
    return rows


if __name__ == "__main__":
    main(quick="--quick" in sys.argv[1:])
