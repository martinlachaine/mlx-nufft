"""FFT path correctness: the switchable FFT helpers of mlx_nufft.gpu_t3
(FFT_STRATEGY, _SMALL_AXIS_DFT / _SMALL_AXIS_DFT_LAST and
_FOUR_STEP_FUSED_TWIDDLE) against the v0.2.0 path and an fp64 numpy
reference.

Covered: fft_axis on native (640, 486, 4096), pow2 (2^17, 2^20 native to MLX,
2^21 through our four-step), non-pow2 four-step (90000, 5400) and degenerate
(16, 18, 24) lengths, each as a batched last axis, a middle axis with trailing
dims and a leading axis, both signs; fft_axis_scrambled on the four-step
lengths; fft_grid_stages (the type-3 3D chain) on grids mixing the cases; and
the FFT_STRATEGY "auto" resolution on a full-3D and a degenerate-axis grid;
and end-to-end plans (type-3 3D, type-3 1D-embedded with the 90000
four-step, type-1/2 1D with the 2^21 four-step) versus their references and
across the switch settings.

Gates: rel L2 <= 4e-6 against the fp64 FFT reference and <= 1e-6 against the v0.2.0
path for every helper-level case (fp32 FFTs land at 3e-7 to 5e-7); at plan
level every setting must stay within 1.5x the v0.2.0 path's own reference
error and within 3x its run-to-run scatter of it (see check_plan).
"""

import sys
import pathlib

import numpy as np
import mlx.core as mx
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
import mlx_nufft.gpu_t3 as g                                  # noqa: E402
from mlx_nufft.gpu_t3 import (GpuT3Plan, fft_axis,           # noqa: E402
                              fft_axis_scrambled, fft_grid_stages)
from mlx_nufft.nd import Type1PlanND, Type2PlanND            # noqa: E402
from harness.gen import gen_anisotropic, direct_sum, rel_l2  # noqa: E402

TOL_REF = 4e-6        # helper vs fp64 numpy FFT; MLX's fp32 FFT lands at
                      # 1.0e-6 to 1.3e-6 on the GitHub runner GPU, 5e-7 on an M5 Max
TOL_OLD = 1e-6        # helper vs the v0.2.0 path
TOL_PLAN = 5e-6       # plan outputs across settings

# (name, FFT_STRATEGY, _FOUR_STEP_FUSED_TWIDDLE, _SMALL_AXIS_DFT,
#  _SMALL_AXIS_DFT_LAST); "v020" is the 0.2.0 path
CONFIGS = [
    ("v020", "strided", False, 0, False),
    ("auto", "auto", True, 32, False),
    ("fused", "strided", True, 0, False),
    ("transpose", "transpose", True, 0, False),
    ("dft", "strided", True, 32, False),
    ("dft_last", "strided", True, 32, True),
    ("all", "transpose", True, 32, True),
]
DEFAULTS = (g.FFT_STRATEGY, g._FOUR_STEP_FUSED_TWIDDLE, g._SMALL_AXIS_DFT,
            g._SMALL_AXIS_DFT_LAST)


def set_config(cfg):
    _, g.FFT_STRATEGY, g._FOUR_STEP_FUSED_TWIDDLE, g._SMALL_AXIS_DFT, \
        g._SMALL_AXIS_DFT_LAST = cfg


def restore():
    g.FFT_STRATEGY, g._FOUR_STEP_FUSED_TWIDDLE, g._SMALL_AXIS_DFT, \
        g._SMALL_AXIS_DFT_LAST = DEFAULTS


def cplx(rng, shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex64)


def np_fft(x, axis, inverse):
    x = x.astype(np.complex128)
    return np.fft.ifft(x, axis=axis) if inverse else np.fft.fft(x, axis=axis)


def run_configs(fn):
    """fn() -> np.ndarray under every config; returns {name: result}."""
    out = {}
    for cfg in CONFIGS:
        set_config(cfg)
        out[cfg[0]] = fn()
    restore()
    return out


def report(fails, label, err_ref, err_old, worst_ref, worst_old, tol_ref,
           tol_old):
    ok = err_ref <= tol_ref and err_old <= tol_old
    if not ok:
        fails.append(f"{label} ref={err_ref:.2e}({worst_ref}) "
                     f"old={err_old:.2e}({worst_old})")
    print(f"  {'PASS' if ok else 'FAIL'} {label:<46} ref<={err_ref:.2e} "
          f"[{worst_ref}]  vs v020<={err_old:.2e} [{worst_old}]")


def check_axis_case(fails, rng, shape, axis, scrambled=False):
    x = cplx(rng, shape)
    n = shape[axis]
    for inverse in (False, True):
        ref = np_fft(x, axis, inverse)
        xm = mx.array(x)

        def run():
            if scrambled:
                y, (n1, n2) = fft_axis_scrambled(xm, axis, inverse, {})
                idx = np.arange(n)
                return np.array(y)[..., (idx % n1) * n2 + idx // n1]
            return np.array(fft_axis(xm, axis, inverse, {}))

        res = run_configs(run)
        old = res["v020"]
        e_ref = {k: rel_l2(v, ref) for k, v in res.items()}
        e_old = {k: rel_l2(v, old) for k, v in res.items() if k != "v020"}
        wr = max(e_ref, key=e_ref.get)
        wo = max(e_old, key=e_old.get)
        label = (f"n={n} {shape} ax{axis} isign={'+' if inverse else '-'}"
                 + (" scr" if scrambled else ""))
        report(fails, label, e_ref[wr], e_old[wo], wr, wo, TOL_REF, TOL_OLD)


def check_chain_case(fails, rng, shape):
    x = cplx(rng, shape)
    for inverse in (False, True):
        xd = x.astype(np.complex128)
        ref = np.fft.ifftn(xd) if inverse else np.fft.fftn(xd)
        xm = mx.array(x)

        def run():
            h = None
            for _ax, h in fft_grid_stages(xm, inverse, {}):
                pass
            return np.array(h)

        res = run_configs(run)
        old = res["v020"]
        e_ref = {k: rel_l2(v, ref) for k, v in res.items()}
        e_old = {k: rel_l2(v, old) for k, v in res.items() if k != "v020"}
        wr = max(e_ref, key=e_ref.get)
        wo = max(e_old, key=e_old.get)
        label = f"chain {shape} isign={'+' if inverse else '-'}"
        report(fails, label, e_ref[wr], e_old[wo], wr, wo, TOL_REF, TOL_OLD)


def check_auto(fails, rng, shape):
    """FFT_STRATEGY "auto" (the default) must resolve to "transpose" when
    every axis is longer than _SMALL_AXIS_DFT and to "strided" otherwise,
    and the chain it runs must reproduce the explicit strategy's output."""
    restore()
    expect = "transpose" if all(n > g._SMALL_AXIS_DFT for n in shape) \
        else "strided"
    got = g._strategy(shape)
    g._SMALL_AXIS_DFT = 0          # nothing is short: every grid tiles
    got0 = g._strategy(shape)
    restore()
    x = cplx(rng, shape)
    xm = mx.array(x)
    worst = 0.0
    for inverse in (False, True):
        outs = {}
        for strat in ("auto", expect):
            g.FFT_STRATEGY = strat
            h = None
            for _ax, h in fft_grid_stages(xm, inverse, {}, eager=True):
                pass
            outs[strat] = np.array(h)
        restore()
        worst = max(worst, rel_l2(outs["auto"], outs[expect]))
    ok = (DEFAULTS[0] == "auto" and got == expect and got0 == "transpose"
          and worst <= TOL_OLD)
    if not ok:
        fails.append(f"auto {shape}: default={DEFAULTS[0]} got={got} "
                     f"expect={expect} dft0={got0} diff={worst:.2e}")
    print(f"  {'PASS' if ok else 'FAIL'} auto {shape} -> {got} "
          f"(expected {expect}; {got0} at _SMALL_AXIS_DFT=0)  "
          f"vs explicit {worst:.1e}")


def check_plan(fails, label, run, ref):
    """Plan outputs under every config against the reference and against
    the v0.2.0 path. The atomic spreads make execute() nondeterministic
    (run-to-run rel L2 up to ~2e-5 at eps=1e-6 on the sigma=1.25 type-3
    grids), so the gates are set from the v0.2.0 path itself: no config may
    exceed 1.5x its reference error (+ TOL_PLAN), nor differ from it by more
    than 3x its own run-to-run scatter (or TOL_PLAN, whichever is larger)."""
    res = run_configs(run)
    set_config(CONFIGS[0])
    noise = rel_l2(run(), res["v020"])
    restore()
    old = res["v020"]
    e_ref = {k: rel_l2(v, ref) for k, v in res.items()}
    e_old = {k: rel_l2(v, old) for k, v in res.items() if k != "v020"}
    wr = max(e_ref, key=e_ref.get)
    wo = max(e_old, key=e_old.get)
    tol_ref = 1.5 * e_ref["v020"] + TOL_PLAN
    tol_old = max(TOL_PLAN, 3.0 * noise)
    report(fails, label, e_ref[wr], e_old[wo], wr, wo, tol_ref, tol_old)
    print(f"       v020 vs ref {e_ref['v020']:.1e}, run-to-run {noise:.1e}; "
          f"gates ref<={tol_ref:.1e} v020<={tol_old:.1e}")
    print("       vs ref:  " + "  ".join(f"{k} {v:.1e}"
                                         for k, v in e_ref.items()))
    print("       vs v020: " + "  ".join(f"{k} {v:.1e}"
                                         for k, v in e_old.items()))


def main():
    rng = np.random.default_rng(3)
    fails = []

    print("== fft_axis: native, pow2 and four-step lengths, all positions ==")
    for n in (640, 486, 4096, 2 ** 17, 2 ** 20, 2 ** 21, 90000, 5400):
        # batched last axis, middle axis with trailing dims, leading axis
        big = n >= 2 ** 20
        check_axis_case(fails, rng, (2 if big else 3, n), 1)
        check_axis_case(fails, rng, (2, n, 6) if big else (3, n, 5), 1)
        check_axis_case(fails, rng, (n, 4, 6) if big else (n, 7, 5), 0)
        if not g._is_native(n):
            check_axis_case(fails, rng, (3, n), 1, scrambled=True)

    print("== fft_axis: degenerate lengths (dense DFT when enabled) ==")
    for n in (16, 18, 24):
        check_axis_case(fails, rng, (5000, n), 1)
        check_axis_case(fails, rng, (300, n, n), 1)
        check_axis_case(fails, rng, (n, 300, n), 0)

    print("== fft_grid_stages: 3D chains ==")
    for shape in [(40, 36, 48), (640, 40, 36), (486, 24, 18), (64, 16, 40),
                  (5400, 24, 24), (90000, 16, 16)]:
        check_chain_case(fails, rng, shape)

    print("== FFT_STRATEGY auto resolution ==")
    check_auto(fails, rng, (640, 40, 36))       # full 3D: transpose
    check_auto(fails, rng, (90000, 16, 16))     # 1D embedding: strided

    print("== plans across settings ==")
    # type-3 3D, native axes: direct-sum reference
    prob = gen_anisotropic(N=64, P=2000, lat=0.02)
    x, c, s = prob["x"], prob["c"], prob["s"]
    f_direct = direct_sum(x, c, s, isign=+1)
    for eps in (1e-4, 1e-6):
        plan = GpuT3Plan(x, s, eps=eps, isign=+1, prec="crit64")
        check_plan(fails, f"t3 3D N=64 eps={eps:.0e} n_up={plan.n_up}",
                   lambda: plan.execute(c), f_direct)
    # type-3 1D embedded: 90000 four-step over the 16/24-cell axes
    P = M = 200_000
    x1 = rng.uniform(-300.0, 300.0, P)
    s1 = rng.uniform(-300.0, 300.0, M)
    c1 = cplx(rng, P)
    f_fin = finufft.nufft1d3(x1, c1.astype(np.complex128), s1, isign=+1,
                             eps=1e-9)
    x3 = (x1, np.zeros(P), np.zeros(P))
    s3 = (s1, np.zeros(M), np.zeros(M))
    for eps in (1e-3, 1e-5):
        plan = GpuT3Plan(x3, s3, eps=eps, isign=+1, prec="crit64")
        check_plan(fails, f"t3 1D X=S=300 eps={eps:.0e} n_up={plan.n_up}",
                   lambda: plan.execute(c1), f_fin)
    # type-1 / type-2 1D, N=2^20: the 2^21 four-step on the last axis
    N = 2 ** 20
    Mp = 200_000
    xp = rng.uniform(-np.pi, np.pi, Mp)
    cp = cplx(rng, Mp)
    fk_ref = finufft.nufft1d1(xp, cp.astype(np.complex128), N, isign=+1,
                              eps=1e-9)
    plan1 = Type1PlanND((xp,), (N,), eps=1e-6, isign=+1)
    check_plan(fails, f"t1 1D N=2^20 eps=1e-06 n_up={plan1.n_up}",
               lambda: plan1.execute(cp), fk_ref)
    fk = cplx(rng, N)
    cj_ref = finufft.nufft1d2(xp, fk.astype(np.complex128), isign=-1,
                              eps=1e-9)
    plan2 = Type2PlanND((xp,), (N,), eps=1e-6, isign=-1)
    check_plan(fails, f"t2 1D N=2^20 eps=1e-06 n_up={plan2.n_up}",
               lambda: plan2.execute(fk), cj_ref)

    print()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
