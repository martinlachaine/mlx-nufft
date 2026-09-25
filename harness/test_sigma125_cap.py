"""Gate the sigma=1.25 kernel-width cap (FINUFFT's single-precision r_dyn
guard, made dimension aware: sizing.W_MAX_FP32_LOWSIGMA_3D = 8 with three
full-band axes, sizing.W_MAX_FP32_LOWSIGMA = 10 otherwise).

(a) Widths: kernel_params(eps, 1.25, nfull) never exceeds the cap for its
    nfull over eps in [1e-16, 1e-1], engages exactly where the uncapped
    rule exceeds it and leaves the widths below it untouched; the type-3
    inner kernel follows the same rule; ND plans pass their dimension (1D
    and 2D keep w=9/10 at eps 1e-5/1e-6, 3D takes 8); GpuT3Plan counts
    its full-band axes with sizing.full_band_axes (isotropic 3, slab 2,
    rod 1, an API-embedded 1D transform 1; an axis is thin at band
    fraction <= THIN_AXIS_BAND = 0.5, so a slab at 0.6 counts three).
(b) Accuracy on the study's sigma=1.25 cases (harness/study_sigma125_cap.py:
    ND types 1/2 in 1D 2^20, 2D 512^2, 3D 128^3 with M=1e6; type 3 in 3D
    for four geometries) at eps 1e-5 and 1e-6: the plan takes the width
    the rule prescribes, its rel-L2 against the fp64 reference is within
    the ABS_GATE acceptance bound and at most FP32_GATE x CPU FINUFFT
    single precision's at the same settings (a yardstick that itself
    fails, rel-L2 above YARDSTICK_MAX as FINUFFT fp32 does in 1D at 2^20,
    is reported and skipped), and where the cap engages (three full-band
    axes) type 1 and type 3 are no worse than the uncapped width (OLD_GATE
    allows run-to-run noise) and type 2 no worse than OLD_GATE_T2 x it
    (the cap trades 19% of type-2 accuracy at eps=1e-5 for the 2x to 7x
    type-1 / type-3 gain; w=9 is type 2's own optimum in 3D).
(c) sigma=2 paths are unaffected: kernel_params(eps, 2.0, nfull) returns
    the same width and beta for every nfull and with the cap lifted over
    an eps sweep, and sigma=2 type-1 / type-2 transforms computed in the
    same run with the plan's own nfull, with nfull dropped (None) and
    with the cap lifted agree bit for bit for type 2 and to within
    run-to-run fp32 atomics noise (1.5e-7 measured; gate 1e-6) for type
    1, whose spread accumulates in varying order.

Prints PASS/FAIL per check; exits non-zero on any failure.

    .venv/bin/python harness/test_sigma125_cap.py
"""

import contextlib
import pathlib
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import rel_l2                                     # noqa: E402
from harness.study_sigma125_cap import (                           # noqa: E402
    ND_CASES, T3_GEOMS, W_UNCAPPED, kernel_variant, nd_problem, nd_reference,
    mlx_nd, fin32_nd, t3_problem, t3_nfull, t3_reference, t3_err, mlx_t3,
    fin32_t3)
from mlx_nufft import sizing                                       # noqa: E402
import mlx_nufft.nd as _nd                                         # noqa: E402
from mlx_nufft.api import _embed3                                  # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan, _inner_kernel_params       # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND                  # noqa: E402

ABS_GATE = 1e-4             # (b): the acceptance suite's rel-L2 bound
FP32_GATE = 1.5             # (b): rel-L2 <= this x FINUFFT fp32's
YARDSTICK_MAX = 1e-2        # (b): a FINUFFT fp32 result above this is unusable
OLD_GATE = 1.03             # (b): type 1 / 3 vs the uncapped width (noise)
OLD_GATE_T2 = 1.25          # (b): type 2 vs the uncapped width
T1_NOISE_GATE = 1e-6        # (c): type-1 rel-L2 between runs (atomics noise)
GATE_EPS = (1e-5, 1e-6)     # (b): eps values where the caps engage
EXPECT_W = {1e-5: {3: 8, 2: 9, 1: 9}, 1e-6: {3: 8, 2: 10, 1: 10}}
T3_NFULL = {"iso20": 3, "iso60": 3, "aniso": 2, "rod": 1}
_GPU_CAP = int(mx.device_info().get("max_buffer_length", 0))


def too_big_for_gpu(plan):
    """The type-3 slab pipeline holds several copies of the padded grid;
    the GitHub macOS runner (3.5 GiB Metal buffer cap) cannot execute the
    256-mode anisotropic case. Width checks need only the plan."""
    n_up = getattr(plan, "n_up", None)
    return bool(_GPU_CAP) and n_up is not None and \
        64 * int(np.prod(n_up)) > _GPU_CAP
SIGMA2_CASES = [("2d", (40, 36), 3000, 1e-5, 1e-6),   # tag, N, M, eps1, eps2
                ("3d", (20, 18, 16), 4000, 1e-6, 1e-5)]
SIGMA2_SEED = 20260925
FAILS = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {detail}")
    if not ok:
        FAILS.append(label)


@contextlib.contextmanager
def nfull_dropped():
    """ND plans' kernel_params sees nfull=None whatever the plan passes."""
    orig = _nd.kernel_params
    _nd.kernel_params = lambda eps, upsampfac, nfull=None: orig(eps, upsampfac)
    try:
        yield
    finally:
        _nd.kernel_params = orig


def check_widths():
    print("== (a) kernel width cap at sigma=1.25 by full-band axes ==")
    caps = {None: sizing.W_MAX_FP32_LOWSIGMA_3D, 3: sizing.W_MAX_FP32_LOWSIGMA_3D,
            2: sizing.W_MAX_FP32_LOWSIGMA, 1: sizing.W_MAX_FP32_LOWSIGMA,
            0: sizing.W_MAX_FP32_LOWSIGMA}
    check("cap constants", caps[3] == 8 and caps[2] == 10,
          f"three axes {caps[3]}, otherwise {caps[2]}")
    eps_sweep = np.logspace(-1, -16, 151)
    with kernel_variant(cap=W_UNCAPPED):
        w_old = np.array([sizing.kernel_params(float(e), 1.25)[0]
                          for e in eps_sweep])
    for nfull, cap in caps.items():
        w_new = np.array([sizing.kernel_params(float(e), 1.25, nfull)[0]
                          for e in eps_sweep])
        check(f"nfull={nfull}: w <= {cap} at every eps in [1e-16, 1e-1] and "
              f"capped exactly where the uncapped rule exceeds it",
              w_new.max() <= cap and np.array_equal(w_new, np.minimum(w_old, cap)),
              f"max w={w_new.max()}, uncapped span {w_old.min()}..{w_old.max()}")
    named = {1e-3: (5, 5, 5), 1e-4: (7, 7, 7), 3e-5: (8, 8, 8),
             1e-5: (9, 8, 9), 1e-6: (10, 8, 10), 1e-7: (12, 8, 10)}
    got = {}
    for e in named:
        with kernel_variant(cap=W_UNCAPPED):
            wo = sizing.kernel_params(e, 1.25)[0]
        got[e] = (wo, sizing.kernel_params(e, 1.25, 3)[0],
                  sizing.kernel_params(e, 1.25, 2)[0])
    check("named eps: (uncapped, three axes, two axes) widths",
          all(got[e] == named[e] for e in named),
          ", ".join(f"{e:.0e}: {got[e][0]}->{got[e][1]}/{got[e][2]}"
                    for e in named))
    beta_ok = all(sizing.kernel_params(e, 1.25, n)[1]
                  == sizing.kernel_params(3e-5, 1.25, n)[1]
                  for e in (1e-5, 1e-6) for n in (3, None))
    check("beta follows the capped width (three axes: the eps=3e-5 kernel)",
          beta_ok, f"beta={sizing.kernel_params(1e-5, 1.25, 3)[1]:.4f}")
    w_in = [max(_inner_kernel_params(float(e), 1.25, n)[0] for e in eps_sweep)
            for n in (3, 2)]
    check("type-3 inner kernel (sigma_inner=1.25) takes the same caps",
          w_in == [8, 10], f"max w2 = {w_in} for nfull 3, 2")

    # plan level: full-band axis counting
    rng = np.random.default_rng(0)
    P = 3000

    def pts(xr, sr):
        return (tuple(rng.uniform(-a, a, P) for a in xr),
                tuple(rng.uniform(-b, b, P) for b in sr))
    geoms = [("isotropic", (np.pi,) * 3, (20.0,) * 3, 3),
             ("slab, z band fraction 0.2", (np.pi, np.pi, 1e-3),
              (20.0, 20.0, 200.0), 2),
             ("rod, y/z band fraction 0.2", (np.pi, 1e-3, 1e-3),
              (20.0, 200.0, 200.0), 1),
             ("slab at band fraction 0.6 (above THIN_AXIS_BAND)",
              (np.pi, np.pi, 1e-3), (20.0, 20.0, 600.0), 3)]
    for name, xr, sr, nfull in geoms:
        x, s = pts(xr, sr)
        plans = {e: GpuT3Plan(x, s, eps=e) for e in GATE_EPS}
        ok = all(p.nfull == nfull and p.w == EXPECT_W[e][nfull]
                 for e, p in plans.items())
        check(f"GpuT3Plan {name}: nfull={nfull}, w={EXPECT_W[1e-5][nfull]}/"
              f"{EXPECT_W[1e-6][nfull]} at eps 1e-5/1e-6", ok,
              ", ".join(f"eps={e:.0e}: nfull={p.nfull} w={p.w} nf={p.nf}"
                        for e, p in plans.items()))
    x1, s1 = rng.uniform(-np.pi, np.pi, P), rng.uniform(-20.0, 20.0, P)
    p = GpuT3Plan(_embed3([x1], 1), _embed3([s1], 1), eps=1e-6)
    check("API-embedded 1D type 3: nfull=1, w=10 at eps=1e-6",
          p.nfull == 1 and p.w == 10, f"nfull={p.nfull} w={p.w} nf={p.nf}")
    for dim in (1, 2, 3):
        xs = tuple(rng.uniform(-np.pi, np.pi, 2000) for _ in range(dim))
        ws = [Type1PlanND(xs, (32,) * dim, eps=e, upsampfac=1.25).w
              for e in GATE_EPS]
        exp = [EXPECT_W[e][dim] for e in GATE_EPS]
        check(f"Type1PlanND {dim}D sigma=1.25: w={exp[0]}/{exp[1]} at "
              f"eps 1e-5/1e-6", ws == exp, f"w={ws}")


def check_sigma2_unchanged():
    print("== (c) sigma=2 paths unaffected by the cap and the nfull argument ==")
    eps_sweep = np.logspace(-1, -9, 81)
    base = [sizing.kernel_params(float(e), 2.0) for e in eps_sweep]
    same = all(sizing.kernel_params(float(e), 2.0, n) == b
               for e, b in zip(eps_sweep, base) for n in (0, 1, 2, 3))
    with kernel_variant(cap=W_UNCAPPED):
        lifted = [sizing.kernel_params(float(e), 2.0) for e in eps_sweep]
    check("kernel_params(eps, 2.0, nfull): width and beta identical for "
          "every nfull and with the cap lifted", same and lifted == base,
          f"{eps_sweep.size} eps values in [1e-9, 1e-1] x nfull None, 0..3")
    rng = np.random.default_rng(SIGMA2_SEED)
    for tag, N, M, eps1, eps2 in SIGMA2_CASES:
        dim = len(N)
        x = tuple(rng.uniform(-np.pi, np.pi, M) for _ in range(dim))
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)
             ).astype(np.complex64)
        fk = (rng.standard_normal(N) + 1j * rng.standard_normal(N)
              ).astype(np.complex64)

        def run():
            p1 = Type1PlanND(x, N, eps=eps1, isign=+1, upsampfac=2.0)
            p2 = Type2PlanND(x, N, eps=eps2, isign=-1, upsampfac=2.0)
            return (p1.w, p2.w, np.asarray(p1.execute(c)),
                    np.asarray(p2.execute(fk)))
        w1, w2, f1, c2 = run()                  # the plans pass nfull=dim
        with nfull_dropped():
            w1n, w2n, f1n, c2n = run()
        with kernel_variant(cap=W_UNCAPPED):
            w1u, w2u, f1u, c2u = run()
        diff2 = max(np.abs(c2 - c2n).max(), np.abs(c2 - c2u).max())
        check(f"t2 {tag} N={N} sigma=2 eps={eps2:.0e}: nfull=None and the "
              f"pre-cap rule give bit-identical outputs",
              np.array_equal(c2, c2n) and np.array_equal(c2, c2u)
              and w2 == w2n == w2u, f"w={w2}, max|diff|={diff2:.1e}")
        e1 = max(rel_l2(f1n, f1), rel_l2(f1u, f1))
        check(f"t1 {tag} N={N} sigma=2 eps={eps1:.0e}: nfull=None and the "
              f"pre-cap rule agree within atomics noise",
              e1 <= T1_NOISE_GATE and w1 == w1n == w1u,
              f"w={w1}, rel_l2={e1:.1e} (gate {T1_NOISE_GATE:.0e})")


def _old_gate(typ):
    return OLD_GATE_T2 if typ == "t2" else OLD_GATE


def check_vs_fp32(label, em, ef):
    check(f"{label} rel-L2 <= {ABS_GATE:.0e}", em <= ABS_GATE,
          f"mlx {em:.2e}")
    if ef > YARDSTICK_MAX:
        print(f"  NOTE {label}: FINUFFT fp32 yardstick unusable (rel-L2 "
              f"{ef:.1e}), ratio check skipped")
        return
    check(f"{label} vs finufft fp32", em <= FP32_GATE * ef,
          f"mlx {em:.2e} vs fp32 {ef:.2e} (ratio {em / ef:.2f})")


def check_accuracy():
    print("== (b) sigma=1.25 accuracy: shipped width vs CPU FINUFFT single "
          f"precision (gate {FP32_GATE}x) and vs the uncapped width ==")
    for dim, N, M, _ in ND_CASES:
        x, c, fk = nd_problem(dim, N, M)
        ref1, ref2 = nd_reference(dim, N, x, c, fk)
        for eps in GATE_EPS:
            w = Type1PlanND(tuple(x), (N,) * dim, eps=eps, upsampfac=1.25).w
            check(f"{dim}D {N}^{dim} eps={eps:.0e}: w={EXPECT_W[eps][dim]}",
                  w == EXPECT_W[eps][dim], f"w={w}")
            f1, c2 = mlx_nd(dim, N, x, c, fk, eps, 1.25)
            g1, d2 = fin32_nd(dim, N, x, c, fk, eps, 1.25)
            old = None
            if dim == 3:
                with kernel_variant(cap=W_UNCAPPED):
                    old = mlx_nd(dim, N, x, c, fk, eps, 1.25)
            for i, (typ, ref, m, f) in enumerate(((("t1", ref1, f1, g1),
                                                    ("t2", ref2, c2, d2)))):
                em, ef = rel_l2(m, ref), rel_l2(f, ref)
                label = f"{typ} {dim}D {N}^{dim} M={M:.0e} eps={eps:.0e} w={w}"
                check_vs_fp32(label, em, ef)
                if old is not None:
                    eo = rel_l2(old[i], ref)
                    check(f"{label} vs uncapped width", em <= _old_gate(typ) * eo,
                          f"capped {em:.2e} vs uncapped {eo:.2e} "
                          f"(ratio {em / eo:.2f}, gate {_old_gate(typ)})")
    for name in T3_GEOMS:
        prob = t3_problem(name)
        nfull = t3_nfull(prob)
        check(f"t3 {name}: nfull={T3_NFULL[name]}", nfull == T3_NFULL[name],
              f"nfull={nfull}")
        ref = t3_reference(prob)
        for eps in GATE_EPS:
            plan = GpuT3Plan(prob["x"], prob["s"], eps=eps, isign=+1)
            w = plan.w
            check(f"t3 {name} eps={eps:.0e}: w={EXPECT_W[eps][nfull]}",
                  w == EXPECT_W[eps][nfull], f"w={w}")
            if too_big_for_gpu(plan):
                print(f"  SKIP t3 {name} eps={eps:.0e} accuracy: grid "
                      f"{tuple(int(n) for n in plan.n_up)} exceeds this GPU's "
                      f"{_GPU_CAP / 2**30:.1f} GiB buffer cap")
                continue
            em = t3_err(mlx_t3(prob, eps), prob, ref)
            ef = t3_err(fin32_t3(prob, eps), prob, ref)
            label = f"t3 {name} eps={eps:.0e} w={w}"
            check_vs_fp32(label, em, ef)
            if nfull == 3:
                with kernel_variant(cap=W_UNCAPPED):
                    eo = t3_err(mlx_t3(prob, eps), prob, ref)
                check(f"{label} vs uncapped width", em <= OLD_GATE * eo,
                      f"capped {em:.2e} vs uncapped {eo:.2e} "
                      f"(ratio {em / eo:.2f}, gate {OLD_GATE})")


if __name__ == "__main__":
    check_widths()
    check_sigma2_unchanged()
    check_accuracy()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
