"""Gate the default upsampling-factor policy of the ND type-1/2 plans
(nd.py: upsampfac=None / "auto" resolves per plan; nd.UPSAMP_AUTO_*).

(a) Resolution table over dims 1/2/3, both types, eps in {1e-2, 1e-3,
    1e-4, 1e-5} plus the policy's own thresholds, and mode counts under,
    one short of, at and one past the UPSAMP_AUTO_MIN_MODES floor plus
    the 256^3 class: nd._resolve_upsampfac agrees with an independent
    statement of the rule (dims below UPSAMP_AUTO_MIN_DIM and grids under
    UPSAMP_AUTO_MIN_MODES modes keep 2.0; 3D type 2 takes 1.25 at eps >=
    UPSAMP_AUTO_EPS_T2; 3D type 1 at eps >= UPSAMP_AUTO_EPS_T1, or at eps
    >= UPSAMP_AUTO_EPS_T1_BIG with at least UPSAMP_AUTO_T1_BIG_MODES
    modes), and plans built at a subset of the table carry it
    (plan.upsampfac, plan.sigma, the kernel width). The shipped constants
    are pinned.
(b) An explicit upsampfac (2.0 or 1.25, also under the mode floor) is
    honored by the plans, by the functional API and by Plan (setpts and
    the lazily built adjoint); the None / "auto" / 0 sentinels resolve
    the policy; MLX_NUFFT_UPSAMPFAC replaces the resolved default, is
    beaten by an explicit value, "auto" restores the policy and an
    unparseable value raises.
(c) An auto plan's output equals an explicit plan's at the resolved sigma
    bit for bit (type 2, whose gather is deterministic; type 1 within the
    T1_NOISE_GATE atomics allowance when two explicit runs differ) and
    differs from the other sigma's; nufft3d2 and Plan(2) give the same
    bits as the explicit plan.
(d) Accuracy of the 3D auto cases vs CPU FINUFFT fp64 (eps=1e-9) at eps
    1e-3 and 1e-4, both types, 64^3, M=2e5, through the plans and the
    functional API: rel-L2 within the harness gate 4*eps + 2e-4; where
    auto chose 1.25 the sigma=2 error at the same eps is printed.

Prints PASS/FAIL per check; exits non-zero on any failure.

    .venv/bin/python harness/test_upsampfac_auto.py
"""

import contextlib
import os
import pathlib
import sys

import numpy as np
import finufft as cpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import rel_l2                                     # noqa: E402
import mlx_nufft as gpu                                            # noqa: E402
import mlx_nufft.nd as nd                                          # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND                  # noqa: E402
from mlx_nufft.sizing import kernel_params                         # noqa: E402

EPS_TABLE = (1e-2, 1e-3, 1e-4, 1e-5)
SMALL = {1: (4096,), 2: (64, 60), 3: (32, 30, 28)}      # 3D: 26880 modes
BIG = {1: (2 ** 24,), 2: (4096, 4096), 3: (256, 256, 256)}   # 2^24 modes
FLOOR3 = ((32, 32, 31), (32, 32, 32), (33, 32, 32))   # 2^15 -1024, 2^15, +1024
MID3 = (36, 32, 30)         # 34560 modes: the 3D working grid above the floor
P_PLAN = 3000               # points for the plan-level checks
P_BITS = 20000              # (c)
ACC_N = 64                  # (d): 64^3 modes
ACC_M = 200_000
ACC_EPS = (1e-3, 1e-4)
T1_NOISE_GATE = 1e-6        # type-1 rel-L2 between runs (atomics noise)
SEED = 20260925
FAILS = []
rng = np.random.default_rng(SEED)


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {detail}")
    if not ok:
        FAILS.append(label)


def pts(dim, n):
    return tuple(rng.uniform(-np.pi, np.pi, n) for _ in range(dim))


def cplx(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            ).astype(np.complex64)


def rule(dim, typ, eps, n_modes, P=None):
    """The documented rule, stated independently of nd._resolve_upsampfac."""
    low, high = nd.UPSAMP_AUTO_LOW, nd.UPSAMP_AUTO_HIGH
    if dim < nd.UPSAMP_AUTO_MIN_DIM:
        return high
    modes = int(np.prod(n_modes))
    if modes < nd.UPSAMP_AUTO_MIN_MODES:
        return high
    rho = 0.0 if P is None else P / (2 ** dim * modes)
    if typ == 2:
        if eps >= nd.UPSAMP_AUTO_EPS_T2_ANY_RHO:
            return low
        return low if (eps >= nd.UPSAMP_AUTO_EPS_T2
                       and rho < nd.UPSAMP_AUTO_T2_MAX_RHO) else high
    if rho >= nd.UPSAMP_AUTO_T1_MAX_RHO:
        return high
    if eps >= nd.UPSAMP_AUTO_EPS_T1:
        return low
    if modes >= nd.UPSAMP_AUTO_T1_BIG_MODES:
        return low if eps >= nd.UPSAMP_AUTO_EPS_T1_BIG else high
    return high


def other(sigma):
    return (nd.UPSAMP_AUTO_HIGH if sigma == nd.UPSAMP_AUTO_LOW
            else nd.UPSAMP_AUTO_LOW)


@contextlib.contextmanager
def env_upsampfac(value):
    """MLX_NUFFT_UPSAMPFAC set to value for the block, then unset again."""
    os.environ[nd.UPSAMP_ENV] = value
    try:
        yield
    finally:
        os.environ.pop(nd.UPSAMP_ENV, None)


def check_table():
    print("== (a) resolution table ==")
    shipped = (nd.UPSAMP_AUTO_MIN_DIM, nd.UPSAMP_AUTO_MIN_MODES,
               nd.UPSAMP_AUTO_EPS_T2, nd.UPSAMP_AUTO_EPS_T1,
               nd.UPSAMP_AUTO_EPS_T1_BIG, nd.UPSAMP_AUTO_T1_BIG_MODES,
               nd.UPSAMP_AUTO_LOW, nd.UPSAMP_AUTO_HIGH)
    check("shipped constants",
          shipped == (3, 2 ** 15, 1e-4, 1e-3, 1e-4, 2 ** 24, 1.25, 2.0),
          "MIN_DIM 3, MIN_MODES 2^15, EPS_T2 1e-4, EPS_T1 1e-3, EPS_T1_BIG "
          "1e-4, T1_BIG_MODES 2^24, low 1.25, high 2.0")
    floor = nd.UPSAMP_AUTO_MIN_MODES
    sizes = [int(np.prod(N)) for N in FLOOR3]
    check("floor rows straddle UPSAMP_AUTO_MIN_MODES",
          sizes == [floor - 1024, floor, floor + 1024]
          and int(np.prod(SMALL[3])) < floor <= int(np.prod(MID3)),
          f"{sizes}, small {int(np.prod(SMALL[3]))}, "
          f"working {int(np.prod(MID3))}")
    eps_list = sorted({*EPS_TABLE, nd.UPSAMP_AUTO_EPS_T2,
                       nd.UPSAMP_AUTO_EPS_T1, nd.UPSAMP_AUTO_EPS_T1_BIG},
                      reverse=True)
    print(f"    {'dim type modes':<20}"
          + "".join(f"{e:>8.0e}" for e in eps_list))
    bad = []
    n_rows = 0
    for dim in (1, 2, 3):
        grids = (SMALL[dim], BIG[dim]) if dim < 3 else \
            (SMALL[3],) + FLOOR3 + (BIG[3],)
        for typ in (1, 2):
            for N in grids:
                got = [nd._resolve_upsampfac(None, dim, typ, e, N)[0]
                       for e in eps_list]
                exp = [rule(dim, typ, e, N) for e in eps_list]
                if got != exp:
                    bad.append((dim, typ, N))
                n_rows += 1
                print(f"    {dim}D  t{typ}  {int(np.prod(N)):>10d}"
                      + "".join(f"{g:>8.2f}" for g in got))
    check("resolver matches the rule over the table", not bad,
          f"{n_rows} rows x {len(eps_list)} eps, mismatches {bad}")
    under = (BIG[3][0] - 1,) + BIG[3][1:]
    got = nd._resolve_upsampfac(None, 3, 1, nd.UPSAMP_AUTO_EPS_T1_BIG,
                                under)[0]
    check(f"type 1 3D N={under} (one row short of the big grid) at "
          "eps=UPSAMP_AUTO_EPS_T1_BIG: 2.0",
          got == nd.UPSAMP_AUTO_HIGH
          and rule(3, 1, nd.UPSAMP_AUTO_EPS_T1_BIG, under) == got,
          f"upsampfac={got}")
    check("resolver names its rule", all(
        nd._resolve_upsampfac(v, 3, 2, 1e-3, MID3)[1] == r
        for v, r in ((2.0, "explicit"), (1.25, "explicit"))
    ) and nd._resolve_upsampfac(None, 3, 2, 1e-3, MID3)[1] != "explicit",
          "explicit values report 'explicit', the policy its clause")

    x = {d: pts(d, P_PLAN) for d in (1, 2, 3)}
    cases = [(1, 1, SMALL[1], 1e-3), (1, 2, SMALL[1], 1e-2),
             (2, 1, SMALL[2], 1e-3), (2, 2, SMALL[2], 1e-2),
             (3, 1, SMALL[3], 1e-3), (3, 2, SMALL[3], 1e-2),
             (3, 1, FLOOR3[0], 1e-3), (3, 1, FLOOR3[1], 1e-3),
             (3, 2, FLOOR3[0], 1e-3), (3, 2, FLOOR3[1], 1e-3),
             (3, 2, MID3, 1e-2), (3, 2, MID3, 1e-4), (3, 2, MID3, 1e-5),
             (3, 1, MID3, 1e-3), (3, 1, MID3, 1e-4), (3, 1, MID3, 1e-5),
             (3, 1, BIG[3], 1e-4), (3, 1, BIG[3], 1e-5),
             (3, 2, BIG[3], 1e-4)]
    for dim, typ, N, eps in cases:
        cls = Type1PlanND if typ == 1 else Type2PlanND
        p = cls(x[dim], N, eps=eps)
        exp = rule(dim, typ, eps, N)
        w_exp = kernel_params(eps, exp, dim)[0]
        check(f"{cls.__name__} {dim}D N={N} eps={eps:.0e}: upsampfac {exp}",
              p.upsampfac == exp and isinstance(p.upsampfac, float)
              and p.sigma == exp and p.w == w_exp,
              f"upsampfac={p.upsampfac} w={p.w} n_up={list(p.n_up)}")


def check_explicit_and_env():
    print("== (b) explicit values, sentinels and MLX_NUFFT_UPSAMPFAC ==")
    x3, x1 = pts(3, P_PLAN), pts(1, P_PLAN)
    N = MID3
    for cls, eps in ((Type2PlanND, 1e-3), (Type2PlanND, 1e-5),
                     (Type1PlanND, 1e-3), (Type1PlanND, 1e-5)):
        auto = rule(3, cls._NUFFT_TYPE, eps, N)
        for val in (2.0, 1.25):
            p = cls(x3, N, eps=eps, upsampfac=val)
            check(f"{cls.__name__} 3D N={N} eps={eps:.0e} upsampfac={val} "
                  f"(auto {auto})",
                  p.upsampfac == val and p.w == kernel_params(eps, val, 3)[0],
                  f"upsampfac={p.upsampfac} w={p.w}")
    p = Type2PlanND(x3, SMALL[3], eps=1e-3, upsampfac=1.25)
    check(f"Type2PlanND 3D N={SMALL[3]} eps=1e-03 upsampfac=1.25 (auto "
          f"{rule(3, 2, 1e-3, SMALL[3])} under the mode floor)",
          p.upsampfac == 1.25 and rule(3, 2, 1e-3, SMALL[3]) == 2.0,
          f"upsampfac={p.upsampfac} w={p.w}")
    p = Type1PlanND(x1, SMALL[1], eps=1e-3, upsampfac=1.25)
    check("Type1PlanND 1D eps=1e-03 upsampfac=1.25 (auto 2.0)",
          p.upsampfac == 1.25, f"upsampfac={p.upsampfac} w={p.w}")
    for sent in (None, "auto", 0, 0.0):
        p = Type2PlanND(x3, N, eps=1e-3, upsampfac=sent)
        check(f"Type2PlanND 3D eps=1e-03 upsampfac={sent!r}: the policy",
              p.upsampfac == rule(3, 2, 1e-3, N), f"upsampfac={p.upsampfac}")

    for typ in (1, 2):
        for val in (None, 2.0, 1.25, 0, "auto"):
            kw = {} if val is None else {"upsampfac": val}
            plan = gpu.Plan(typ, N, eps=1e-3, **kw)
            plan.setpts(*x3)
            exp = rule(3, typ, 1e-3, N) if val in (None, 0, "auto") else val
            opt = f", upsampfac={val!r}" if kw else ""
            check(f"Plan({typ}, {N}, eps=1e-03{opt}).setpts: {exp}",
                  plan._plan.upsampfac == exp,
                  f"upsampfac={plan._plan.upsampfac}")
    plan = gpu.Plan(1, N, eps=1e-3, dtype="complex64")
    plan.setpts(*x3)
    plan.execute_adjoint(cplx(N))
    check("Plan(1).execute_adjoint builds its type-2 plan on the policy",
          plan._adjoint.upsampfac == rule(3, 2, 1e-3, N),
          f"upsampfac={plan._adjoint.upsampfac}")
    plan = gpu.Plan(2, N, eps=1e-3, dtype="complex64", upsampfac=2.0)
    plan.setpts(*x3)
    plan.execute_adjoint(cplx(P_PLAN))
    check("Plan(2, upsampfac=2.0).execute_adjoint keeps the explicit value",
          plan._adjoint.upsampfac == 2.0,
          f"upsampfac={plan._adjoint.upsampfac}")

    with env_upsampfac("2.0"):
        p = Type2PlanND(x3, N, eps=1e-3)
        check("env=2.0: Type2PlanND 3D eps=1e-03 (auto 1.25)",
              p.upsampfac == 2.0, f"upsampfac={p.upsampfac} w={p.w}")
        q = Type2PlanND(x3, N, eps=1e-3, upsampfac=1.25)
        check("env=2.0 vs explicit 1.25: explicit wins", q.upsampfac == 1.25,
              f"upsampfac={q.upsampfac}")
        plan = gpu.Plan(2, N, eps=1e-3)
        plan.setpts(*x3)
        check("env=2.0 through Plan(2).setpts", plan._plan.upsampfac == 2.0,
              f"upsampfac={plan._plan.upsampfac}")
    with env_upsampfac("1.25"):
        p = Type1PlanND(x3, N, eps=1e-5)
        check("env=1.25: Type1PlanND 3D eps=1e-05 (auto 2.0)",
              p.upsampfac == 1.25 and p.w == kernel_params(1e-5, 1.25, 3)[0],
              f"upsampfac={p.upsampfac} w={p.w}")
        p = Type1PlanND(x3, SMALL[3], eps=1e-3)
        check("env=1.25: Type1PlanND 3D eps=1e-03 under the mode floor "
              "(auto 2.0)", p.upsampfac == 1.25,
              f"upsampfac={p.upsampfac} w={p.w}")
        p = Type1PlanND(x1, SMALL[1], eps=1e-3)
        check("env=1.25: Type1PlanND 1D eps=1e-03 (auto 2.0)",
              p.upsampfac == 1.25, f"upsampfac={p.upsampfac} w={p.w}")
        q = Type1PlanND(x3, N, eps=1e-5, upsampfac=2.0)
        check("env=1.25 vs explicit 2.0: explicit wins", q.upsampfac == 2.0,
              f"upsampfac={q.upsampfac}")
    with env_upsampfac("auto"):
        p = Type2PlanND(x3, N, eps=1e-3)
        check("env=auto: the policy", p.upsampfac == rule(3, 2, 1e-3, N),
              f"upsampfac={p.upsampfac}")
    with env_upsampfac("2"):
        p = Type2PlanND(x3, N, eps=1e-3)
        check("env=2 parses as 2.0", p.upsampfac == 2.0,
              f"upsampfac={p.upsampfac}")
    with env_upsampfac("fast"):
        try:
            Type2PlanND(x3, N, eps=1e-3)
            raised = None
        except ValueError as e:
            raised = str(e)
        check("env=fast raises ValueError naming the variable",
              raised is not None and nd.UPSAMP_ENV in raised, f"{raised}")
    check("env unset after the blocks", nd.UPSAMP_ENV not in os.environ, "")


def check_bits():
    print("== (c) auto plan output vs explicit plans ==")
    x = pts(3, P_BITS)
    N = MID3
    c, fk = cplx(P_BITS), cplx(N)
    for eps in (1e-3, 1e-5):
        for typ, cls, inp in ((2, Type2PlanND, fk), (1, Type1PlanND, c)):
            s = rule(3, typ, eps, N)
            auto = cls(x, N, eps=eps)
            same = cls(x, N, eps=eps, upsampfac=s)
            oth = cls(x, N, eps=eps, upsampfac=other(s))
            a = np.asarray(auto.execute(inp))
            e = np.asarray(same.execute(inp))
            e2 = np.asarray(same.execute(inp))
            o = np.asarray(oth.execute(inp))
            geom = (auto.upsampfac == s and auto.w == same.w
                    and list(auto.n_up) == list(same.n_up))
            label = (f"t{typ} 3D N={N} P={P_BITS} eps={eps:.0e}: auto (sigma "
                     f"{auto.upsampfac}) vs explicit sigma {s}")
            if typ == 2 or np.array_equal(e, e2):
                check(f"{label}: bit-identical", geom and np.array_equal(a, e),
                      f"w={auto.w} n_up={list(auto.n_up)} "
                      f"max|diff|={np.abs(a - e).max():.1e}")
            else:
                err = rel_l2(a, e)
                check(f"{label}: within atomics noise (explicit runs differ "
                      f"by {rel_l2(e2, e):.1e})",
                      geom and err <= T1_NOISE_GATE,
                      f"rel_l2={err:.1e} (gate {T1_NOISE_GATE:.0e})")
            d = rel_l2(a, o)
            check(f"{label}: differs from sigma {other(s)}",
                  d > T1_NOISE_GATE, f"rel_l2 vs sigma {other(s)} = {d:.1e}")

    eps = 1e-3
    s = rule(3, 2, eps, N)
    e = np.asarray(Type2PlanND(x, N, eps=eps, isign=-1,
                               upsampfac=s).execute(fk))
    fn = gpu.nufft3d2(*x, fk, eps=eps)
    plan = gpu.Plan(2, N, eps=eps, dtype="complex64")
    plan.setpts(*x)
    pl = plan.execute(fk)
    check(f"nufft3d2 eps={eps:.0e} (auto) == explicit sigma {s} plan bits",
          np.array_equal(fn, e), f"max|diff|={np.abs(fn - e).max():.1e}")
    check(f"Plan(2).execute eps={eps:.0e} (auto) == explicit sigma {s} "
          "plan bits", np.array_equal(pl, e),
          f"max|diff|={np.abs(pl - e).max():.1e}")
    o = np.asarray(Type2PlanND(x, N, eps=eps, isign=-1,
                               upsampfac=other(s)).execute(fk))
    fn2 = gpu.nufft3d2(*x, fk, eps=eps, upsampfac=other(s))
    check(f"nufft3d2 eps={eps:.0e} upsampfac={other(s)} == explicit sigma "
          f"{other(s)} plan bits", np.array_equal(fn2, o),
          f"max|diff|={np.abs(fn2 - o).max():.1e}")


def check_accuracy():
    print("== (d) 3D auto accuracy vs CPU FINUFFT fp64 (gate 4*eps + 2e-4) ==")
    N = (ACC_N,) * 3
    x = pts(3, ACC_M)
    c = rng.standard_normal(ACC_M) + 1j * rng.standard_normal(ACC_M)
    fk = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    ref1 = cpu.nufft3d1(*x, c, N, eps=1e-9, isign=+1)
    ref2 = cpu.nufft3d2(*x, fk, eps=1e-9, isign=-1)
    for eps in ACC_EPS:
        thr = 4 * eps + 2e-4
        for typ, cls, inp, ref in ((1, Type1PlanND, c, ref1),
                                   (2, Type2PlanND, fk, ref2)):
            auto = cls(x, N, eps=eps)
            err = rel_l2(np.asarray(auto.execute(inp.astype(np.complex64))),
                         ref)
            detail = (f"rel_l2={err:.2e} (gate {thr:.1e}), w={auto.w} "
                      f"n_up={list(auto.n_up)}")
            if auto.upsampfac != nd.UPSAMP_AUTO_HIGH:
                ctrl = cls(x, N, eps=eps, upsampfac=nd.UPSAMP_AUTO_HIGH)
                e2 = rel_l2(np.asarray(ctrl.execute(
                    inp.astype(np.complex64))), ref)
                detail += f"; sigma=2 {e2:.2e} (ratio {err / e2:.2f})"
            check(f"t{typ} 3D {ACC_N}^3 M={ACC_M:.0e} eps={eps:.0e} auto "
                  f"sigma={auto.upsampfac}", err <= thr, detail)
        e1 = rel_l2(gpu.nufft3d1(*x, c, N, eps=eps), ref1)
        e2 = rel_l2(gpu.nufft3d2(*x, fk, eps=eps), ref2)
        check(f"nufft3d1 / nufft3d2 {ACC_N}^3 M={ACC_M:.0e} eps={eps:.0e} "
              "(auto) vs cpu", e1 <= thr and e2 <= thr,
              f"rel_l2 t1={e1:.2e} t2={e2:.2e} (gate {thr:.1e})")


def check_density():
    print("== (e) point density limits the sigma=1.25 default ==")
    check("shipped density constants",
          (nd.UPSAMP_AUTO_T1_MAX_RHO, nd.UPSAMP_AUTO_EPS_T2_ANY_RHO,
           nd.UPSAMP_AUTO_T2_MAX_RHO) == (0.1, 1e-3, 0.3),
          "T1_MAX_RHO 0.1, EPS_T2_ANY_RHO 1e-3, T2_MAX_RHO 0.3")
    bad = []
    cells = 8 * int(np.prod(MID3))
    for N in (MID3, BIG[3]):
        cells = 8 * int(np.prod(N))
        for rho in (0.0, 0.05, 0.099, 0.1, 0.2, 0.299, 0.3, 1.0):
            P = int(round(rho * cells)) if rho else None
            for typ in (1, 2):
                for e in (1e-2, 1e-3, 1e-4, 1e-5):
                    got = nd._resolve_upsampfac(None, 3, typ, e, N, P)[0]
                    if got != rule(3, typ, e, N, P):
                        bad.append((N, rho, typ, e, got))
    check("resolver matches the density rule (2 grids x 8 densities x 2 types x 4 eps)",
          not bad, f"mismatches {bad[:4]}")
    cells = 8 * int(np.prod(MID3))
    P_hi = int(0.36 * cells)
    x = pts(3, P_hi)
    for typ, e, want in ((1, 1e-3, nd.UPSAMP_AUTO_HIGH), (2, 1e-3, nd.UPSAMP_AUTO_LOW),
                         (2, 1e-4, nd.UPSAMP_AUTO_HIGH)):
        cls = nd.Type1PlanND if typ == 1 else nd.Type2PlanND
        p = cls(x, MID3, eps=e, isign=+1)
        check(f"plan t{typ} 3D {MID3} rho=0.36 eps={e:.0e} resolves {want}",
              p.upsampfac == want, f"upsampfac={p.upsampfac}")


if __name__ == "__main__":
    preset = os.environ.pop(nd.UPSAMP_ENV, None)
    if preset is not None:
        print(f"NOTE {nd.UPSAMP_ENV}={preset!r} was set; unset for this test "
              "(the policy itself is under test)")
    check_table()
    check_explicit_and_env()
    check_bits()
    check_accuracy()
    check_density()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
