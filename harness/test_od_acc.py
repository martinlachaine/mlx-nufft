"""Gate the OD spread accumulation variants (nd.OD_ACC).

(a) The default type-1 plan in dims 2 and 3 accumulates its tile in int32
    fixed point ("fix"); dim 1 keeps the float CAS loop ("cas").
(b) At eps 1e-5 the fixed-point result is as accurate as the float one
    against double-precision CPU FINUFFT (within 3 percent), on random and
    clustered points, and at eps 1e-6 within 25 percent (the quantization
    noise of about 1e-6).
(c) The fixed-point and float results agree with each other to 3e-6.
(d) Strengths with a wide dynamic range (one point 1e4 times the others)
    stay within the eps 1e-5 gate.

Prints PASS/FAIL per check; exits non-zero on any failure.

    .venv/bin/python harness/test_od_acc.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import finufft                                   # noqa: E402
import mlx_nufft.nd as nd                        # noqa: E402
from mlx_nufft import Type1PlanND                # noqa: E402

FAILS = []
CPU = {2: finufft.nufft2d1, 3: finufft.nufft3d1}
SIZES = {2: 256, 3: 48}


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def plan(x, N, eps, acc):
    saved = nd.OD_ACC
    nd.OD_ACC = acc
    try:
        return Type1PlanND(x, N, eps=eps, isign=1, upsampfac=2.0)
    finally:
        nd.OD_ACC = saved


def main():
    rng = np.random.default_rng(11)
    print("== (a) default accumulation ==")
    for dim in (1, 2, 3):
        N = (SIZES.get(dim, 2 ** 16),) * dim
        x = tuple(rng.uniform(-np.pi, np.pi, 200_000) for _ in range(dim))
        p = Type1PlanND(x, N, eps=1e-5, isign=1, upsampfac=2.0)
        want = "fix" if dim > 1 else "cas"
        check(f"{dim}D default accumulation is {want}", p._od_ex or getattr(p, "_od_acc", None) == want,
              f"got {'od_ex' if p._od_ex else getattr(p, '_od_acc', None)}")
    print("== (b, c) accuracy against fp64 FINUFFT ==")
    for dim in (2, 3):
        N = (SIZES[dim],) * dim
        M = 8 * int(np.prod(N))                  # density 1 on the sigma-2 grid
        for dist in ("rand", "cluster"):
            x = tuple(rng.uniform(-np.pi, np.pi, M) if dist == "rand"
                      else rng.uniform(-0.3, 0.3, M) for _ in range(dim))
            c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)
            ref = CPU[dim](*x, c.astype(np.complex128), N, eps=1e-9, isign=1)
            for eps, tol in ((1e-5, 1.03), (1e-6, 1.25)):
                out = {acc: plan(x, N, eps, acc).execute(c) for acc in ("cas", "fix")}
                e = {acc: rel(out[acc], ref) for acc in out}
                check(f"{dim}D {dist} eps={eps:.0e}: fix error within {tol}x of cas",
                      e["fix"] <= tol * e["cas"], f"fix {e['fix']:.2e} cas {e['cas']:.2e}")
                d = rel(out["fix"], out["cas"])
                check(f"{dim}D {dist} eps={eps:.0e}: fix vs cas", d <= 3e-6, f"{d:.1e}")
    print("== (d) wide dynamic range of strengths ==")
    N = (SIZES[3],) * 3
    M = 8 * int(np.prod(N))
    x = tuple(rng.uniform(-np.pi, np.pi, M) for _ in range(3))
    c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)
    c[::50_000] *= 1e4
    ref = CPU[3](*x, c.astype(np.complex128), N, eps=1e-9, isign=1)
    e = {acc: rel(plan(x, N, 1e-5, acc).execute(c), ref) for acc in ("cas", "fix")}
    check("3D eps=1e-5 strengths spanning 1e4: fix within 1.03x of cas",
          e["fix"] <= 1.03 * e["cas"], f"fix {e['fix']:.2e} cas {e['cas']:.2e}")
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
