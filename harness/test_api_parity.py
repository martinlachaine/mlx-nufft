"""finufft-API parity: mlx-nufft vs CPU FINUFFT across the full matrix.

Covers dims 1/2/3 x types 1/2/3 x isign +/-, odd and even mode counts,
n_trans > 1 via the Plan interface, complex64/complex128 dtype round-trip,
out= filling, and the eps=1e-3 bring-up tolerance alongside eps=1e-5.

Thresholds: comparing two independent approximations (our fp32 pipeline vs
CPU FINUFFT fp64 at the same eps), the difference is bounded by the sum of
both truncation errors plus the fp32 floor: thr = 4*eps + 2e-4.
"""

import sys
import pathlib

import numpy as np
import finufft as cpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
import mlx_nufft as gpu                                  # noqa: E402
from harness.gen import rel_l2                               # noqa: E402

rng = np.random.default_rng(7)
M = 4000
FAILS = []


def check(label, got, ref, thr):
    err = rel_l2(np.asarray(got).ravel(), np.asarray(ref).ravel())
    ok = err <= thr
    print(f"  {'PASS' if ok else 'FAIL'} {label}: rel_l2={err:.3e} "
          f"(thr {thr:.1e})")
    if not ok:
        FAILS.append(label)


def pts(dim, n=M):
    return [rng.uniform(-np.pi, np.pi, n) for _ in range(dim)]


if __name__ == "__main__":
    c = (rng.standard_normal(M) + 1j * rng.standard_normal(M))

    for eps in (1e-3, 1e-5):
        thr = 4 * eps + 2e-4
        print(f"== eps={eps:g} ==")
        for dim, N in [(1, (90,)), (1, (87,)),
                       (2, (48, 36)), (2, (45, 34)),
                       (3, (24, 20, 16)), (3, (21, 18, 14))]:
            x = pts(dim)
            for isign in (+1, -1):
                tag = f"{dim}d N={N} isign={isign:+d}"
                # ---- type 1 ----
                f1g = getattr(gpu, f"nufft{dim}d1")(*x, c, N, eps=eps,
                                                    isign=isign)
                f1c = getattr(cpu, f"nufft{dim}d1")(
                    *[v.astype(np.float64) for v in x], c.astype(np.complex128),
                    N, eps=eps, isign=isign)
                check(f"t1 {tag}", f1g, f1c, thr)
                # ---- type 2 ----
                fk = (rng.standard_normal(N) + 1j * rng.standard_normal(N))
                c2g = getattr(gpu, f"nufft{dim}d2")(*x, fk, eps=eps,
                                                    isign=isign)
                c2c = getattr(cpu, f"nufft{dim}d2")(
                    *[v.astype(np.float64) for v in x],
                    fk.astype(np.complex128), eps=eps, isign=isign)
                check(f"t2 {tag}", c2g, c2c, thr)
                # ---- type 3 ----
                s = [rng.uniform(-30, 30, 1500) for _ in range(dim)]
                f3g = getattr(gpu, f"nufft{dim}d3")(*x, c, *s, eps=eps,
                                                    isign=isign)
                f3c = getattr(cpu, f"nufft{dim}d3")(
                    *[v.astype(np.float64) for v in x], c.astype(np.complex128),
                    *[v.astype(np.float64) for v in s], eps=eps, isign=isign)
                check(f"t3 {tag}", f3g, f3c, thr)

    print("== Plan interface, n_trans=3, complex64, out= ==")
    eps, thr = 1e-5, 4e-4
    N = (40, 36)
    x = pts(2)
    cs = (rng.standard_normal((3, M)) + 1j * rng.standard_normal((3, M))
          ).astype(np.complex64)
    p = gpu.Plan(1, N, n_trans=3, eps=eps, isign=+1, dtype="complex64")
    p.setpts(*x)
    out = np.empty((3,) + N, dtype=np.complex64)
    got = p.execute(cs, out=out)
    assert got is out and out.dtype == np.complex64
    pc = cpu.Plan(1, N, n_trans=3, eps=eps, isign=+1, dtype="complex64")
    pc.setpts(*[v.astype(np.float32) for v in x])
    ref = pc.execute(cs)
    check("Plan t1 2d n_trans=3 c64 out=", out, ref, thr)

    p2 = gpu.Plan(2, N, n_trans=3, eps=eps, isign=-1, dtype="complex64")
    p2.setpts(*x)
    fks = (rng.standard_normal((3,) + N)
           + 1j * rng.standard_normal((3,) + N)).astype(np.complex64)
    got2 = p2.execute(fks)
    pc2 = cpu.Plan(2, N, n_trans=3, eps=eps, isign=-1, dtype="complex64")
    pc2.setpts(*[v.astype(np.float32) for v in x])
    ref2 = pc2.execute(fks)
    check("Plan t2 2d n_trans=3 c64", got2, ref2, thr)

    p3 = gpu.Plan(3, 2, n_trans=1, eps=eps, isign=+1)
    s = [rng.uniform(-30, 30, 1500) for _ in range(2)]
    p3.setpts(x[0], x[1], s=s[0], t=s[1])
    got3 = p3.execute(c)
    ref3 = cpu.nufft2d3(*[v.astype(np.float64) for v in x],
                        c.astype(np.complex128),
                        *[v.astype(np.float64) for v in s], eps=eps, isign=+1)
    check("Plan t3 2d", got3, ref3, thr)

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
