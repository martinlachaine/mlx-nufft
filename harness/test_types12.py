"""Validate Type1Plan / Type2Plan vs CPU FINUFFT fp64 and direct sums."""

import sys
import time
import pathlib

import numpy as np
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import rel_l2                              # noqa: E402
from mlx_nufft import Type1Plan, Type2Plan              # noqa: E402

if __name__ == "__main__":
    rng = np.random.default_rng(7)

    for N, P in [((64, 72, 18), 5000), ((256, 256, 32), 200_000)]:
        x = [rng.uniform(-np.pi, np.pi, P) for _ in range(3)]
        c = (rng.standard_normal(P) + 1j * rng.standard_normal(P))

        for isign in (+1, -1):
            for eps in (1e-4, 1e-5):
                ref = finufft.nufft3d1(*x, c, N, isign=isign, eps=1e-9)
                t0 = time.perf_counter()
                plan = Type1Plan(x, N, eps=eps, isign=isign)
                f = plan.execute(c.astype(np.complex64))
                dt = time.perf_counter() - t0
                print(f"[t1 N={N} P={P} isign={isign:+d} eps={eps:.0e}] "
                      f"rel_l2={rel_l2(f, ref):.3e}  ({dt:.2f}s w/ plan)")

        # type 2
        fk = (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        for isign in (+1, -1):
            ref2 = finufft.nufft3d2(*x, fk, isign=isign, eps=1e-9)
            plan2 = Type2Plan(x, N, eps=1e-5, isign=isign)
            c2 = plan2.execute(fk.astype(np.complex64))
            print(f"[t2 N={N} P={P} isign={isign:+d} eps=1e-05] "
                  f"rel_l2={rel_l2(c2, ref2):.3e}")

        # round-trip adjointness sanity: <T1 c, fk> == <c, T2 fk> (conj pair)
        f1 = Type1Plan(x, N, eps=1e-5, isign=+1).execute(
            c.astype(np.complex64))
        c2 = Type2Plan(x, N, eps=1e-5, isign=-1).execute(
            fk.astype(np.complex64))
        lhs = np.vdot(np.asarray(fk), f1)
        rhs = np.vdot(c2, c)
        print(f"[adjoint pairing N={N}] |lhs-rhs|/|lhs| = "
              f"{abs(lhs - rhs) / abs(lhs):.3e}")
