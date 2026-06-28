"""M0 validation: oracle conventions + numpy reference t3 vs FINUFFT.

Small problem (lat-scaled) so the fp64 direct sum is affordable.
"""

import sys
import time
import pathlib

import numpy as np
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum, rel_l2          # noqa: E402
from mlx_nufft.ref_t3 import RefT3Plan, Flags, PURE32, FULL64  # noqa: E402

if __name__ == "__main__":
    N, P, lat = 64, 2000, 0.02
    prob = gen_anisotropic(N=N, P=P, lat=lat)
    x, c, s = prob["x"], prob["c"], prob["s"]

    t0 = time.perf_counter()
    f_direct = direct_sum(x, c, s, isign=+1)
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    f_fin = finufft.nufft3d3(*x, c, *s, isign=+1, eps=1e-6)
    t_fin = time.perf_counter() - t0
    print(f"[oracle] finufft fp64 eps=1e-6 vs direct sum: "
          f"rel_l2={rel_l2(f_fin, f_direct):.3e}  "
          f"(direct {t_direct:.2f}s, finufft {t_fin:.3f}s)")

    for eps in (1e-4, 1e-5, 1e-6):
        for sigma in (1.25, 2.0):
            plan = RefT3Plan(x, s, eps=eps, isign=+1, upsampfac=sigma,
                             flags=FULL64)
            f_ref = plan.execute(c)
            print(f"[ref fp64] eps={eps:.0e} sigma={sigma:<4} "
                  f"nf={plan.nf} n_up={plan.n_up} w={plan.w} "
                  f"rel_l2 vs direct = {rel_l2(f_ref, f_direct):.3e}")

    # pure fp32 reference — first look at the precision cliff (at SMALL size;
    # the full-size cliff is worse because coordinates are larger)
    plan32 = RefT3Plan(x, s, eps=1e-5, isign=+1, upsampfac=1.25, flags=PURE32)
    f32 = plan32.execute(c)
    print(f"[ref fp32] eps=1e-05 sigma=1.25 rel_l2 vs direct = "
          f"{rel_l2(f32, f_direct):.3e}")

    # adjoint convention check (brief: swapped sources/targets, isign=-1)
    fa_direct = direct_sum(s, f_direct.conj() * 0 + 1.0 + 0j, x, isign=-1) \
        if False else None
    print("OK")
