"""M1 correctness: GPU t3 vs direct sum / FINUFFT on small problems."""

import sys
import pathlib

import numpy as np
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum, rel_l2   # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan                  # noqa: E402

if __name__ == "__main__":
    N, P, lat = 64, 2000, 0.02
    prob = gen_anisotropic(N=N, P=P, lat=lat)
    x, c, s = prob["x"], prob["c"], prob["s"]
    f_direct = direct_sum(x, c, s, isign=+1)

    for prec in ("crit64", "fp32"):
        for eps in (1e-4, 1e-5, 1e-6):
            plan = GpuT3Plan(x, s, eps=eps, isign=+1, prec=prec)
            f = plan.execute(c)
            print(f"[gpu {prec:>6}] eps={eps:.0e} nf={plan.nf} "
                  f"n_up={plan.n_up} w={plan.w} "
                  f"rel_l2 vs direct = {rel_l2(f, f_direct):.3e}")

    # adjoint convention (swap sources/targets, isign=-1)
    fa_ref = direct_sum(s, np.ones(s[0].size) + 0j, x, isign=-1)
    plan_a = GpuT3Plan(s, x, eps=1e-5, isign=-1, prec="crit64")
    fa = plan_a.execute(np.ones(s[0].size, dtype=np.complex64))
    print(f"[gpu adjoint crit64] eps=1e-05 rel_l2 vs direct = "
          f"{rel_l2(fa, fa_ref):.3e}")

    f_fin = finufft.nufft3d3(*x, c, *s, isign=+1, eps=1e-6)
    plan = GpuT3Plan(x, s, eps=1e-6, isign=+1, prec="crit64")
    f = plan.execute(c)
    print(f"[gpu vs finufft eps=1e-6] rel_l2 = {rel_l2(f, f_fin):.3e}")
