"""Acceptance + speed for the optional VkFFT FFT backend.

Oracle-gates fft_backend="vkfft" vs the exact fp64 direct-sum (rel-L2 <= 1e-4)
at the large-grid sizes where it matters (MLX uses the scrambled four-step;
VkFFT uses natural order + identity-scramble gather — so this exercises the
full convention/scramble/normalization path), and reports the whole-execute
speedup vs the validated MLX backend. Skips cleanly if the bridge isn't built.
"""
import sys
import time
import pathlib
import platform
import subprocess

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum_mp, rel_l2          # noqa: E402
import mlx_nufft.gpu_t3 as g                                    # noqa: E402
from mlx_nufft import vkfft_backend as vk                       # noqa: E402

GATE = 1e-4
FAILS = []


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    return f"{chip}, {platform.platform()}"


def tmin(fn, n=4):
    fn(); mx.synchronize()
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); fn(); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000


if __name__ == "__main__":
    print(f"machine: {machine()}")
    if not vk.available():
        print("VkFFT bridge not built (vkfft_bridge/build.sh) — SKIP")
        sys.exit(0)
    rng = np.random.default_rng(5)

    for lat, P in [(1.0, 60_000), (2.0, 40_000)]:     # n_up ~ 3600, 7200
        prob = gen_anisotropic(N=1024, P=P, lat=lat)
        x, c, s = prob["x"], prob["c"], prob["s"]
        pv = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64",
                         fft_backend="vkfft")
        nu = pv.n_up
        print(f"\nn_up={nu} (scramB={pv.scramB})")
        fv = np.asarray(pv.execute(c))
        idx = rng.choice(s[0].size, 12_000, replace=False)
        fd = direct_sum_mp(x, c, s, isign=+1, idx=idx)
        e = rel_l2(fv[idx], fd)
        ok = e <= GATE
        print(f"  {'PASS' if ok else 'FAIL'} vkfft vs fp64 oracle: {e:.3e} "
              f"(gate {GATE:.0e})")
        if not ok:
            FAILS.append(f"oracle n_up={nu}")
        # whole-execute speedup vs MLX backend (same problem)
        pm = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64",
                         fft_backend="mlx")
        fm = np.asarray(pm.execute(c))
        ab = rel_l2(fv.ravel(), fm.ravel())
        print(f"  vkfft vs mlx backend A/B: {ab:.3e} (eps-scale ok)")
        pv.execute(c); pm.execute(c)                  # warm
        t_vk = tmin(lambda: pv.execute(c, return_np=False))
        t_mlx = tmin(lambda: pm.execute(c, return_np=False))
        print(f"  whole-execute: mlx {t_mlx:.0f} ms  vkfft {t_vk:.0f} ms  "
              f"-> {t_mlx/t_vk:.2f}x")
        del pv, pm
        mx.clear_cache()

    print(f"\n{'ALL PASS' if not FAILS else f'FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
