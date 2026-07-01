"""Acceptance + speed for the optional VkFFT FFT backend.

Oracle-gates fft_backend="vkfft" vs the exact fp64 direct-sum (rel-L2 <= 1e-4)
at the large-grid sizes where it matters (MLX uses the scrambled four-step;
VkFFT uses natural order + identity-scramble gather — so this exercises the
full convention/scramble/normalization path), and reports the whole-execute
speedup vs the validated MLX backend. Also covers the extended bridge: raw
1D/2D/3D fftn vs numpy, the non-slab type-3 3D FFT (oracle-gated end to end,
FFT stage A/B-gated vs the mlx fft_axis path at 1e-6), and Type1PlanND /
Type2PlanND fft_backend="vkfft" vs "mlx" (A/B <= 1e-6; the mode deconvolution
sits after the FFT so backend agreement holds at the fp32 floor there — in
type 3 the deconvolution scales the grid BEFORE the FFT, so end-to-end A/B
between two fp32 FFTs is eps-scale and the fp64 oracle is the arbiter).
Skips cleanly if the bridge isn't built.
"""
import sys
import time
import pathlib
import platform
import subprocess

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import (gen_anisotropic, gen_generic, direct_sum_mp,
                         rel_l2, uniform_points)                 # noqa: E402
import mlx_nufft.gpu_t3 as g                                    # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND               # noqa: E402
from mlx_nufft import vkfft_backend as vk                       # noqa: E402

GATE = 1e-4          # vs the exact fp64 oracle (fp32 pipeline, eps=1e-5)
GATE_AB = 1e-6       # vkfft vs mlx where only the FFT differs (fp32 floor)
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


def check(ok, label, err, gate):
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {err:.3e} (gate {gate:.0e})")
    if not ok:
        FAILS.append(label)


def raw_fftn_checks(rng):
    """Bridge fftn vs numpy: fwd/inv, dims 1-3, non-page-multiple sizes."""
    print("\nraw fftn vs numpy")
    cases = [(60, 48, 40), (33, 22, 14), (720, 500), (60000,)]
    for shape in cases:
        a = (rng.standard_normal(shape)
             + 1j * rng.standard_normal(shape)).astype(np.complex64)
        m = mx.array(a); mx.eval(m)
        vk.fftn_inplace(m, -1, 0)
        e = rel_l2(np.array(m), np.fft.fftn(a.astype(np.complex128)))
        check(e <= GATE_AB, f"fftn fwd {shape}", e, GATE_AB)
        m = mx.array(a); mx.eval(m)
        vk.fftn_inplace(m, 1, 1)
        e = rel_l2(np.array(m), np.fft.ifftn(a.astype(np.complex128)))
        check(e <= GATE_AB, f"fftn inv {shape}", e, GATE_AB)


def t3_nonslab_checks(rng):
    """Non-slab type 3: whole-grid VkFFT 3D vs the mlx fft_axis loop (raw
    A/B at the FFT stage) + the full transform vs the fp64 oracle."""
    print("\ntype-3 non-slab (3D whole-grid FFT)")
    prob = gen_generic(N=256, P=50_000, X=10.0, S=10.0)
    x, c, s = prob["x"], prob["c"], prob["s"]
    # upsampfac=2.0: better-conditioned deconvolution keeps this instance
    # comfortably inside the fp32-pipeline gate (sigma=1.25 sits right at
    # 1e-4 for BOTH backends); the vkfft whole-grid path is identical.
    pv = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64",
                     upsampfac=2.0, fft_backend="vkfft")
    assert not pv.slab_mode and pv._vkfft_fft3
    print(f"  n_up={pv.n_up} (non-slab, vkfft_fft3={pv._vkfft_fft3})")
    # raw FFT-stage A/B on a random inner grid: identical input through the
    # bridge vs the plan's fft_axis path (only the FFT implementation differs)
    a = (rng.standard_normal(tuple(pv.n_up))
         + 1j * rng.standard_normal(tuple(pv.n_up))).astype(np.complex64)
    Hm = mx.array(a)
    for ax in (2, 1, 0):
        Hm = g.fft_axis(Hm, ax, inverse=True, twiddle_cache={})
    mx.eval(Hm)
    Hv = mx.array(a); mx.eval(Hv)
    vk.fftn_inplace(Hv, 1, 1)
    e = rel_l2(np.array(Hv), np.array(Hm))
    check(e <= GATE_AB, "fft stage vkfft vs mlx (n_up grid)", e, GATE_AB)
    # end to end vs the exact oracle (the deconvolution amplifies band-edge
    # modes before the FFT, so backend A/B is eps-scale by construction;
    # the fp64 direct sum is the arbiter, same gate as the slab path)
    fv = np.asarray(pv.execute(c))
    idx = rng.choice(s[0].size, 8_000, replace=False)
    fd = direct_sum_mp(x, c, s, isign=+1, idx=idx)
    e = rel_l2(fv[idx], fd)
    check(e <= GATE, "t3 non-slab vkfft vs fp64 oracle", e, GATE)
    pm = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64",
                     upsampfac=2.0, fft_backend="mlx")
    fm = np.asarray(pm.execute(c))
    em = rel_l2(fm[idx], fd)
    print(f"  info: mlx vs oracle {em:.3e}; vkfft vs mlx A/B "
          f"{rel_l2(fv, fm):.3e} (small at sigma=2.0; eps-scale at "
          f"sigma=1.25 where the pre-FFT deconvolution is steep)")
    del pv, pm
    mx.clear_cache()


def t12_checks(rng):
    """Type 1/2 ND with fft_backend="vkfft" vs "mlx" (deconvolution after
    the FFT -> backend agreement at the fp32 floor)."""
    print("\ntype-1/2 ND fft_backend A/B")
    for dim, N in [(3, 64), (2, 500), (1, 30000)]:
        xs, n_modes, M = uniform_points(dim, N, rho=0.5)
        cj = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        f_ab = []
        for backend in ("mlx", "vkfft"):
            p = Type1PlanND(xs, n_modes, eps=1e-6, isign=+1,
                            fft_backend=backend)
            f_ab.append(p.execute(cj))
        e = rel_l2(f_ab[1], f_ab[0])
        check(e <= GATE_AB, f"t1 {dim}D N={N} vkfft vs mlx", e, GATE_AB)
        fk = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)
        c_ab = []
        for backend in ("mlx", "vkfft"):
            p = Type2PlanND(xs, n_modes, eps=1e-6, isign=-1,
                            fft_backend=backend)
            c_ab.append(p.execute(fk))
        e = rel_l2(c_ab[1], c_ab[0])
        check(e <= GATE_AB, f"t2 {dim}D N={N} vkfft vs mlx", e, GATE_AB)
    mx.clear_cache()


if __name__ == "__main__":
    print(f"machine: {machine()}")
    if not vk.available():
        print("VkFFT bridge not built (vkfft_bridge/build.sh) — SKIP")
        sys.exit(0)
    rng = np.random.default_rng(5)

    raw_fftn_checks(rng)
    t3_nonslab_checks(rng)
    t12_checks(rng)

    print("\ntype-3 slab (batched 2D lateral FFT)")
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
        check(e <= GATE, f"slab vkfft vs fp64 oracle n_up={nu}", e, GATE)
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
