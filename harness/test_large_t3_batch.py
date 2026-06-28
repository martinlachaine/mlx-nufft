"""Acceptance: large type-3 grids and execute_batch.

These exercise one root constraint: mx.fast.metal_kernel buffers are
int32-indexed (max 2**31-1 elements). The slab z-major grid (single execute)
and the fused batch buffers (execute_batch, x nch) cross it at large lateral
grids; the pipeline z-chunks / complex64-counts to stay under the cap.

Gate: rel-L2 <= 1e-4 vs an exact fp64 direct-sum oracle on a target subset
(the acceptance gate). Run on an M-series Max with ample RAM.
"""

import sys
import pathlib
import platform
import subprocess

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum_mp, rel_l2          # noqa: E402
import mlx_nufft.gpu_t3 as g                                    # noqa: E402

GATE = 1e-4
FAILS = []


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, {platform.platform()}"


def check(label, err, gate=GATE):
    ok = err <= gate
    print(f"  {'PASS' if ok else 'FAIL'} {label}: rel_l2={err:.3e} "
          f"(gate {gate:.0e})")
    if not ok:
        FAILS.append(label)


if __name__ == "__main__":
    print(f"machine: {machine()}")
    rng = np.random.default_rng(5)

    # ---- Test 1: large-grid type-3 single execute vs oracle ----
    # lat=2.0 -> n_up=[7200,7200,24]: padz z-major grid = 2.49e9 > 2^31
    # (handled by the z-chunked padz).
    print("\n[1] large-grid type-3 (n_up ~ 7200) single execute vs oracle")
    prob = gen_anisotropic(N=1024, P=40_000, lat=2.0)
    x, c, s = prob["x"], prob["c"], prob["s"]
    plan = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64")
    print(f"    n_up={plan.n_up}  padz_elems={plan.n_up[0]*plan.n_up[1]*plan.n_up[2]*2:.3e}"
          f" ({plan.n_up[0]*plan.n_up[1]*plan.n_up[2]*2/2**31:.2f}x 2^31)")
    f = np.asarray(plan.execute(c))
    idx = rng.choice(s[0].size, 12_000, replace=False)
    fd = direct_sum_mp(x, c, s, isign=+1, idx=idx)
    check(f"type-3 n_up={plan.n_up} vs fp64 oracle", rel_l2(f[idx], fd))
    del plan
    import mlx.core as mx
    mx.clear_cache()

    # ---- Test 1b: very large grid — n_up beyond the spread ceiling (~9657).
    # lat=3.0 -> n_up=[10800,...]; only runs because the spread grid is
    # complex64-counted (an nf-grid of 1.5e9 floats would be 1.4x 2^31 as
    # float32). Correctness, not just non-crash.
    print("\n[1b] very large grid (n_up ~ 10800)")
    prob = gen_anisotropic(N=1024, P=40_000, lat=3.0)
    x, c, s = prob["x"], prob["c"], prob["s"]
    plan = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64")
    nf = plan.nf
    print(f"    n_up={plan.n_up}  nf-grid float32 would be {nf[0]*nf[1]*nf[2]*2/2**31:.2f}x 2^31"
          f" (complex64 {nf[0]*nf[1]*nf[2]/2**31:.2f}x)")
    f = np.asarray(plan.execute(c))
    idx = rng.choice(s[0].size, 10_000, replace=False)
    fd = direct_sum_mp(x, c, s, isign=+1, idx=idx)
    check(f"type-3 n_up={plan.n_up} vs fp64 oracle", rel_l2(f[idx], fd))
    del plan
    mx.clear_cache()

    # ---- Test 2: execute_batch (several transforms, one plan) vs oracle ----
    # exercises BOTH the small fused path and the large fallback path.
    for tag, lat, P in [("small fused", 0.75, 100_000),
                        ("large fallback", 2.0, 40_000)]:
        print(f"\n[2:{tag}] execute_batch (nch=3) vs per-transform oracle")
        prob = gen_anisotropic(N=1024, P=P, lat=lat)
        x, c, s = prob["x"], prob["c"], prob["s"]
        g._SLAB_THRESHOLD = min(g._SLAB_THRESHOLD, 1.0e9)   # force slab path
        plan = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64")
        cs = np.stack([c,
                       c * (0.5 + 0.5j),
                       (rng.standard_normal(c.size)
                        + 1j * rng.standard_normal(c.size))]).astype(np.complex64)
        fb = np.asarray(plan.execute_batch(cs))
        assert fb.shape == (3, s[0].size), fb.shape
        idx = rng.choice(s[0].size, 8_000, replace=False)
        for ch in range(3):
            fd = direct_sum_mp(x, cs[ch], s, isign=+1, idx=idx)
            check(f"{tag} ch{ch} batch vs oracle",
                  rel_l2(np.asarray(fb[ch])[idx], fd))
        del plan
        mx.clear_cache()

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
