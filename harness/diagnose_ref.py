"""Precision diagnosis with the numpy reference: which stage costs accuracy?

Small problem (small coords) isolates the fp32 *pipeline* floor
(spread/fft/interp); the fp32 *coordinate/phase* cliff needs large coords
and is diagnosed on the GPU (see bench/diagnose_gpu).
Note: numpy has no complex64 FFT (it promotes to complex128), so fft64=False
in the reference still computes the FFT in fp64 — the FFT's own fp32 noise
is only visible in the GPU pipeline.
"""

import sys
import pathlib
import itertools

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum, rel_l2     # noqa: E402
from mlx_nufft.ref_t3 import RefT3Plan, Flags             # noqa: E402

if __name__ == "__main__":
    prob = gen_anisotropic(N=64, P=2000, lat=0.02)
    x, c, s = prob["x"], prob["c"], prob["s"]
    fd = direct_sum(x, c, s, isign=+1)

    names = ["coords64", "phases64", "spread64", "fft64", "interp64",
             "deconv64"]
    base = dict.fromkeys(names, True)

    print("eps=1e-5 sigma=1.25; flip each stage to fp32 alone "
          "(True=fp64); lat=0.02 (small coords)")
    f_all = RefT3Plan(x, s, eps=1e-5, flags=Flags(**base)).execute(c)
    print(f"  all fp64           : {rel_l2(f_all, fd):.3e}")
    for k in names:
        fl = dict(base)
        fl[k] = False
        f = RefT3Plan(x, s, eps=1e-5, flags=Flags(**fl)).execute(c)
        print(f"  only {k:9s} fp32: {rel_l2(f, fd):.3e}")
    f_none = RefT3Plan(x, s, eps=1e-5,
                       flags=Flags(**dict.fromkeys(names, False))).execute(c)
    print(f"  all fp32           : {rel_l2(f_none, fd):.3e}")
