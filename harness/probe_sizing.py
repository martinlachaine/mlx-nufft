"""Probe FINUFFT's internal type-3 grid sizes vs our sizing prediction.

Runs finufft with debug=1 (C++ prints internal nf etc. to stdout) at a few
coordinate extents, and prints our set_nhg_type3 prediction for the same
problems, including the largest anisotropic case at lat=1.5 (prediction only
— memory may not allow the fp64 run; that is the point of this probe).
"""

import sys
import pathlib

import numpy as np
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic                      # noqa: E402
from mlx_nufft.sizing import kernel_params, set_nhg_type3, next235even  # noqa: E402


def predict(prob, eps, sigma):
    w, _ = kernel_params(eps, sigma)
    x, s = prob["x"], prob["s"]
    nfs, nups = [], []
    for d in range(3):
        X = 0.5 * (x[d].max() - x[d].min())
        S = 0.5 * (s[d].max() - s[d].min())
        nf, h, gam = set_nhg_type3(S, X, sigma, w)
        nfs.append(nf)
        nups.append(next235even(int(np.ceil(sigma * nf))))
    return w, nfs, nups


def mem_gb(nfs, nups, bytes_per=8):
    spread = np.prod([float(v) for v in nfs]) * bytes_per
    inner = np.prod([float(v) for v in nups]) * bytes_per
    return spread / 1e9, inner / 1e9


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "predict"

    if mode == "predict":
        for lat in (0.02, 0.1, 0.5, 1.5):
            prob = gen_anisotropic(N=256, P=5000, lat=lat)
            for eps, sigma in ((1e-5, 1.25), (1e-5, 2.0), (1e-6, 2.0)):
                w, nfs, nups = predict(prob, eps, sigma)
                sp32, in32 = mem_gb(nfs, nups, 8)
                sp64, in64 = mem_gb(nfs, nups, 16)
                print(f"lat={lat:<5} eps={eps:.0e} sigma={sigma:<4} w={w} "
                      f"nf={nfs} n_up={nups}  "
                      f"c64: {sp32:.2f}+{in32:.2f} GB  "
                      f"c128: {sp64:.2f}+{in64:.2f} GB")
    else:
        # run finufft with debug to see its actual internal sizes
        lat = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
        upsamp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        prob = gen_anisotropic(N=128, P=2000, lat=lat)
        x, c, s = prob["x"], prob["c"], prob["s"]
        f = finufft.nufft3d3(*x, c, *s, isign=+1, eps=eps, debug=1,
                             upsampfac=upsamp)
        print("done", np.linalg.norm(f))
