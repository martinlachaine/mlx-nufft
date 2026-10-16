"""Gate the chunked type-3 target setup.

Type3Plan computes its target-side arrays (cell indices, offsets, and the
phase-and-deconvolution factors) in chunks on a thread pool once a plan has
more than gpu_t3._CHUNK targets. Every operation is elementwise, so the result
must be bit-identical to one pass over all targets, in every precision mode.

Prints PASS/FAIL per check; exits non-zero on any failure.

    .venv/bin/python harness/test_t3_plan_chunks.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen import gen_anisotropic, gen_generic     # noqa: E402
import mlx_nufft.gpu_t3 as g3                     # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def arrays(p):
    out = [np.asarray(v) for v in p.t_i1] + [np.asarray(v) for v in p.t_fr]
    return out + [np.asarray(p.mx_post), np.asarray(p.mx_post_c)]


def main():
    cases = (("generic", gen_generic(N=768, P=2000, seed=4), 3e-5),
             ("anisotropic", gen_anisotropic(N=768, P=2000, lat=1.5, seed=4), 1e-5))
    chunk = g3._CHUNK
    for name, g, eps in cases:
        M = g["s"][0].size
        for prec in ("crit64", "fp32", "phases64", "coords64"):
            g3._CHUNK = chunk
            a = arrays(g3.GpuT3Plan(g["x"], g["s"], eps=eps, isign=+1, prec=prec))
            g3._CHUNK = 1 << 40               # one pass on the calling thread
            b = arrays(g3.GpuT3Plan(g["x"], g["s"], eps=eps, isign=+1, prec=prec))
            same = all(u.dtype == v.dtype and np.array_equal(u.view(np.uint8), v.view(np.uint8))
                       for u, v in zip(a, b))
            check(f"{name} {prec} M={M} ({-(-M // chunk)} chunks): bit-identical", same)
    g3._CHUNK = chunk
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
