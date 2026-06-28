"""df64 (double-single) on-GPU setup demo.

The M2 finding is that the only precision-critical per-point operations
(coordinate rescale -> int+frac, pre-phase angle) are plan-time, so the
spike does them in fp64 on the host. If a workload changes its geometry
every call, they must run on the GPU - and Apple GPUs have no fp64. This
demo shows the df64 path: inputs split into (hi, lo) fp32 pairs, Dekker
TwoProd/TwoSum compensated arithmetic in a Metal kernel, producing the
same (i1, frac) and prephase as the host-fp64 plan.

Run: demo_df64.py [lat] [P]
"""

import sys
import time
import pathlib

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic                       # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan                  # noqa: E402

HDR = """
#include <metal_math>
struct df64 { float hi; float lo; };

inline df64 df_make(float h, float l) { df64 r; r.hi = h; r.lo = l; return r; }

inline df64 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float e = (a - (s - bb)) + (b - bb);
    return df_make(s, e);
}

inline df64 two_prod(float a, float b) {
    float p = a * b;
    float e = metal::fma(a, b, -p);
    return df_make(p, e);
}

inline df64 df_add(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    float lo = s.lo + (a.lo + b.lo);
    float hi = s.hi + lo;
    return df_make(hi, lo - (hi - s.hi));
}

inline df64 df_mul(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    float lo = p.lo + (a.hi * b.lo + a.lo * b.hi);
    float hi = p.hi + lo;
    return df_make(hi, lo - (hi - p.hi));
}
"""

SRC = """
    uint j = thread_position_in_grid.x;
    if (j >= (uint)P0[0]) return;
    // consts: [C_hi, C_lo, invgh_hi, invgh_lo, nf_half, w_half, D_hi, D_lo,
    //          twopi_hi, twopi_lo, inv2pi, isign]
    df64 x = df_make(xhi[j], xlo[j]);
    df64 negC = df_make(-cst[0], -cst[1]);
    df64 invgh = df_make(cst[2], cst[3]);

    // xi = (x - C) * invgh + nf/2
    df64 dx = df_add(x, negC);
    df64 xi = df_mul(dx, invgh);
    xi = df_add(xi, df_make(cst[4], 0.0f));

    // i1 = ceil(xi - w/2), frac = xi - i1
    df64 a = df_add(xi, df_make(-cst[5], 0.0f));
    float i1f = metal::ceil(a.hi);
    if (a.hi - i1f + a.lo > 0.0f) i1f += 1.0f;
    int i1v = (int)i1f;
    float fr = (a.hi - i1f) + a.lo + cst[5];     // frac = xi - i1

    // prephase angle = D*x, range-reduced mod 2pi
    df64 D = df_make(cst[6], cst[7]);
    df64 ph = df_mul(x, D);
    float k = metal::rint(ph.hi * cst[10]);
    df64 red = df_add(ph, df_mul(df_make(-k, 0.0f),
                                 df_make(cst[8], cst[9])));
    float ang = cst[11] * (red.hi + red.lo);
    i1o[j] = i1v;
    fro[j] = fr;
    pre[2*j]   = metal::precise::cos(ang);
    pre[2*j+1] = metal::precise::sin(ang);
"""


def split_df(v):
    hi = np.asarray(v, dtype=np.float32)
    lo = (np.asarray(v, dtype=np.float64) - hi).astype(np.float32)
    return hi, lo


if __name__ == "__main__":
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    prob = gen_anisotropic(N=128, P=P, lat=lat)
    x, s = prob["x"], prob["s"]

    plan = GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64",
                     sort_points=False)
    kern = mx.fast.metal_kernel(
        name="df64setup", input_names=["xhi", "xlo", "cst", "P0"],
        output_names=["i1o", "fro", "pre"], header=HDR, source=SRC)

    print(f"df64-on-GPU setup vs host-fp64 plan   (lat={lat}, P={P})")
    for d in range(3):
        w = plan.w
        hd = 2.0 * np.pi / plan.nf[d]
        invgh = 1.0 / (plan.gam[d] * hd)
        Chi, Clo = split_df([plan.C[d]])
        ghi, glo = split_df([invgh])
        Dhi, Dlo = split_df([plan.D[d]])
        thi, tlo = split_df([2.0 * np.pi])
        cst = np.array([Chi[0], Clo[0], ghi[0], glo[0],
                        plan.nf[d] / 2.0, w / 2.0, Dhi[0], Dlo[0],
                        thi[0], tlo[0], 1.0 / (2 * np.pi),
                        float(plan.isign)], dtype=np.float32)
        xhi, xlo = split_df(x[d])
        ins = [mx.array(xhi), mx.array(xlo), mx.array(cst),
               mx.array(np.array([P], dtype=np.int32))]
        outs = kern(inputs=ins,
                    output_shapes=[(P,), (P,), (2 * P,)],
                    output_dtypes=[mx.int32, mx.float32, mx.float32],
                    grid=(P, 1, 1), threadgroup=(256, 1, 1))
        mx.eval(*outs)
        mx.synchronize()
        t0 = time.perf_counter()
        outs = kern(inputs=ins,
                    output_shapes=[(P,), (P,), (2 * P,)],
                    output_dtypes=[mx.int32, mx.float32, mx.float32],
                    grid=(P, 1, 1), threadgroup=(256, 1, 1))
        mx.eval(*outs)
        mx.synchronize()
        dt = time.perf_counter() - t0
        i1g, frg, preg = (np.array(v) for v in outs)

        # host fp64 references (plan was built unsorted)
        i1h, frh = plan.i1[d], plan.fr[d]
        ang = plan.D[d] * np.asarray(x[d], dtype=np.float64)
        preh = np.exp(1j * plan.isign * ang)
        mism = int((i1g != i1h).sum())
        # frac comparison only where i1 matches (else off-by-one pairs)
        m = i1g == i1h
        dfr = np.abs(frg[m] - frh[m]).max()
        dpre = np.abs((preg[0::2] + 1j * preg[1::2]) - preh).max()
        print(f"  dim{d}: i1 mismatches {mism}/{P}  "
              f"max|dfrac| {dfr:.2e}  max|dprephase| {dpre:.2e}  "
              f"kernel {1000*dt:.2f} ms")
