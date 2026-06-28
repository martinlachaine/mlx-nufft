"""Type-1 and type-2 3D NUFFT plans on Apple GPU (derivatives of type 3).

Conventions mirror FINUFFT (modeord=0 / CMCL):
  type 1:  f[k] = sum_j c[j] exp(i*isign * k . x_j),  k_d in [-N_d/2, N_d/2)
  type 2:  c[j] = sum_k f[k] exp(i*isign * k . x_j)
x in [-pi, pi)^3 (any values accepted; folded mod 2pi in fp64 at plan time).
Mode dims must be even. fp32 GPU pipeline; coordinate handling in fp64 at
plan time by default ('crit64') — at N=2048 modes, fp32 coordinates alone
cost ~2e-4, so this matters exactly as in type 3.

Whole-grid path only (fine grid = next235even(max(2w, sigma*N)) per dim);
the type-3 slab machinery is not needed at type-1/2 problem sizes we target.
"""

import numpy as np
import mlx.core as mx

from .sizing import kernel_params, kernel_ft, next235even
from .gpu_t3 import _es_msl, fft_axis

PI = np.pi


class _Points12:
    """Shared point/kernel setup for types 1 and 2."""

    def __init__(self, x, n_modes, eps, isign, upsampfac, prec, sort_points):
        assert prec in ("fp32", "crit64")
        self.prec = prec
        self.isign = int(np.sign(isign))
        self.eps = eps
        self.sigma = upsampfac
        self.w, self.beta = kernel_params(eps, upsampfac)
        w = self.w
        self.N = tuple(int(n) for n in n_modes)
        assert len(self.N) == 3 and all(n % 2 == 0 for n in self.N), \
            "mode dims must be 3D and even"
        self.n_up = [next235even(max(2 * w, int(np.ceil(upsampfac * n))))
                     for n in self.N]

        x64 = [np.asarray(v, dtype=np.float64) for v in x]
        self.P = x64[0].size
        rdt = np.float64 if prec == "crit64" else np.float32

        self.i1, self.fr = [], []
        for d in range(3):
            nu = self.n_up[d]
            h = 2.0 * PI / nu
            xi = (np.mod(x64[d], 2.0 * PI).astype(rdt) / rdt(h))
            ii = np.ceil(xi - rdt(w / 2.0)).astype(np.int32)
            self.i1.append(ii)
            self.fr.append((xi - ii.astype(rdt)).astype(np.float32))

        if sort_points:
            self.perm = np.argsort(
                self.i1[0].astype(np.int64) * self.n_up[1] + self.i1[1],
                kind="stable")
        else:
            self.perm = np.arange(self.P)
        self.sorted = sort_points

        # mode deconvolution: 1/phihat(2 pi k / n_up); no (-1)^k here —
        # unlike type 3, the type-1/2 fine grid is uncentered (u_l = l*h,
        # x folded into [0, 2pi)), so there is no half-grid shift to absorb.
        # FFT normalization for isign=+1 folded into dim 0.
        decs = []
        for d in range(3):
            k = np.arange(-self.N[d] // 2, self.N[d] // 2, dtype=np.float64)
            ph = kernel_ft(2.0 * PI * k / self.n_up[d], self.beta, w)
            decs.append(1.0 / ph)
        if self.isign > 0:
            decs[0] = decs[0] * float(np.prod([float(n) for n in self.n_up]))
        self.decs = decs

        self.mx_i1 = [mx.array(v[self.perm]) for v in self.i1]
        self.mx_fr = [mx.array(v[self.perm]) for v in self.fr]
        self.mx_perm = mx.array(self.perm.astype(np.uint32))
        self.mx_dec = [mx.array(v.astype(np.float32)) for v in decs]
        self._twiddles = {}

    def _fft_grid(self, Hf):
        nu1, nu2, nu3 = self.n_up
        H = mx.view(Hf, dtype=mx.complex64).reshape(nu1, nu2, nu3)
        for ax in (2, 1, 0):
            Hn = fft_axis(H, ax, inverse=self.isign > 0,
                          twiddle_cache=self._twiddles)
            del H
            H = Hn
            del Hn
        mx.eval(H)
        return H


class Type1Plan(_Points12):
    """f[k] = sum_j c[j] exp(i*isign * k . x_j) on an even CMCL mode box."""

    def __init__(self, x, n_modes, eps=1e-5, isign=+1, upsampfac=1.25,
                 prec="crit64", sort_points=True):
        super().__init__(x, n_modes, eps, isign, upsampfac, prec, sort_points)
        w = self.w
        nu1, nu2, nu3 = self.n_up
        N1, N2, N3 = self.N
        es = _es_msl("k1", w, self.beta)

        spread_src = f"""
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    if (lane >= {w * w}u || j >= {self.P}u) return;
    int lx = (int)(lane / {w}u), ly = (int)(lane % {w}u);
    float wxy = k1_es((float)lx - frx[j]) * k1_es((float)ly - fry[j]);
    float cre = cj[2*j] * wxy, cim = cj[2*j+1] * wxy;
    int ix = i1x[j] + lx;  ix -= {nu1} * (ix >= {nu1});  ix += {nu1} * (ix < 0);
    int iy = i1y[j] + ly;  iy -= {nu2} * (iy >= {nu2});  iy += {nu2} * (iy < 0);
    size_t base = ((size_t)ix * {nu2} + (size_t)iy) * {nu3};
    int iz0 = i1z[j];
    float fz = frz[j];
    for (int lz = 0; lz < {w}; ++lz) {{
        float wz = k1_es((float)lz - fz);
        int iz = iz0 + lz;
        iz -= {nu3} * (iz >= {nu3});  iz += {nu3} * (iz < 0);
        atomic_fetch_add_explicit(&grid[2*(base + (size_t)iz)],
                                  cre * wz, memory_order_relaxed);
        atomic_fetch_add_explicit(&grid[2*(base + (size_t)iz) + 1],
                                  cim * wz, memory_order_relaxed);
    }}
"""
        self._spread = mx.fast.metal_kernel(
            name="t1spread", input_names=["cj", "i1x", "i1y", "i1z",
                                          "frx", "fry", "frz"],
            output_names=["grid"], header="#include <metal_math>\n" + es,
            source=spread_src, atomic_outputs=True)

        # crop FFT-order fine grid to CMCL mode box + deconvolve
        crop_src = f"""
    uint m3 = thread_position_in_grid.x;
    uint m2 = thread_position_in_grid.y;
    uint m1 = thread_position_in_grid.z;
    if (m3 >= {N3}u || m2 >= {N2}u || m1 >= {N1}u) return;
    int q1 = (int)m1 - {N1 // 2};  int r1 = q1 + {nu1} * (q1 < 0);
    int q2 = (int)m2 - {N2 // 2};  int r2 = q2 + {nu2} * (q2 < 0);
    int q3 = (int)m3 - {N3 // 2};  int r3 = q3 + {nu3} * (q3 < 0);
    size_t src = ((size_t)r1 * {nu2} + r2) * {nu3} + r3;
    size_t dst = ((size_t)m1 * {N2} + m2) * {N3} + m3;
    float d = dec1[m1] * dec2[m2] * dec3[m3];
    fk[2*dst]   = v[2*src] * d;
    fk[2*dst+1] = v[2*src+1] * d;
"""
        self._crop = mx.fast.metal_kernel(
            name="t1crop", input_names=["v", "dec1", "dec2", "dec3"],
            output_names=["fk"], source=crop_src)

    def execute(self, c, return_np=True):
        w = self.w
        nu1, nu2, nu3 = self.n_up
        N1, N2, N3 = self.N
        cmx = mx.array(np.asarray(c).astype(np.complex64)) \
            if not isinstance(c, mx.array) else c
        cpf = mx.view(mx.take(cmx, self.mx_perm), dtype=mx.float32)
        bf = self._spread(
            inputs=[cpf, self.mx_i1[0], self.mx_i1[1], self.mx_i1[2],
                    self.mx_fr[0], self.mx_fr[1], self.mx_fr[2]],
            output_shapes=[(nu1 * nu2 * nu3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(w * w, self.P, 1), threadgroup=(w * w, 1024 // (w * w), 1),
            init_value=0)[0]
        mx.eval(bf)
        H = self._fft_grid(bf)
        del bf
        vf = mx.view(H, dtype=mx.float32).reshape(-1)
        fk = self._crop(
            inputs=[vf, self.mx_dec[0], self.mx_dec[1], self.mx_dec[2]],
            output_shapes=[(N1 * N2 * N3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(N3, N2, N1),
            threadgroup=(min(N3, 256), max(1, 256 // min(N3, 256)), 1))[0]
        del H, vf
        res = mx.view(fk, dtype=mx.complex64).reshape(N1, N2, N3)
        mx.eval(res)
        return np.array(res) if return_np else res


class Type2Plan(_Points12):
    """c[j] = sum_k f[k] exp(i*isign * k . x_j), f on an even CMCL box."""

    def __init__(self, x, n_modes, eps=1e-5, isign=+1, upsampfac=1.25,
                 prec="crit64", sort_points=False):
        super().__init__(x, n_modes, eps, isign, upsampfac, prec, sort_points)
        w = self.w
        nu1, nu2, nu3 = self.n_up
        N1, N2, N3 = self.N
        es = _es_msl("k2", w, self.beta)

        # pad CMCL mode box into FFT-order fine grid + deconvolve
        pad_src = f"""
    uint r3 = thread_position_in_grid.x;
    uint r2 = thread_position_in_grid.y;
    uint r1 = thread_position_in_grid.z;
    if (r3 >= {nu3}u || r2 >= {nu2}u || r1 >= {nu1}u) return;
    size_t dst = ((size_t)r1 * {nu2} + r2) * {nu3} + r3;
    int q1 = (int)r1;  q1 -= {nu1} * (q1 >= {nu1 - N1 // 2});
    int q2 = (int)r2;  q2 -= {nu2} * (q2 >= {nu2 - N2 // 2});
    int q3 = (int)r3;  q3 -= {nu3} * (q3 >= {nu3 - N3 // 2});
    bool inband = (q1 >= {-(N1 // 2)} && q1 < {N1 // 2}
                && q2 >= {-(N2 // 2)} && q2 < {N2 // 2}
                && q3 >= {-(N3 // 2)} && q3 < {N3 // 2});
    if (!inband) {{ H[2*dst] = 0.0f; H[2*dst+1] = 0.0f; return; }}
    int m1 = q1 + {N1 // 2}, m2 = q2 + {N2 // 2}, m3 = q3 + {N3 // 2};
    size_t src = ((size_t)m1 * {N2} + m2) * {N3} + m3;
    float d = dec1[m1] * dec2[m2] * dec3[m3];
    H[2*dst]   = fk[2*src] * d;
    H[2*dst+1] = fk[2*src+1] * d;
"""
        self._pad = mx.fast.metal_kernel(
            name="t2pad", input_names=["fk", "dec1", "dec2", "dec3"],
            output_names=["H"], source=pad_src)

        gather_src = f"""
    uint kk = thread_position_in_grid.x;
    if (kk >= {self.P}u) return;
    float wx[{w}], wy[{w}], wz[{w}];
    int jx[{w}], jy[{w}], jz[{w}];
    float fx = tfrx[kk], fy = tfry[kk], fz = tfrz[kk];
    int x0 = ti1x[kk], y0 = ti1y[kk], z0 = ti1z[kk];
    for (int l = 0; l < {w}; ++l) {{
        wx[l] = k2_es((float)l - fx);
        wy[l] = k2_es((float)l - fy);
        wz[l] = k2_es((float)l - fz);
        int a = x0 + l; a -= {nu1} * (a >= {nu1}); a += {nu1} * (a < 0); jx[l] = a;
        int b = y0 + l; b -= {nu2} * (b >= {nu2}); b += {nu2} * (b < 0); jy[l] = b;
        int cc = z0 + l; cc -= {nu3} * (cc >= {nu3}); cc += {nu3} * (cc < 0); jz[l] = cc;
    }}
    float accre = 0.0f, accim = 0.0f;
    for (int lx = 0; lx < {w}; ++lx) {{
        for (int ly = 0; ly < {w}; ++ly) {{
            float wxy = wx[lx] * wy[ly];
            size_t base = ((size_t)jx[lx] * {nu2} + (size_t)jy[ly]) * {nu3};
            float sre = 0.0f, sim = 0.0f;
            for (int lz = 0; lz < {w}; ++lz) {{
                float wv = wz[lz];
                size_t idx = base + (size_t)jz[lz];
                sre = metal::fma(v[2*idx], wv, sre);
                sim = metal::fma(v[2*idx+1], wv, sim);
            }}
            accre = metal::fma(sre, wxy, accre);
            accim = metal::fma(sim, wxy, accim);
        }}
    }}
    out[2*kk]   = accre;
    out[2*kk+1] = accim;
"""
        self._gather = mx.fast.metal_kernel(
            name="t2gather",
            input_names=["v", "ti1x", "ti1y", "ti1z",
                         "tfrx", "tfry", "tfrz"],
            output_names=["out"], header="#include <metal_math>\n" + es,
            source=gather_src)

    def execute(self, fk, return_np=True):
        nu1, nu2, nu3 = self.n_up
        N1, N2, N3 = self.N
        fmx = mx.array(np.ascontiguousarray(fk).astype(np.complex64)) \
            if not isinstance(fk, mx.array) else fk
        fkf = mx.view(fmx.reshape(-1), dtype=mx.float32)
        Hf = self._pad(
            inputs=[fkf, self.mx_dec[0], self.mx_dec[1], self.mx_dec[2]],
            output_shapes=[(nu1 * nu2 * nu3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(nu3, nu2, nu1),
            threadgroup=(min(nu3, 32), 1024 // min(nu3, 32), 1))[0]
        mx.eval(Hf)
        H = self._fft_grid(Hf)
        del Hf
        vf = mx.view(H, dtype=mx.float32).reshape(-1)
        out = self._gather(
            inputs=[vf, self.mx_i1[0], self.mx_i1[1], self.mx_i1[2],
                    self.mx_fr[0], self.mx_fr[1], self.mx_fr[2]],
            output_shapes=[(self.P * 2,)],
            output_dtypes=[mx.float32],
            grid=(self.P, 1, 1), threadgroup=(256, 1, 1))[0]
        del H, vf
        res = mx.view(out, dtype=mx.complex64)
        mx.eval(res)
        if not return_np:
            return res
        res = np.array(res)
        if self.sorted:
            inv = np.empty_like(self.perm)
            inv[self.perm] = np.arange(self.P)
            res = res[inv]
        return res
