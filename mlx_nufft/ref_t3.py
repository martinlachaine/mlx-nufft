"""Numpy reference implementation of the 3D type-3 NUFFT, FINUFFT-structured.

Purpose: (1) validate conventions exactly against CPU FINUFFT in fp64,
(2) act as the precision-diagnosis instrument: each stage can be run in
fp32 or fp64 independently (Flags), so we can localize where single
precision loses accuracy before writing any Metal.

Structure (mirrors FINUFFT type 3, see Barnett-Magland-af Klinteberg 2019
sec. 4 and cuFINUFFT's t3 path):
  f_k = sum_j c_j exp(i*isign * s_k . x_j)
  - center/rescale sources x -> xi (fine-grid coords), targets s -> theta
  - prephase c'_j = c_j exp(i*isign D.x_j)            [D = target centers]
  - spread c' onto nf1 x nf2 x nf3 fine grid (ES kernel, width w)
  - inner type-2: deconvolve modes by kernel Fourier series, zero-pad to
    n_up = sigma*nf, FFT, interpolate at theta_k (ES kernel again)
  - target deconvolve by phihat(theta) and postphase exp(i*isign C.(s-D))
"""

from dataclasses import dataclass

import numpy as np

from .sizing import kernel_params, kernel_ft, set_nhg_type3, next235even

PI = np.pi


@dataclass
class Flags:
    """Per-stage precision: True = fp64, False = fp32."""

    coords64: bool = True   # center/rescale of source & target coordinates
    phases64: bool = True   # pre/post phase angle computation
    spread64: bool = True   # spread kernel eval + grid accumulation
    fft64: bool = True      # pad/deconvolve/FFT of the fine grid
    interp64: bool = True   # interpolation kernel eval + gather accumulation
    deconv64: bool = True   # final deconvolution multiply


PURE32 = Flags(False, False, False, False, False, False)
FULL64 = Flags(True, True, True, True, True, True)


def _ceil_int(a):
    return np.ceil(a).astype(np.int64)


class RefT3Plan:
    def __init__(self, x, s, eps=1e-5, isign=+1, upsampfac=1.25,
                 flags: Flags = FULL64, chunk=2048):
        """x: tuple of 3 fp64 arrays (P,) source coords; s: 3 arrays (M,)."""
        self.flags = f = flags
        self.isign = float(np.sign(isign))
        self.eps = eps
        self.sigma = upsampfac
        self.chunk = chunk
        self.w, self.beta = kernel_params(eps, upsampfac)
        w = self.w

        rdt = np.float64 if f.coords64 else np.float32
        x = [np.asarray(xd).astype(rdt) for xd in x]
        s = [np.asarray(sd).astype(rdt) for sd in s]
        self.P = x[0].size
        self.M = s[0].size

        self.nf, self.h, self.gam = [], [], []
        self.C, self.D, self.Ssafe = [], [], []
        for d in range(3):
            xmin, xmax = float(x[d].min()), float(x[d].max())
            smin, smax = float(s[d].min()), float(s[d].max())
            C = 0.5 * (xmin + xmax)
            D = 0.5 * (smin + smax)
            X = 0.5 * (xmax - xmin)
            S = 0.5 * (smax - smin)
            nf, h, gam = set_nhg_type3(S, X, upsampfac, w)
            # gam = nf/(2 sigma Ssafe) -> recover Ssafe consistently:
            Ssafe = nf / (2.0 * upsampfac * gam)
            self.C.append(C); self.D.append(D); self.Ssafe.append(Ssafe)
            self.nf.append(nf); self.h.append(h); self.gam.append(gam)

        self.n_up = [next235even(int(np.ceil(upsampfac * nf))) for nf in self.nf]

        # --- rescaled source grid coordinates xi in [0, nf) -----------------
        # xi = (x - C)/gam/h + nf/2 ; split into leftmost index + fp32 frac
        self.i1 = []      # leftmost affected grid index (int64), per dim
        self.fr = []      # xi - i1, in coord dtype, per dim
        for d in range(3):
            xi = (x[d] - rdt(self.C[d])) / rdt(self.gam[d]) / rdt(self.h[d]) \
                + rdt(self.nf[d] / 2.0)
            i1 = _ceil_int(xi - rdt(w / 2.0))
            self.i1.append(i1)
            self.fr.append((xi - i1.astype(rdt)).astype(np.float64 if f.spread64
                                                        else np.float32))

        # --- rescaled targets: theta (rad per nf-grid-unit), grid coords ----
        # theta = h*gam*(s-D) = pi*(s-D)/(sigma*Ssafe); on n_up grid:
        # lam = theta/h2 + n_up/2,  h2 = 2*pi/n_up
        self.t_i1 = []
        self.t_fr = []
        self.theta = []
        for d in range(3):
            theta = (s[d] - rdt(self.D[d])) * rdt(self.h[d] * self.gam[d])
            self.theta.append(theta.astype(np.float64))
            lam = theta / rdt(2.0 * PI / self.n_up[d]) + rdt(self.n_up[d] / 2.0)
            i1 = _ceil_int(lam - rdt(w / 2.0))
            self.t_i1.append(i1)
            self.t_fr.append((lam - i1.astype(rdt)).astype(
                np.float64 if f.interp64 else np.float32))

        # --- pre/post phases (angle computation precision = phases64) -------
        pdt = np.float64 if f.phases64 else np.float32
        x_p = [xd.astype(pdt) for xd in x]
        s_p = [sd.astype(pdt) for sd in s]
        pre_ang = (pdt(self.D[0]) * x_p[0] + pdt(self.D[1]) * x_p[1]
                   + pdt(self.D[2]) * x_p[2])
        post_ang = (pdt(self.C[0]) * (s_p[0] - pdt(self.D[0]))
                    + pdt(self.C[1]) * (s_p[1] - pdt(self.D[1]))
                    + pdt(self.C[2]) * (s_p[2] - pdt(self.D[2])))
        cdt_ph = np.complex128 if f.phases64 else np.complex64
        self.prephase = np.exp(1j * self.isign * pre_ang.astype(np.float64)
                               ).astype(cdt_ph)
        postphase = np.exp(1j * self.isign * post_ang.astype(np.float64))

        # --- deconvolution factors (always computed fp64; storage by flag) --
        # target: 1/prod_d phihat(theta_d); modes: 1/phihat(2 pi q / n_up)
        tdec = np.ones(self.M, dtype=np.float64)
        for d in range(3):
            tdec *= kernel_ft(self.theta[d], self.beta, w)
        cdt_dec = np.complex128 if f.deconv64 else np.complex64
        self.post = (postphase / tdec).astype(cdt_dec)

        self.mode_dec = []   # per-dim 1/phihat(2 pi q/n_up) * (-1)^q, len nf
        for d in range(3):
            q = np.arange(-self.nf[d] // 2, self.nf[d] // 2, dtype=np.float64)
            ph = kernel_ft(2.0 * PI * q / self.n_up[d], self.beta, w)
            dec = ((-1.0) ** np.abs(q)) / ph
            self.mode_dec.append(dec.astype(np.float64 if f.fft64
                                            else np.float32))

    # ------------------------------------------------------------------
    def _kernel_1d(self, fr, dtype):
        """ES kernel weights for offsets l=0..w-1 at distance (l - fr).

        fr: (n,) fractional positions. Returns (n, w) weights in dtype.
        """
        w, beta = self.w, dtype(self.beta)
        l = np.arange(w, dtype=dtype)
        d = l[None, :] - fr.astype(dtype)[:, None]          # (n, w)
        z2 = (dtype(2.0 / w) * d) ** 2
        arg = beta * (np.sqrt(np.maximum(dtype(1.0) - z2, dtype(0.0)))
                      - dtype(1.0))
        out = np.exp(arg)
        out[z2 > 1.0] = 0.0
        return out

    def execute(self, c):
        f = self.flags
        w = self.w
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up

        # 1) prephase
        cdt = np.complex128 if f.spread64 else np.complex64
        cp = (np.asarray(c) * self.prephase).astype(cdt)

        # 2) spread
        sdt = np.float64 if f.spread64 else np.float32
        b = np.zeros(nf1 * nf2 * nf3, dtype=cdt)
        for a0 in range(0, self.P, self.chunk):
            sl = slice(a0, min(a0 + self.chunk, self.P))
            wx = self._kernel_1d(self.fr[0][sl], sdt)
            wy = self._kernel_1d(self.fr[1][sl], sdt)
            wz = self._kernel_1d(self.fr[2][sl], sdt)
            ix = (self.i1[0][sl, None] + np.arange(w)) % nf1
            iy = (self.i1[1][sl, None] + np.arange(w)) % nf2
            iz = (self.i1[2][sl, None] + np.arange(w)) % nf3
            vals = (cp[sl, None, None, None]
                    * wx[:, :, None, None]
                    * wy[:, None, :, None]
                    * wz[:, None, None, :])
            flat = ((ix[:, :, None, None] * nf2 + iy[:, None, :, None]) * nf3
                    + iz[:, None, None, :])
            np.add.at(b, flat.ravel(), vals.ravel())
        b = b.reshape(nf1, nf2, nf3)

        # 3) deconvolve modes, zero-pad (FFT-wrapped), FFT
        fdt = np.complex128 if f.fft64 else np.complex64
        dec = (self.mode_dec[0][:, None, None]
               * self.mode_dec[1][None, :, None]
               * self.mode_dec[2][None, None, :])
        H = np.zeros((nu1, nu2, nu3), dtype=fdt)
        r1 = np.arange(-nf1 // 2, nf1 // 2) % nu1
        r2 = np.arange(-nf2 // 2, nf2 // 2) % nu2
        r3 = np.arange(-nf3 // 2, nf3 // 2) % nu3
        H[np.ix_(r1, r2, r3)] = (b * dec).astype(fdt)
        if self.isign > 0:
            v = np.fft.ifftn(H) * (nu1 * nu2 * nu3)
        else:
            v = np.fft.fftn(H)
        v = v.astype(np.complex128 if f.interp64 else np.complex64)

        # 4) interpolate at targets
        idt = np.float64 if f.interp64 else np.float32
        out = np.zeros(self.M, dtype=v.dtype)
        vflat = v.ravel()
        for a0 in range(0, self.M, self.chunk):
            sl = slice(a0, min(a0 + self.chunk, self.M))
            wx = self._kernel_1d(self.t_fr[0][sl], idt)
            wy = self._kernel_1d(self.t_fr[1][sl], idt)
            wz = self._kernel_1d(self.t_fr[2][sl], idt)
            ix = (self.t_i1[0][sl, None] + np.arange(w)) % nu1
            iy = (self.t_i1[1][sl, None] + np.arange(w)) % nu2
            iz = (self.t_i1[2][sl, None] + np.arange(w)) % nu3
            flat = ((ix[:, :, None, None] * nu2 + iy[:, None, :, None]) * nu3
                    + iz[:, None, None, :])
            wgt = (wx[:, :, None, None] * wy[:, None, :, None]
                   * wz[:, None, None, :])
            out[sl] = (vflat[flat] * wgt).sum(axis=(1, 2, 3))

        # 5) final deconvolution + postphase
        return out * self.post


def nufft3d3_ref(x1, x2, x3, c, s1, s2, s3, eps=1e-5, isign=+1,
                 upsampfac=1.25, flags: Flags = FULL64):
    plan = RefT3Plan((x1, x2, x3), (s1, s2, s3), eps=eps, isign=isign,
                     upsampfac=upsampfac, flags=flags)
    return plan.execute(c)
