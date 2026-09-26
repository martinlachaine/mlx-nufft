"""Type-3 NUFFT sizing and ES-kernel parameters, mirroring FINUFFT.

All formulas are ports of FINUFFT's setup_spreader() / set_nhg_type3()
(src/spreadinterp.cpp, src/finufft_core.cpp) so that grid sizes and kernel
shapes match the CPU oracle's algorithm family.
"""

import logging

import numpy as np

PI = np.pi
_log = logging.getLogger(__name__)

# FINUFFT setup_spreadinterp's single-precision guard ("ns reducing from N
# to 8 to prevent r_{dyn}-related catastrophic cancellation"): the
# deconvolution divides by the kernel FT, whose dynamic range across the
# band, r_dyn = phihat(0)/phihat(pi/sigma), is 34 at w=8, 56 at w=9, 93 at
# w=10 and 1863 at w=16 for sigma=1.25 (2 to 8 at every w for sigma=2), and
# it multiplies the fp32 grid's rounding error. A target at a corner of
# the band sees the product over the axes that reach the band edge, so
# the width past which a wider kernel makes the result worse depends on
# how many axes do. With three it is 8 (corner amplification 4.0e4 at
# w=8, 1.8e5 at w=9, 8.0e5 at w=10: FINUFFT's rule; w=9/10 measured 2x to
# 7x worse in 3D). With two or fewer it is 10, the width the API's eps
# floor of 1e-6 implies: 2D 8.6e3 at w=10, a slab with a thin third axis
# 2.4e4, a rod 7.2e2, all under the three-axis figure at w=8, and w=8
# measured 2x worse than w=9/10 on 2D grids and the thin slab. The grid
# here is always fp32, so the caps apply at every sigma < 2 (FINUFFT
# applies its one at its one low sigma, 1.25); at sigma=1.25 they engage
# below eps 1.3e-5 (three axes) and 7.9e-7 (otherwise).
W_MAX_FP32_LOWSIGMA_3D = 8
W_MAX_FP32_LOWSIGMA = 10
# An axis is thin when its targets reach at most this fraction of the
# kernel band [-pi/sigma, pi/sigma] in the deconvolution (band_fraction
# below; the acceptance suite's slab has 0.26 on z, an embedded 1D/2D
# transform 0). Up to 0.5 the per-axis amplification is at most 2.8 at
# w=10 (2.5 at w=9) against 93 at the band edge, which is what keeps the
# slab and rod figures above under the three-axis one.
THIN_AXIS_BAND = 0.5


def next235even(n: int) -> int:
    """Smallest even integer >= n whose prime factors are all in {2,3,5}."""
    n = max(int(n), 2)
    if n % 2:
        n += 1
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 2


def kernel_params(eps: float, upsampfac: float, nfull=None):
    """Kernel width w and ES beta for tolerance eps at given upsampling factor.

    Port of FINUFFT setup_spreader(): for sigma=2, w = ceil(log10(10/eps));
    otherwise w from the Liu lower-bound formula, then capped for sigma < 2
    by nfull, the number of axes that carry a full band (ND types 1/2: the
    dimension; type 3: full_band_axes; None: unknown, takes the three-axis
    cap). beta = (beta/w)*w with the FINUFFT-tuned ratios.
    """
    ns = cap_kernel_width(finufft_width(eps, upsampfac), upsampfac, nfull)
    betaoverns = 2.30
    if ns == 2:
        betaoverns = 2.20
    elif ns == 3:
        betaoverns = 2.26
    elif ns == 4:
        betaoverns = 2.38
    if upsampfac != 2.0:
        gamma = 0.97
        betaoverns = gamma * PI * (1.0 - 1.0 / (2.0 * upsampfac))
    return ns, betaoverns * ns


def finufft_width(eps: float, upsampfac: float):
    """FINUFFT's kernel width for tolerance eps, before the fp32 low-sigma
    cap: w = ceil(log10(10/eps)) at sigma=2, the Liu lower-bound formula
    otherwise, clamped to [2, 16]. This is the width double-precision
    FINUFFT uses, so harness memory models of the CPU reference call it."""
    if upsampfac == 2.0:
        ns = int(np.ceil(-np.log10(eps / 10.0)))
    else:
        ns = int(np.ceil(-np.log(eps) / (PI * np.sqrt(1.0 - 1.0 / upsampfac))))
    return max(2, min(ns, 16))


def cap_kernel_width(ns, upsampfac, nfull=None):
    """Apply the fp32 low-sigma cap for nfull full-band axes (None: the
    three-axis cap) to a chosen width. A debug-level note, not a warning:
    every 3D sigma=1.25 plan at eps <= 1e-5 takes it, and the API already
    warns once about the fp32 envelope when it clamps eps."""
    cap = W_MAX_FP32_LOWSIGMA_3D if nfull is None or nfull >= 3 \
        else W_MAX_FP32_LOWSIGMA
    if upsampfac < 2.0 and ns > cap:
        _log.debug("kernel width %d reduced to %d at upsampfac=%g, %s "
                   "full-band axes (fp32 r_dyn guard)", ns, cap, upsampfac,
                   "unknown" if nfull is None else nfull)
        ns = cap
    return ns


def band_fraction(S, X):
    """Fraction of the kernel band [-pi/sigma, pi/sigma] that one axis's
    targets reach in the type-3 deconvolution: S/Ssafe, with Ssafe the
    half-extent set_nhg_type3 pads S to (1/X once S*X < 1, so the
    argument pi*(s-D)/(sigma*Ssafe) stops at S*X of the band edge; 1 when
    both vanish, as on the axes a 1D/2D transform embeds in 3D)."""
    if X == 0.0:
        return 0.0 if S == 0.0 else 1.0
    return min(1.0, S * X)


def full_band_axes(S, X):
    """Number of axes whose targets reach past THIN_AXIS_BAND of the kernel
    band: the effective dimension kernel_params caps the width for."""
    return sum(band_fraction(s, x) > THIN_AXIS_BAND for s, x in zip(S, X))


def es_kernel(d, beta, w):
    """ES kernel value at distance d (in fine-grid units), support |d| <= w/2.

    psi(d) = exp(beta*(sqrt(1-(2d/w)^2)-1)), zero outside support.
    """
    z2 = (2.0 * np.asarray(d, dtype=np.float64) / w) ** 2
    inside = z2 <= 1.0
    out = np.zeros_like(z2)
    out[inside] = np.exp(beta * (np.sqrt(1.0 - z2[inside]) - 1.0))
    return out


_GL_CACHE = {}


def _gauss_legendre(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def kernel_ft(xi, beta, w, nquad=128):
    """phihat(xi) = int_{-w/2}^{w/2} psi(d) e^{i xi d} dd, computed in fp64.

    xi is in radians per fine-grid unit. Returns real array (kernel is even).
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    nodes, weights = _gauss_legendre(nquad)
    # map [-1,1] -> [0, w/2]; use even symmetry: 2*int_0^{w/2} psi cos(xi d)
    d = 0.5 * (nodes + 1.0) * (w / 2.0)
    wq = weights * (w / 4.0)
    vals = es_kernel(d, beta, w)
    return 2.0 * (wq * vals) @ np.cos(np.outer(d, xi))


# ---- fast M-point kernel_ft: cached Chebyshev fit per (beta, w, nquad) ---
# Every caller needs |xi| <= pi/upsampfac < pi (type-3 rescaled targets and
# 2*pi*q/n_up mode arguments both live there), so one fit over [0, pi]
# serves all plans sharing kernel parameters. phihat is even and entire, so
# it is fit in t = 2*(xi/pi)^2 - 1 (even symmetry halves the degree).
_CHEB_XMAX = PI
_CHEB_TOL = 1e-12          # relative to the nquad-node quadrature
_CHEB_DEGS = (16, 24, 32, 48, 64, 96, 128)
_CHEB_CACHE = {}


def _kernel_ft_cheb(beta, w, nquad):
    """Chebyshev coefficients matching kernel_ft to <= _CHEB_TOL relative on
    a dense probe of [0, pi], or None when no degree validates (wide kernels
    whose dynamic range puts the quadrature's own fp64 cancellation noise
    above the tolerance) — callers then keep the quadrature path."""
    key = (float(beta), int(w), int(nquad))
    if key in _CHEB_CACHE:
        return _CHEB_CACHE[key]
    cheb = np.polynomial.chebyshev
    probe = np.linspace(0.0, _CHEB_XMAX, 4097)
    ref = kernel_ft(probe, beta, w, nquad)
    co = None
    if (ref > 0.0).all():              # relative gate needs a positive band
        tp = 2.0 * (probe / _CHEB_XMAX) ** 2 - 1.0

        def g(t):
            xi = _CHEB_XMAX * np.sqrt(0.5 * (t + 1.0))
            return kernel_ft(xi, beta, w, nquad)

        for deg in _CHEB_DEGS:
            cand = cheb.chebinterpolate(g, deg)
            if (np.abs(cheb.chebval(tp, cand) - ref)
                    <= _CHEB_TOL * ref).all():
                co = cand
                break
    _CHEB_CACHE[key] = co
    return co


def kernel_ft_fast(xi, beta, w, nquad=128):
    """kernel_ft via the cached Chebyshev fit: O(degree) fma per point vs
    O(nquad) cos. Falls back to the quadrature when the fit cannot validate,
    and per-point for any |xi| beyond the fitted [0, pi] range."""
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    co = _kernel_ft_cheb(beta, w, nquad)
    if co is None:
        return kernel_ft(xi, beta, w, nquad)
    t = (2.0 / (_CHEB_XMAX * _CHEB_XMAX)) * (xi * xi) - 1.0
    out = np.polynomial.chebyshev.chebval(t, co)
    oob = t > 1.0
    if oob.any():
        out[oob] = kernel_ft(xi[oob], beta, w, nquad)
    return out


def set_nhg_type3(S, X, upsampfac, w):
    """Port of FINUFFT set_nhg_type3 for one dimension.

    S: half-extent of (centered) target frequencies
    X: half-extent of (centered) source points
    Returns (nf, h, gam).
    """
    nss = w + 1
    Xsafe, Ssafe = X, S
    if X == 0.0:
        if S == 0.0:
            Xsafe, Ssafe = 1.0, 1.0
        else:
            Xsafe = max(Xsafe, 1.0 / S)
    else:
        Ssafe = max(Ssafe, 1.0 / X)
    nfd = 2.0 * upsampfac * Ssafe * Xsafe / PI + nss
    nf = int(nfd)
    if nf < 2 * w:
        nf = 2 * w
    nf = next235even(nf)
    h = 2.0 * PI / nf
    gam = nf / (2.0 * upsampfac * Ssafe)
    return nf, h, gam
