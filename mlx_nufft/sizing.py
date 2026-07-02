"""Type-3 NUFFT sizing and ES-kernel parameters, mirroring FINUFFT.

All formulas are ports of FINUFFT's setup_spreader() / set_nhg_type3()
(src/spreadinterp.cpp, src/finufft_core.cpp) so that grid sizes and kernel
shapes match the CPU oracle's algorithm family.
"""

import numpy as np

PI = np.pi


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


def kernel_params(eps: float, upsampfac: float):
    """Kernel width w and ES beta for tolerance eps at given upsampling factor.

    Port of FINUFFT setup_spreader(): for sigma=2, w = ceil(log10(10/eps));
    otherwise w from the Liu lower-bound formula. beta = (beta/w)*w with the
    FINUFFT-tuned ratios.
    """
    if upsampfac == 2.0:
        ns = int(np.ceil(-np.log10(eps / 10.0)))
    else:
        ns = int(np.ceil(-np.log(eps) / (PI * np.sqrt(1.0 - 1.0 / upsampfac))))
    ns = max(2, min(ns, 16))
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
