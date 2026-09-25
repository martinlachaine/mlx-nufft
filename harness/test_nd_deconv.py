"""ND type-1/2 mode deconvolution through the certified Chebyshev proxy.

_PointsND (Type1PlanND / Type2PlanND) builds its 1/phihat mode factors
with sizing.kernel_ft_fast, as the type-3 plan does, instead of the
128-node quadrature at every mode (which was the bulk of a 1D N=2^20 plan
build). Gates: plan.decs vs the quadrature-built factors within the
proxy's certified 1e-12 relative, dims 1/2/3, both plan types,
representative w/beta/N including the 1D N=2^20 build; kernels the proxy
cannot certify (upsampfac 1.25 at w=12 and 14, widths the fp32 width cap
keeps every plan below, so kernel_ft_fast is called on the mode arguments
directly) and a certified fit disabled by hand both give the quadrature
factors bit for bit; type-1 and type-2 transforms (the factors are read by
every path's crop / pad kernel) vs CPU FINUFFT in dims 1/2/3, the 1D case
at N=2^20.
"""

import contextlib
import sys
import pathlib

import numpy as np
import finufft as cpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import rel_l2                           # noqa: E402
from mlx_nufft import sizing                             # noqa: E402
from mlx_nufft.sizing import (kernel_ft, kernel_ft_fast,  # noqa: E402
                              kernel_params, next235even)
from mlx_nufft.nd import Type1PlanND, Type2PlanND        # noqa: E402

rng = np.random.default_rng(31)
DEC_THR = 1e-12                  # the proxy's certified relative tolerance
FAILS = []


def check(label, err, thr):
    ok = err <= thr
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {err:.3e} (thr {thr:.1e})")
    if not ok:
        FAILS.append(label)


def check_true(label, ok):
    print(f"  {'PASS' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)


def pts(dim, n):
    return [rng.uniform(-np.pi, np.pi, n) for _ in range(dim)]


def strengths(n):
    return (rng.standard_normal(n)
            + 1j * rng.standard_normal(n)).astype(np.complex64)


def quad_decs(plan):
    """The factors as v0.2.0 built them: 1/phihat by the 128-node quadrature
    at every mode argument 2*pi*k/n_up, FFT normalization on dim 0."""
    decs = []
    for d in range(plan.dim):
        N, nu = plan.N[d], plan.n_up[d]
        k = np.arange(-(N // 2), N - N // 2, dtype=np.float64)
        decs.append(1.0 / kernel_ft(2.0 * np.pi * k / nu, plan.beta, plan.w))
    if plan.isign > 0:
        decs[0] = decs[0] * float(np.prod([float(n) for n in plan.n_up]))
    return decs


def max_rel(got, ref):
    return max(float(np.max(np.abs(g - r) / np.abs(r)))
               for g, r in zip(got, ref))


def bit_equal(got, ref):
    return all(np.array_equal(g, r) for g, r in zip(got, ref))


def certified(plan):
    return sizing._kernel_ft_cheb(plan.beta, plan.w, 128) is not None


@contextlib.contextmanager
def cap_lifted():
    """kernel_params without the fp32 low-sigma width cap: the widths the
    uncapped rule prescribes, which the fit declines from w=12 at sigma
    1.25 (cap_kernel_width reads the two constants at call time)."""
    saved = sizing.W_MAX_FP32_LOWSIGMA_3D, sizing.W_MAX_FP32_LOWSIGMA
    sizing.W_MAX_FP32_LOWSIGMA_3D = sizing.W_MAX_FP32_LOWSIGMA = 16
    try:
        yield
    finally:
        sizing.W_MAX_FP32_LOWSIGMA_3D, sizing.W_MAX_FP32_LOWSIGMA = saved


if __name__ == "__main__":
    print("== 1. deconvolution factors: Chebyshev proxy vs quadrature ==")
    cases = [                                   # (cls, N, eps, upsampfac)
        (Type1PlanND, (2 ** 20,), 1e-3, 2.0),   # the profiled 1D build, w=4
        (Type2PlanND, (2 ** 20,), 1e-5, 2.0),   # w=6, isign=-1 (unscaled)
        (Type1PlanND, (513, 255), 1e-6, 2.0),   # odd N, w=7
        (Type2PlanND, (512, 384), 1e-8, 2.0),   # w=9
        (Type1PlanND, (128, 96, 64), 1e-10, 2.0),   # w=11
        (Type1PlanND, (33, 17, 9), 1e-12, 2.0),     # w=13, degree-24 fit
        (Type1PlanND, (4096,), 1e-5, 1.25),         # sigma 1.25, w=9
        (Type2PlanND, (64, 48), 1e-6, 1.25),        # sigma 1.25, w=10
    ]
    for cls, N, eps, sigma in cases:
        plan = cls(pts(len(N), 500), N, eps=eps, upsampfac=sigma)
        tag = f"{cls.__name__} N={N} eps={eps:.0e} sigma={sigma} w={plan.w}"
        check_true(f"{tag}: fit certified", certified(plan))
        check(f"{tag}: decs vs quadrature max rel",
              max_rel(plan.decs, quad_decs(plan)), DEC_THR)

    # natural fallback, at the sizing level: sigma 1.25 at eps 1e-7 / 1e-8
    # prescribes w=12 / 14, widths the fit cannot certify. The fp32 width
    # cap keeps every plan at w <= 10 there, so the deconvolution call is
    # exercised directly: kernel_ft_fast on the mode arguments 2*pi*k/n_up
    # must return the quadrature bit for bit
    for eps in (1e-7, 1e-8):
        with cap_lifted():
            w, beta = kernel_params(eps, 1.25)
        check_true(f"fallback sigma 1.25 w={w}: fit declined",
                   w >= 12 and sizing._kernel_ft_cheb(beta, w, 128) is None)
        same = True
        for N in (40, 513, 4096):
            nu = next235even(max(2 * w, int(np.ceil(1.25 * N))))
            xi = 2.0 * np.pi * np.arange(-(N // 2), N - N // 2) / nu
            same &= np.array_equal(kernel_ft_fast(xi, beta, w),
                                   kernel_ft(xi, beta, w))
        check_true(f"fallback sigma 1.25 w={w}: kernel_ft_fast equals the "
                   "quadrature bit for bit on the mode arguments", same)
    # forced fallback: a certified kernel with its cached fit disabled
    w, beta = kernel_params(1e-3, 2.0)
    key = (float(beta), int(w), 128)
    missing = object()
    saved = sizing._CHEB_CACHE.get(key, missing)
    sizing._CHEB_CACHE[key] = None
    try:
        plan = Type1PlanND(pts(3, 500), (24, 20, 16), eps=1e-3)
        check_true(f"forced fallback w={plan.w} (fit disabled): decs equal "
                   "the quadrature bit for bit",
                   bit_equal(plan.decs, quad_decs(plan)))
    finally:
        if saved is missing:
            del sizing._CHEB_CACHE[key]
        else:
            sizing._CHEB_CACHE[key] = saved

    print("== 2. type-1 / type-2 transforms vs cpu "
          "(the factors are read by the crop / pad kernels) ==")
    P = 20000
    for dim, N in [(1, (2 ** 20,)), (1, (4096,)), (2, (128, 96)),
                   (3, (32, 32, 24))]:
        x = pts(dim, P)
        c = strengths(P)
        ref = getattr(cpu, f"nufft{dim}d1")(*x, c.astype(np.complex128), N,
                                            eps=1e-9, isign=+1)
        for eps in (1e-3, 1e-5):
            f = Type1PlanND(x, N, eps=eps).execute(c)
            check(f"t1 {dim}d N={N} P={P} eps={eps:.0e}: vs cpu",
                  rel_l2(f, ref), 4 * eps + 2e-4)
        fk = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        c2 = Type2PlanND(x, N, eps=1e-5).execute(fk.astype(np.complex64))
        ref2 = getattr(cpu, f"nufft{dim}d2")(*x, fk, eps=1e-9, isign=-1)
        check(f"t2 {dim}d N={N} P={P} eps=1e-05: vs cpu", rel_l2(c2, ref2),
              4e-5 + 2e-4)

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
