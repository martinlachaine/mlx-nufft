"""General double-single (df64) GPU math primitives.

These expose the extended-precision machinery the NUFFT plans use internally
(df64 phase accumulation + reduction mod 2pi, then f32 transcendentals) as
standalone tools for callers that hit the same f32-precision wall — most
commonly a large-magnitude fp64 phase that cannot be reduced mod 2pi in f32
but whose cos/sin you want to evaluate on the GPU.

The reduction is identical to the prephase inside GpuT3Plan.set_sources: form
the phase in df64, k = rint(phi/2pi), then phi - k*2pi in df64 (the product
k*2pi is exact via two_prod), and f32 cos/sin of the O(1) residual. Valid while
the integer quotient phi/2pi stays f32-exact, i.e. |phi| <~ 2^24 * 2pi ~ 1.05e8
radians; beyond that the residual grows and accuracy degrades gracefully.
"""

import numpy as np
import mlx.core as mx

from .gpu_t3 import _DF64_HDR

PI = np.pi

# valid magnitude ceiling: |phi| where rint(phi/2pi) stays an exact f32 integer
EXPI_MAX_PHASE = float(2 ** 24 * 2.0 * PI)      # ~1.054e8 radians

_expi_cache = {}


def _build_expi_kernel(ncomp):
    """e^{i*isign*phi} for phi = sum of `ncomp` df64 phase components:
    accumulate in df64, reduce mod 2pi (k = rint(phi/2pi); phi - k*2pi in
    df64, k*2pi exact via two_prod), then f32 cos/sin of the O(1) residual.

    consts: [2pi_hi, 2pi_lo, 1/2pi, isign]."""
    acc = "    df64 ph = df_make(0.0f, 0.0f);\n"
    for c in range(ncomp):
        acc += f"    ph = df_add(ph, df_make(ph_hi{c}[j], ph_lo{c}[j]));\n"
    src = f"""
    uint j = thread_position_in_grid.x;
    if (j >= (uint)P0[0]) return;
{acc}    float k = metal::rint(ph.hi * cst[2]);                 // ph.hi / 2pi
    df64 red = df_add(ph, df_mul(df_make(-k, 0.0f),
                                 df_make(cst[0], cst[1])));    // ph - k*2pi (df64)
    float ang = cst[3] * (red.hi + red.lo);
    out[2*j]   = metal::precise::cos(ang);
    out[2*j+1] = metal::precise::sin(ang);
"""
    innames = []
    for c in range(ncomp):
        innames += [f"ph_hi{c}", f"ph_lo{c}"]
    innames += ["cst", "P0"]
    return mx.fast.metal_kernel(
        name=f"expi_df64_n{ncomp}", input_names=innames,
        output_names=["out"], header=_DF64_HDR, source=src)


def expi(phases, isign=1, return_np=False):
    """Compute e^{i*isign*phi} on the GPU for a large-magnitude fp64 phase via
    a double-single (df64) reduction mod 2pi followed by f32 cos/sin — the
    general form of the prephase machinery inside GpuT3Plan.set_sources.

    Use this whenever an fp64 phase is too large to reduce in f32 (f32 loses
    all fractional bits by |phi| ~ 1e3) but you want the cos/sin on the GPU.

    Parameters
    ----------
    phases : fp64 array, or a sequence of fp64 arrays
        Either the phase phi directly, or a small set of phase components of
        identical shape that are summed *in df64* to form phi = sum_k phases[k]
        (more accurate than an fp64 host sum, and keeps the sum on-GPU).
    isign : int
        +1 (default) or -1; computes e^{i*isign*phi}.
    return_np : bool
        If True, copy the result to a numpy complex64 array; otherwise return
        the device mx.array (complex64).

    Returns
    -------
    e^{i*isign*phi} as complex64, same shape as the input.

    Accuracy: matches an fp64 host reference (np.exp) to ~1e-6 per element for
    |phi| up to EXPI_MAX_PHASE (~1.05e8 rad) — approaching ~2e-6 right at the
    ceiling, ~2e-7 for |phi| <~ 1e7. Beyond the ceiling the integer quotient
    phi/2pi is no longer f32-exact and accuracy degrades gracefully (~linear in
    |phi|, no cliff).

    Notes
    -----
    - A sequence of arrays is treated as SUMMABLE phase components (summed in
      df64). A flat python list/tuple of *scalars* is therefore rejected — it
      would otherwise be silently summed into a single phase; pass
      ``np.asarray(...)`` for a vector of per-element phases.
    - ``isign`` is normalized by sign only (>=0 -> +1, else -1).
    - NaN/inf phases propagate per element to NaN (no cross-element effect).
    """
    if isinstance(phases, (list, tuple)):
        comps = [np.asarray(v, dtype=np.float64) for v in phases]
    else:
        comps = [np.asarray(phases, dtype=np.float64)]
    if len(comps) < 1:
        raise ValueError("expi: at least one phase component is required")
    shape = comps[0].shape
    if any(c.shape != shape for c in comps):
        raise ValueError("expi: all phase components must share one shape")
    if len(comps) > 1 and shape == ():
        raise ValueError(
            "expi: got a multi-element sequence of scalars, which would be "
            "summed into a single phase. Pass np.asarray(...) for a vector of "
            "per-element phases, or 1-D arrays as summable phase components.")
    isign = +1 if isign >= 0 else -1
    P = int(np.prod(shape)) if shape else 1
    ncomp = len(comps)
    kern = _expi_cache.get(ncomp)
    if kern is None:
        kern = _expi_cache[ncomp] = _build_expi_kernel(ncomp)
    cst = np.zeros(4, dtype=np.float32)
    cst[0] = np.float32(2.0 * PI)
    cst[1] = np.float32(2.0 * PI - np.float64(cst[0]))
    cst[2] = np.float32(1.0 / (2.0 * PI))
    cst[3] = np.float32(isign)
    ins = []
    with np.errstate(invalid="ignore"):       # inf-in -> nan-out, no warning
        for c in comps:
            cf = c.ravel()
            hi = cf.astype(np.float32)
            lo = (cf - hi).astype(np.float32)
            ins += [mx.array(hi), mx.array(lo)]
    ins += [mx.array(cst), mx.array(np.array([P], dtype=np.int32))]
    out = kern(inputs=ins, output_shapes=[(2 * P,)],
               output_dtypes=[mx.float32],
               grid=(P, 1, 1), threadgroup=(256, 1, 1))[0]
    res = mx.reshape(mx.view(out, dtype=mx.complex64), shape)
    mx.eval(res)
    return np.array(res) if return_np else res
