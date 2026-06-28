"""Metal/Apple-GPU type-3 3D NUFFT via MLX (fp32 pipeline).

Plan/execute split mirroring (cu)FINUFFT:
  plan: sizing, centers, coordinate rescale -> (int index, fp32 frac),
        pre/post phases, deconvolution vectors, point sort. Host numpy.
  execute: prephase multiply, spread (custom Metal kernel, atomic float),
        pad+mode-deconvolve (custom Metal kernel), 3D FFT (mx.fft),
        gather/interp + final deconvolve (custom Metal kernel). All fp32.

Precision modes:
  'fp32'  : per-point setup math in fp32 (faithful naive Metal port; M1).
  'crit64': precision-critical setup ops (coordinate rescale, pre/post
            phase angles) in fp64 on host, cached in plan; identical fp32
            GPU pipeline (M2 candidate).
Scalar plan constants and the small deconvolution vectors are always fp64
on host (any Metal app has host doubles at plan time).

sigma_inner: optional smaller upsampling for the inner type-2 grid (with a
correspondingly wider interp kernel) to cut the dominant memory term.
"""

import numpy as np
import mlx.core as mx

from .sizing import kernel_params, kernel_ft, set_nhg_type3, next235even

PI = np.pi


def _es_msl(prefix, w, beta):
    return f"""
inline float {prefix}_es(float d) {{
    float z = d * {2.0 / w}f;
    float t = metal::max(1.0f - z * z, 0.0f);
    return metal::precise::exp({beta}f * (metal::precise::sqrt(t) - 1.0f));
}}
"""


_FFT_NATIVE_MAX = 4096
_FFT_POW2_MAX = 2 ** 20   # mlx 0.31.2: pow2 lengths above this hit a missing
                          # four_step_mem_8192 Metal kernel; route through our
                          # own four-step split instead (needed for 1D NUFFTs)
_SLAB_THRESHOLD = 3e9

# mx.fast.metal_kernel buffers are indexed with int32: a kernel input/output
# may hold at most 2**31-1 elements (probe-confirmed: 2147483647 OK,
# 2147483648 raises TypeError). The slab z-major grid Z is nu3*nu1*nu2*2
# float32 elements; at large lateral grids (nu1=nu2 >~ 6700 with nu3~24) it
# crosses this limit, so padz is produced in z-chunks that each stay under it.
_MK_MAX_ELEMS = 2 ** 31 - 1

# low_mem (per-stage mx.clear_cache to return freed buffers to the OS) trades
# ~2x speed for a smaller resident set. It only earns its keep when the working
# set is a real fraction of machine RAM; on a large-RAM machine the clear_cache
# is pure allocator churn. Default auto-keys on RAM: low_mem
# when ~2 live grids exceed _LOW_MEM_FRAC of total RAM. A 16 GB machine at a
# 5.6 GB grid stays low_mem=True (11.2 / 16 GB); a 128 GB machine turns it
# off (11.2 / 128 GB).
_LOW_MEM_FRAC = 0.30
_RAM_BYTES = None


def _total_ram_bytes():
    global _RAM_BYTES
    if _RAM_BYTES is None:
        try:
            import subprocess
            _RAM_BYTES = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True).stdout)
        except Exception:
            _RAM_BYTES = 16 * 2 ** 30   # conservative fallback
    return _RAM_BYTES


def _split_n(n):
    """Factor n = n1*n2, both <= _FFT_NATIVE_MAX, as balanced as possible.

    Balanced factors keep both sub-FFT lengths in MLX's efficient range;
    degenerate splits (e.g. 2 x 2400) trigger pathological internal copies
    on huge batches.
    """
    best = None
    for n1 in range(int(np.sqrt(n)), 1, -1):
        if n % n1 == 0 and n // n1 <= _FFT_NATIVE_MAX:
            best = (n1, n // n1)
            break
    if best is None:
        raise ValueError(f"cannot split FFT length {n}")
    return best


def fft_axis(H, axis, inverse, twiddle_cache=None):
    """(i)FFT along one axis; four-step Cooley-Tukey for long non-pow2 axes.

    MLX's Metal FFT is native (Stockham radix) only for n <= 4096 or powers
    of two; other lengths fall back to Bluestein with a >=2x pow2 pad, which
    at our sizes demands a ~17 GB buffer. The four-step split keeps every
    library FFT call native AND contiguous-on-the-last-axis: explicit one-way
    transposes replace MLX's internal transpose round-trips for strided axes.

    x viewed as (L, n1, n2, R), k = k1*n2 + k2, f = f1 + n1*f2:
      T1: (L, n2, R, n1) -> FFT last (k1->f1)
      twiddle (k2, f1)
      T2: (L, f1, R, n2) -> FFT last (k2->f2)
      T3: (L, f2, f1, R) -> reshape (L, n, R)
    """
    n = H.shape[axis]
    fft1 = mx.fft.ifft if inverse else mx.fft.fft
    if n <= _FFT_NATIVE_MAX or ((n & (n - 1)) == 0 and n <= _FFT_POW2_MAX):
        return fft1(H, axis=axis)
    n1, n2 = _split_n(n)
    sgn = +1.0 if inverse else -1.0
    key = (n, sgn)
    if twiddle_cache is not None and key in twiddle_cache:
        T = twiddle_cache[key]
    else:
        k2 = np.arange(n2, dtype=np.float64)[:, None]
        f1 = np.arange(n1, dtype=np.float64)[None, :]
        T = mx.array(np.exp(sgn * 2j * PI * f1 * k2 / n).astype(np.complex64))
        if twiddle_cache is not None:
            twiddle_cache[key] = T
    shp = list(H.shape)
    lead = shp[:axis]
    rest = shp[axis + 1:]
    nl = len(lead)
    Hv = H.reshape(*lead, n1, n2, *rest)
    nd = Hv.ndim
    # axes ids: lead 0..nl-1, n1 at nl, n2 at nl+1, rest nl+2..nd-1
    p1 = list(range(nl)) + [nl + 1] + list(range(nl + 2, nd)) + [nl]
    Y = mx.contiguous(mx.transpose(Hv, p1))          # (L, n2, R, n1)
    A = fft1(Y, axis=-1)                             # k1 -> f1
    del Y
    tshape = [1] * nl + [n2] + [1] * len(rest) + [n1]
    B = A * T.reshape(tshape)
    del A
    # (L, n2, R, f1) -> (L, f1, R, n2)
    p2 = list(range(nl)) + [nd - 1] + list(range(nl + 1, nd - 1)) + [nl]
    Y2 = mx.contiguous(mx.transpose(B, p2))          # (L, f1, R, n2)
    del B
    C = fft1(Y2, axis=-1)                            # k2 -> f2
    del Y2
    # (L, f1, R, f2) -> (L, f2, f1, R) -> reshape (L, n, R)
    p3 = list(range(nl)) + [nd - 1, nl] + list(range(nl + 1, nd - 1))
    X = mx.contiguous(mx.transpose(C, p3)).reshape(*lead, n, *rest)
    del C
    return X


def fft_axis_scrambled(H, axis, inverse, twiddle_cache=None):
    """fft_axis without the final un-scrambling transpose.

    Returns (X, (n1, n2)) where the transformed axis is stored in
    "scrambled" order: true frequency f = f1 + n1*f2 lives at position
    f1*n2 + f2. The permutation is separable per axis, so a downstream
    gather can absorb it via i -> (i % n1)*n2 + i//n1 at zero cost.
    For native lengths returns (fft(H), (1, n)) (identity scramble).
    """
    if axis != H.ndim - 1 and axis != -1:
        raise ValueError("fft_axis_scrambled requires the last axis "
                         "(scramble is only separable there)")
    n = H.shape[axis]
    fft1 = mx.fft.ifft if inverse else mx.fft.fft
    if n <= _FFT_NATIVE_MAX or ((n & (n - 1)) == 0 and n <= _FFT_POW2_MAX):
        return fft1(H, axis=axis), (1, n)
    n1, n2 = _split_n(n)
    sgn = +1.0 if inverse else -1.0
    key = (n, sgn)
    if twiddle_cache is not None and key in twiddle_cache:
        T = twiddle_cache[key]
    else:
        k2 = np.arange(n2, dtype=np.float64)[:, None]
        f1 = np.arange(n1, dtype=np.float64)[None, :]
        T = mx.array(np.exp(sgn * 2j * PI * f1 * k2 / n).astype(np.complex64))
        if twiddle_cache is not None:
            twiddle_cache[key] = T
    shp = list(H.shape)
    lead = shp[:axis]
    rest = shp[axis + 1:]
    nl = len(lead)
    Hv = H.reshape(*lead, n1, n2, *rest)
    nd = Hv.ndim
    p1 = list(range(nl)) + [nl + 1] + list(range(nl + 2, nd)) + [nl]
    Y = mx.contiguous(mx.transpose(Hv, p1))          # (L, n2, R, n1)
    A = fft1(Y, axis=-1)                             # k1 -> f1
    del Y
    tshape = [1] * nl + [n2] + [1] * len(rest) + [n1]
    B = A * T.reshape(tshape)
    del A
    p2 = list(range(nl)) + [nd - 1] + list(range(nl + 1, nd - 1)) + [nl]
    Y2 = mx.contiguous(mx.transpose(B, p2))          # (L, f1, R, n2)
    del B
    C = fft1(Y2, axis=-1)                            # k2 -> f2
    del Y2
    # scrambled: axis stored as (f1, f2) blocks = f1*n2 + f2
    X = C.reshape(*lead, n, *rest)
    del C
    return X, (n1, n2)


_DF64_HDR = """
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


def _build_df64_setup_kernel():
    """Per-point type-3 source setup in double-single arithmetic.

    consts layout: per dim d at offset 8d: [C_hi, C_lo, invgh_hi, invgh_lo,
    nf/2, w/2, D_hi, D_lo]; tail at 24: [2pi_hi, 2pi_lo, 1/2pi, isign].
    """
    body = ["""
    uint j = thread_position_in_grid.x;
    if (j >= (uint)P0[0]) return;
    df64 ph = df_make(0.0f, 0.0f);
"""]
    for d in range(3):
        b = d * 8
        body.append(f"""
    {{
        df64 x = df_make(xh{d}[j], xl{d}[j]);
        df64 dx = df_add(x, df_make(-cst[{b}], -cst[{b + 1}]));
        df64 xi = df_mul(dx, df_make(cst[{b + 2}], cst[{b + 3}]));
        xi = df_add(xi, df_make(cst[{b + 4}], 0.0f));
        df64 a = df_add(xi, df_make(-cst[{b + 5}], 0.0f));
        float i1f = metal::ceil(a.hi);
        if (a.hi - i1f + a.lo > 0.0f) i1f += 1.0f;
        i1o{d}[j] = (int)i1f;
        fro{d}[j] = (a.hi - i1f) + a.lo + cst[{b + 5}];
        ph = df_add(ph, df_mul(x, df_make(cst[{b + 6}], cst[{b + 7}])));
    }}
""")
    body.append("""
    float k = metal::rint(ph.hi * cst[26]);
    df64 red = df_add(ph, df_mul(df_make(-k, 0.0f),
                                 df_make(cst[24], cst[25])));
    float ang = cst[27] * (red.hi + red.lo);
    pre[2*j]   = metal::precise::cos(ang);
    pre[2*j+1] = metal::precise::sin(ang);
""")
    return mx.fast.metal_kernel(
        name="t3df64setup",
        input_names=["xh0", "xl0", "xh1", "xl1", "xh2", "xl2", "cst", "P0"],
        output_names=["i1o0", "i1o1", "i1o2", "fro0", "fro1", "fro2", "pre"],
        header=_DF64_HDR, source="".join(body))


def _inner_kernel_params(eps, sigma_inner):
    """Width/beta for the inner interp kernel at a given (small) sigma."""
    ns = int(np.ceil(-np.log(eps) / (PI * np.sqrt(1.0 - 1.0 / sigma_inner))))
    ns = max(2, min(ns, 16))
    beta = 0.97 * PI * (1.0 - 1.0 / (2.0 * sigma_inner)) * ns
    return ns, beta


class GpuT3Plan:
    def __init__(self, x, s, eps=1e-5, isign=+1, upsampfac=1.25,
                 prec="crit64", sigma_inner=None, sort_points=True,
                 low_mem=None, source_extent=None, fft_backend="mlx"):
        """source_extent: optional [(lo, hi)] * 3 coordinate bounds to size
        the plan for, instead of the extents of the initial x sample — use
        for repeated fixed-geometry workloads so set_sources() accepts any
        later source set that stays within the known coordinate bounds.

        fft_backend: "mlx" (default, the validated four-step) or "vkfft" (the
        optional VkFFT-Metal backend, ~2x faster on large grids — requires the
        bridge dylib, see mlx_nufft/vkfft_backend.py). The vkfft path emits
        the lateral FFT in natural order, so its gather uses an identity
        scramble; it covers the slab full-padz path (grid < 2**31 elements,
        n_up ≲ 9000/axis at nu3=24)."""
        assert prec in ("fp32", "coords64", "phases64", "crit64")
        assert fft_backend in ("mlx", "vkfft")
        if fft_backend == "vkfft":
            from . import vkfft_backend as _vk
            _vk.require()          # fail fast with a clear message if not built
        self.fft_backend = fft_backend
        self.prec = prec
        self.isign = int(np.sign(isign))
        self.eps = eps
        self.sigma = upsampfac
        self.w, self.beta = kernel_params(eps, upsampfac)
        w = self.w
        if sigma_inner is None or sigma_inner == upsampfac:
            self.sigma_in = upsampfac
            self.w2, self.beta2 = self.w, self.beta
        else:
            self.sigma_in = sigma_inner
            self.w2, self.beta2 = _inner_kernel_params(eps, sigma_inner)

        x64 = [np.asarray(v, dtype=np.float64) for v in x]
        s64 = [np.asarray(v, dtype=np.float64) for v in s]
        self.P = x64[0].size
        self.M = s64[0].size

        # ---- sizing (fp64 host scalars) ----------------------------------
        self.nf, self.gam, self.C, self.D = [], [], [], []
        for d in range(3):
            if source_extent is not None:
                xlo, xhi = (float(source_extent[d][0]),
                            float(source_extent[d][1]))
            else:
                xlo, xhi = float(x64[d].min()), float(x64[d].max())
            C = 0.5 * (xlo + xhi)
            D = 0.5 * (s64[d].min() + s64[d].max())
            X = 0.5 * (xhi - xlo)
            S = 0.5 * (s64[d].max() - s64[d].min())
            nf, h, gam = set_nhg_type3(S, X, upsampfac, w)
            self.nf.append(nf), self.gam.append(gam)
            self.C.append(C), self.D.append(D)
        self.n_up = [next235even(int(np.ceil(self.sigma_in * nf)))
                     for nf in self.nf]

        # ---- per-point setup: precision mode applies here ----------------
        rdt = np.float64 if prec in ("crit64", "coords64") else np.float32
        pdt = np.float64 if prec in ("crit64", "phases64") else np.float32
        self._rdt, self._pdt = rdt, pdt
        self._sort_points = sort_points
        # frozen source extents (set_sources validates new frames fit them)
        if source_extent is not None:
            self._xext = [(float(lo), float(hi)) for lo, hi in source_extent]
        else:
            self._xext = [(float(x64[d].min()), float(x64[d].max()))
                          for d in range(3)]
        ss = [v.astype(rdt) for v in s64]

        self.t_i1, self.t_fr = [], []
        theta64 = []
        for d in range(3):
            hd = 2.0 * PI / self.nf[d]
            th = (ss[d] - rdt(self.D[d])) * rdt(hd * self.gam[d])
            theta64.append(th.astype(np.float64))
            lam = th / rdt(2.0 * PI / self.n_up[d]) + rdt(self.n_up[d] / 2.0)
            ii = np.ceil(lam - rdt(self.w2 / 2.0)).astype(np.int32)
            self.t_i1.append(ii)
            self.t_fr.append((lam - ii.astype(rdt)).astype(np.float32))

        sp = [v.astype(pdt) for v in s64]
        post_ang = sum(pdt(self.C[d]) * (sp[d] - pdt(self.D[d]))
                       for d in range(3))
        postphase = np.exp(1j * self.isign * post_ang.astype(np.float64))

        # ---- deconvolution (host fp64) ------------------------------------
        tdec = np.ones(self.M, dtype=np.float64)
        for d in range(3):
            tdec *= kernel_ft(theta64[d], self.beta, w)
        post = (postphase / tdec).astype(np.complex64)

        decs = []
        for d in range(3):
            q = np.arange(-self.nf[d] // 2, self.nf[d] // 2, dtype=np.float64)
            ph = kernel_ft(2.0 * PI * q / self.n_up[d], self.beta2, self.w2)
            decs.append(((-1.0) ** np.abs(q)) / ph)
        # slab mode: z-DFT is fused into the pad kernel (no 1/nu3 factor),
        # lateral per-slab iFFTs carry 1/(nu1*nu2)
        self.slab_mode = ((np.prod([float(n) for n in self.n_up]) * 8)
                          > _SLAB_THRESHOLD)
        if self.isign > 0:
            scale = float(self.n_up[0]) * float(self.n_up[1])
            if not self.slab_mode:
                scale *= float(self.n_up[2])
            decs[0] = decs[0] * scale
        # z-DFT twiddle table tw[l3, m3] = exp(i*isign*2*pi*r3(m3)*l3/nu3),
        # with r3 the FFT-wrapped index of mode q3 = m3 - nf3/2 (fp64 host)
        nf3, nu3 = self.nf[2], self.n_up[2]
        q3 = np.arange(-nf3 // 2, nf3 // 2)
        r3 = np.where(q3 < 0, q3 + nu3, q3)[None, :].astype(np.float64)
        l3 = np.arange(nu3, dtype=np.float64)[:, None]
        tw = np.exp(2j * PI * self.isign * r3 * l3 / nu3).astype(np.complex64)
        self._tw_np = tw

        # ---- upload frozen target-side plan arrays to mx ------------------
        self.mx_ti1 = [mx.array(v) for v in self.t_i1]
        self.mx_tfr = [mx.array(v) for v in self.t_fr]
        self.mx_post = mx.array(np.ascontiguousarray(
            np.stack([post.real.astype(np.float32),
                      post.imag.astype(np.float32)], axis=-1)))
        self.mx_dec = [mx.array(v.astype(np.float32)) for v in decs]
        self.mx_tw = mx.array(np.ascontiguousarray(
            np.stack([self._tw_np.real, self._tw_np.imag], axis=-1)
            ).astype(np.float32))
        self.mx_post_c = mx.array(post)
        mx.eval(self.mx_post, self.mx_tw, self.mx_post_c,
                *self.mx_ti1, *self.mx_tfr, *self.mx_dec)

        self._twiddles = {}

        def _scram(n):
            if n <= _FFT_NATIVE_MAX or (n & (n - 1)) == 0:
                return (1, n)
            return _split_n(n)

        # slab-mode lateral FFTs skip their final un-scramble transpose;
        # the gather kernel absorbs the separable index permutation
        self.scramA = (1, self.n_up[0])      # row axis: full four-step
        self.scramB = _scram(self.n_up[1])   # last axis: scrambled
        if self.fft_backend == "vkfft":
            # VkFFT emits the lateral FFT in NATURAL order (both axes), so the
            # gather must not de-scramble: force identity on both axes.
            self.scramA = (1, self.n_up[0])
            self.scramB = (1, self.n_up[1])
        # mlx's memory limit is soft; in low_mem mode return freed buffers
        # to the OS after each stage so the pool stays ~2 live grids
        if low_mem is None:
            grid_bytes = np.prod([float(n) for n in self.n_up]) * 8
            low_mem = (2.0 * grid_bytes
                       > _LOW_MEM_FRAC * _total_ram_bytes())
        self.low_mem = bool(low_mem)
        self._kernels_P = None
        self._df64_setup = None
        self.set_sources(x)   # source-side state (+ kernel build)

    # ------------------------------------------------------------------
    def set_sources(self, x, backend="host"):
        """Per-frame source update against the frozen grid/target setup.

        The grid (nf, gam, C, D), target-side arrays, deconvolution and
        compiled kernels are reused; only the source-side state (cell
        indices, fractions, prephase, sort order) is rebuilt. New sources
        must lie within the extents the plan was sized for (plus half the
        kernel guard band) or a ValueError is raised — rebuild the plan in
        that case.

        backend="host": fp64 numpy (crit64-grade), ~tens of ms at P=1e5.
        backend="gpu":  df64 (double-single) Metal kernel + GPU sort,
                        crit64-grade, ~ms at P=1e5.
        """
        x64 = [np.asarray(v, dtype=np.float64) for v in x]
        P = x64[0].size
        for d in range(3):
            lo, hi = self._xext[d]
            guard = 0.45 * (self.w + 1) * self.gam[d] * (2.0 * PI
                                                         / self.nf[d])
            if x64[d].min() < lo - guard or x64[d].max() > hi + guard:
                raise ValueError(
                    f"set_sources: dim {d} sources exceed the frozen plan "
                    f"extent [{lo:.4g}, {hi:.4g}] (+guard {guard:.3g}); "
                    "rebuild the plan for this geometry")
        if P != self._kernels_P:
            self.P = P
            self._build_kernels()
            self._kernels_P = P

        if backend == "host":
            rdt, pdt = self._rdt, self._pdt
            w = self.w
            xs = [v.astype(rdt) for v in x64]
            i1, fr = [], []
            for d in range(3):
                hd = 2.0 * PI / self.nf[d]
                xi = ((xs[d] - rdt(self.C[d])) / rdt(self.gam[d]) / rdt(hd)
                      + rdt(self.nf[d] / 2.0))
                ii = np.ceil(xi - rdt(w / 2.0)).astype(np.int32)
                i1.append(ii)
                fr.append((xi - ii.astype(rdt)).astype(np.float32))
            xp = [v.astype(pdt) for v in x64]
            pre_ang = sum(pdt(self.D[d]) * xp[d] for d in range(3))
            prephase = np.exp(1j * self.isign * pre_ang.astype(np.float64))
            if self._sort_points:
                perm = np.argsort(
                    i1[0].astype(np.int64) * self.nf[1] + i1[1],
                    kind="stable")
            else:
                perm = np.arange(P)
            self.perm = perm
            self.i1 = [v[perm] for v in i1]
            self.fr = [v[perm] for v in fr]
            self.mx_perm = mx.array(perm.astype(np.uint32))
            self.mx_i1 = [mx.array(v) for v in self.i1]
            self.mx_fr = [mx.array(v) for v in self.fr]
            self.mx_pre = mx.array(prephase[perm].astype(np.complex64))
        elif backend == "gpu":
            self._set_sources_gpu(x64, P)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        mx.eval(self.mx_perm, self.mx_pre, *self.mx_i1, *self.mx_fr)

    def _set_sources_gpu(self, x64, P):
        """df64 (double-single) Metal source setup: rescale + prephase, then
        GPU sort. Validated against the host fp64 path (REPORT df64 demo)."""
        if self._df64_setup is None:
            self._df64_setup = _build_df64_setup_kernel()
        cst = np.zeros(28, dtype=np.float32)
        for d in range(3):
            hd = 2.0 * PI / self.nf[d]
            invgh = 1.0 / (self.gam[d] * hd)
            base = d * 8
            cst[base + 0] = np.float32(self.C[d])
            cst[base + 1] = np.float32(self.C[d] - np.float64(cst[base + 0]))
            cst[base + 2] = np.float32(invgh)
            cst[base + 3] = np.float32(invgh - np.float64(cst[base + 2]))
            cst[base + 4] = self.nf[d] / 2.0
            cst[base + 5] = self.w / 2.0
            cst[base + 6] = np.float32(self.D[d])
            cst[base + 7] = np.float32(self.D[d] - np.float64(cst[base + 6]))
        cst[24] = np.float32(2.0 * PI)
        cst[25] = np.float32(2.0 * PI - np.float64(cst[24]))
        cst[26] = 1.0 / (2.0 * PI)
        cst[27] = float(self.isign)
        ins = []
        for d in range(3):
            hi = x64[d].astype(np.float32)
            lo = (x64[d] - hi).astype(np.float32)
            ins += [mx.array(hi), mx.array(lo)]
        ins += [mx.array(cst), mx.array(np.array([P], dtype=np.int32))]
        outs = self._df64_setup(
            inputs=ins,
            output_shapes=[(P,), (P,), (P,), (P,), (P,), (P,), (2 * P,)],
            output_dtypes=[mx.int32, mx.int32, mx.int32,
                           mx.float32, mx.float32, mx.float32, mx.float32],
            grid=(P, 1, 1), threadgroup=(256, 1, 1))
        i10, i11, i12, fr0, fr1, fr2, pref = outs
        pre = mx.view(pref.reshape(P, 2), dtype=mx.complex64).reshape(P)
        if self._sort_points:
            key = i10 * self.nf[1] + i11
            perm = mx.argsort(key)
        else:
            perm = mx.arange(P, dtype=mx.uint32)
        self.mx_perm = perm.astype(mx.uint32)
        self.mx_i1 = [mx.take(v, perm) for v in (i10, i11, i12)]
        self.mx_fr = [mx.take(v, perm) for v in (fr0, fr1, fr2)]
        self.mx_pre = mx.take(pre, perm)
        self.perm = None     # numpy mirrors not materialized on this path
        self.i1 = None
        self.fr = None

    # ------------------------------------------------------------------
    def _build_kernels(self):
        w, w2 = self.w, self.w2
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up
        es1 = _es_msl("k1", w, self.beta)
        es2 = _es_msl("k2", w2, self.beta2)

        spread_src = f"""
    uint lane = thread_position_in_grid.x;   // 0..W*W-1
    uint j = thread_position_in_grid.y;      // point index (sorted order)
    if (lane >= {w * w}u || j >= {self.P}u) return;
    int lx = (int)(lane / {w}u), ly = (int)(lane % {w}u);
    float wxy = k1_es((float)lx - frx[j]) * k1_es((float)ly - fry[j]);
    float cre = cj[2*j] * wxy, cim = cj[2*j+1] * wxy;
    int ix = i1x[j] + lx;  ix -= {nf1} * (ix >= {nf1});  ix += {nf1} * (ix < 0);
    int iy = i1y[j] + ly;  iy -= {nf2} * (iy >= {nf2});  iy += {nf2} * (iy < 0);
    size_t base = ((size_t)ix * {nf2} + (size_t)iy) * {nf3};
    int iz0 = i1z[j];
    float fz = frz[j];
    for (int lz = 0; lz < {w}; ++lz) {{
        float wz = k1_es((float)lz - fz);
        int iz = iz0 + lz;
        iz -= {nf3} * (iz >= {nf3});  iz += {nf3} * (iz < 0);
        atomic_fetch_add_explicit(&grid[2*(base + (size_t)iz)],
                                  cre * wz, memory_order_relaxed);
        atomic_fetch_add_explicit(&grid[2*(base + (size_t)iz) + 1],
                                  cim * wz, memory_order_relaxed);
    }}
"""
        self._spread = mx.fast.metal_kernel(
            name="t3spread", input_names=["cj", "i1x", "i1y", "i1z",
                                          "frx", "fry", "frz"],
            output_names=["grid"], header="#include <metal_math>\n" + es1,
            source=spread_src, atomic_outputs=True)

        # complex64-typed slab spread: identical atomic float adds, but the
        # output buffer is counted in COMPLEX elements (nf1*nf2*nf3) instead of
        # float-interleaved (2x), halving the mx.fast.metal_kernel int32
        # element count and ~doubling the large-grid ceiling (n_up ~9.7k ->
        # ~13.6k). Bit-identical to the float32 spread; used by the slab path
        # only (the non-slab path keeps `_spread` for its small grids). The
        # atomic adds target the same memory via a `device atomic_float*` cast.
        spread_c_src = f"""
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    if (lane >= {w * w}u || j >= {self.P}u) return;
    device atomic_float* g = (device atomic_float*) grid;
    int lx = (int)(lane / {w}u), ly = (int)(lane % {w}u);
    float wxy = k1_es((float)lx - frx[j]) * k1_es((float)ly - fry[j]);
    float cre = cj[2*j] * wxy, cim = cj[2*j+1] * wxy;
    int ix = i1x[j] + lx;  ix -= {nf1} * (ix >= {nf1});  ix += {nf1} * (ix < 0);
    int iy = i1y[j] + ly;  iy -= {nf2} * (iy >= {nf2});  iy += {nf2} * (iy < 0);
    size_t base = ((size_t)ix * {nf2} + (size_t)iy) * {nf3};
    int iz0 = i1z[j];
    float fz = frz[j];
    for (int lz = 0; lz < {w}; ++lz) {{
        float wz = k1_es((float)lz - fz);
        int iz = iz0 + lz;
        iz -= {nf3} * (iz >= {nf3});  iz += {nf3} * (iz < 0);
        atomic_fetch_add_explicit(&g[2*(base + (size_t)iz)],
                                  cre * wz, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[2*(base + (size_t)iz) + 1],
                                  cim * wz, memory_order_relaxed);
    }}
"""
        self._spread_c = mx.fast.metal_kernel(
            name="t3spread_c", input_names=["cj", "i1x", "i1y", "i1z",
                                            "frx", "fry", "frz"],
            output_names=["grid"], header="#include <metal_math>\n" + es1,
            source=spread_c_src, atomic_outputs=True)

        # thread per OUTPUT cell of the padded array: no init pass needed
        pad_src = f"""
    uint r3 = thread_position_in_grid.x;
    uint r2 = thread_position_in_grid.y;
    uint r1 = thread_position_in_grid.z;
    if (r3 >= {nu3}u || r2 >= {nu2}u || r1 >= {nu1}u) return;
    size_t dst = ((size_t)r1 * {nu2} + r2) * {nu3} + r3;
    int q1 = (int)r1;  q1 -= {nu1} * (q1 >= {nu1 - nf1 // 2});
    int q2 = (int)r2;  q2 -= {nu2} * (q2 >= {nu2 - nf2 // 2});
    int q3 = (int)r3;  q3 -= {nu3} * (q3 >= {nu3 - nf3 // 2});
    bool inband = (q1 >= {-(nf1 // 2)} && q1 < {nf1 // 2}
                && q2 >= {-(nf2 // 2)} && q2 < {nf2 // 2}
                && q3 >= {-(nf3 // 2)} && q3 < {nf3 // 2});
    if (!inband) {{ H[2*dst] = 0.0f; H[2*dst+1] = 0.0f; return; }}
    int m1 = q1 + {nf1 // 2}, m2 = q2 + {nf2 // 2}, m3 = q3 + {nf3 // 2};
    size_t src = ((size_t)m1 * {nf2} + m2) * {nf3} + m3;
    float d = dec1[m1] * dec2[m2] * dec3[m3];
    H[2*dst]   = bf[2*src] * d;
    H[2*dst+1] = bf[2*src+1] * d;
"""
        self._pad = mx.fast.metal_kernel(
            name="t3pad", input_names=["bf", "dec1", "dec2", "dec3"],
            output_names=["H"], source=pad_src)

        gather_src = f"""
    uint kk = thread_position_in_grid.x;
    if (kk >= {self.M}u) return;
    float wx[{w2}], wy[{w2}], wz[{w2}];
    int jx[{w2}], jy[{w2}], jz[{w2}];
    float fx = tfrx[kk], fy = tfry[kk], fz = tfrz[kk];
    int x0 = ti1x[kk], y0 = ti1y[kk], z0 = ti1z[kk];
    for (int l = 0; l < {w2}; ++l) {{
        wx[l] = k2_es((float)l - fx);
        wy[l] = k2_es((float)l - fy);
        wz[l] = k2_es((float)l - fz);
        int a = x0 + l; a -= {nu1} * (a >= {nu1}); a += {nu1} * (a < 0); jx[l] = a;
        int b = y0 + l; b -= {nu2} * (b >= {nu2}); b += {nu2} * (b < 0); jy[l] = b;
        int c = z0 + l; c -= {nu3} * (c >= {nu3}); c += {nu3} * (c < 0); jz[l] = c;
    }}
    float accre = 0.0f, accim = 0.0f;
    for (int lx = 0; lx < {w2}; ++lx) {{
        for (int ly = 0; ly < {w2}; ++ly) {{
            float wxy = wx[lx] * wy[ly];
            size_t base = ((size_t)jx[lx] * {nu2} + (size_t)jy[ly]) * {nu3};
            float sre = 0.0f, sim = 0.0f;
            for (int lz = 0; lz < {w2}; ++lz) {{
                float wv = wz[lz];
                size_t idx = base + (size_t)jz[lz];
                sre = metal::fma(v[2*idx], wv, sre);
                sim = metal::fma(v[2*idx+1], wv, sim);
            }}
            accre = metal::fma(sre, wxy, accre);
            accim = metal::fma(sim, wxy, accim);
        }}
    }}
    float pre_ = post[2*kk], pim_ = post[2*kk+1];
    out[2*kk]   = accre * pre_ - accim * pim_;
    out[2*kk+1] = accre * pim_ + accim * pre_;
"""
        self._gather = mx.fast.metal_kernel(
            name="t3gather",
            input_names=["v", "ti1x", "ti1y", "ti1z",
                         "tfrx", "tfry", "tfrz", "post"],
            output_names=["out"], header="#include <metal_math>\n" + es2,
            source=gather_src)

        # ---- slab-mode kernels -------------------------------------------
        # pad + mode-deconvolve + brute z-DFT, writing z-major Z[l3, r1, r2]
        padz_src = f"""
    uint r2 = thread_position_in_grid.x;
    uint r1 = thread_position_in_grid.y;
    if (r2 >= {nu2}u || r1 >= {nu1}u) return;
    device float* Zf = (device float*) Z;     // complex64 out, write interleaved
    int q1 = (int)r1;  q1 -= {nu1} * (q1 >= {nu1 - nf1 // 2});
    int q2 = (int)r2;  q2 -= {nu2} * (q2 >= {nu2 - nf2 // 2});
    bool inband = (q1 >= {-(nf1 // 2)} && q1 < {nf1 // 2}
                && q2 >= {-(nf2 // 2)} && q2 < {nf2 // 2});
    if (!inband) {{
        for (int l3 = 0; l3 < {nu3}; ++l3) {{
            size_t dst = ((size_t)l3 * {nu1} + r1) * {nu2} + r2;
            Zf[2*dst] = 0.0f; Zf[2*dst+1] = 0.0f;
        }}
        return;
    }}
    int m1 = q1 + {nf1 // 2}, m2 = q2 + {nf2 // 2};
    float d12 = dec1[m1] * dec2[m2];
    size_t src0 = ((size_t)m1 * {nf2} + m2) * {nf3};
    float bre[{nf3}], bim[{nf3}];
    for (int m3 = 0; m3 < {nf3}; ++m3) {{
        float d = dec3[m3];
        bre[m3] = bf[src0 + m3].real * d;
        bim[m3] = bf[src0 + m3].imag * d;
    }}
    for (int l3 = 0; l3 < {nu3}; ++l3) {{
        float accre = 0.0f, accim = 0.0f;
        for (int m3 = 0; m3 < {nf3}; ++m3) {{
            float twre = tw[2*(l3*{nf3} + m3)];
            float twim = tw[2*(l3*{nf3} + m3) + 1];
            accre = metal::fma(bre[m3], twre, metal::fma(-bim[m3], twim, accre));
            accim = metal::fma(bre[m3], twim, metal::fma( bim[m3], twre, accim));
        }}
        size_t dst = ((size_t)l3 * {nu1} + r1) * {nu2} + r2;
        Zf[2*dst]   = accre * d12;
        Zf[2*dst+1] = accim * d12;
    }}
"""
        # bf is the complex64 spread grid (see _spread_c); read via .real/.imag.
        # Z is complex64 (nu3*nu1*nu2 elems, not 2x): keeps the full (un-chunked)
        # slab path valid to ~n_up 9460/axis at nu3=24, so the common 4K-class
        # sizes take one padz + one FFT instead of z-chunking.
        self._padz = mx.fast.metal_kernel(
            name="t3padz", input_names=["bf", "dec1", "dec2", "dec3", "tw"],
            output_names=["Z"], source=padz_src)
        self._padz_chunks = {}

        # per-slab gather: accumulate this slab's z-weighted xy-interp
        gslab_src = f"""
    uint kk = thread_position_in_grid.x;
    if (kk >= {self.M}u) return;
    int kz = kzbuf[0];
    int z0 = ti1z[kk];
    int l = kz - z0;
    l += {nu3} * (l < 0); l -= {nu3} * (l >= {nu3});
    float acre = accin[2*kk], acim = accin[2*kk+1];
    if (l < 0 || l >= {w2}) {{ accout[2*kk] = acre; accout[2*kk+1] = acim; return; }}
    float wz = k2_es((float)l - tfrz[kk]);
    float wx[{w2}], wy[{w2}];
    int jx[{w2}], jy[{w2}];
    float fx = tfrx[kk], fy = tfry[kk];
    int x0 = ti1x[kk], y0 = ti1y[kk];
    for (int t = 0; t < {w2}; ++t) {{
        wx[t] = k2_es((float)t - fx);
        wy[t] = k2_es((float)t - fy);
        int a = x0 + t; a -= {nu1} * (a >= {nu1}); a += {nu1} * (a < 0);
        jx[t] = (a % {self.scramA[0]}) * {self.scramA[1]} + a / {self.scramA[0]};
        int b = y0 + t; b -= {nu2} * (b >= {nu2}); b += {nu2} * (b < 0);
        jy[t] = (b % {self.scramB[0]}) * {self.scramB[1]} + b / {self.scramB[0]};
    }}
    float sre = 0.0f, sim = 0.0f;
    for (int lx = 0; lx < {w2}; ++lx) {{
        size_t base = (size_t)jx[lx] * {nu2};
        float pre = 0.0f, pim = 0.0f;
        for (int ly = 0; ly < {w2}; ++ly) {{
            float wv = wy[ly];
            size_t idx = base + (size_t)jy[ly];
            pre = metal::fma(vk[2*idx], wv, pre);
            pim = metal::fma(vk[2*idx+1], wv, pim);
        }}
        sre = metal::fma(pre, wx[lx], sre);
        sim = metal::fma(pim, wx[lx], sim);
    }}
    accout[2*kk]   = acre + sre * wz;
    accout[2*kk+1] = acim + sim * wz;
"""
        self._gather_slab = mx.fast.metal_kernel(
            name="t3gatherslab",
            input_names=["vk", "accin", "kzbuf", "ti1x", "ti1y", "ti1z",
                         "tfrx", "tfry", "tfrz"],
            output_names=["accout"], header="#include <metal_math>\n" + es2,
            source=gslab_src)

    # ------------------------------------------------------------------
    def _build_batch_kernels(self, nch):
        """Batched (multi-channel) variants of spread/padz/gather_slab.

        Layouts: spread grid (cell, ch, re/im) — channel-adjacent atomics;
        Z and v (ch, l3, nu1, nu2, re/im) — channel-outermost so per-slab
        FFTs stay contiguous; accumulator (target, ch, re/im).
        """
        w, w2 = self.w, self.w2
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up
        es1 = _es_msl("k1", w, self.beta)
        es2 = _es_msl("k2", w2, self.beta2)

        spread_src = f"""
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    if (lane >= {w * w}u || j >= {self.P}u) return;
    int lx = (int)(lane / {w}u), ly = (int)(lane % {w}u);
    float wxy = k1_es((float)lx - frx[j]) * k1_es((float)ly - fry[j]);
    float cre[{nch}], cim[{nch}];
    for (int ch = 0; ch < {nch}; ++ch) {{
        cre[ch] = cj[(j * {nch} + ch) * 2] * wxy;
        cim[ch] = cj[(j * {nch} + ch) * 2 + 1] * wxy;
    }}
    int ix = i1x[j] + lx;  ix -= {nf1} * (ix >= {nf1});  ix += {nf1} * (ix < 0);
    int iy = i1y[j] + ly;  iy -= {nf2} * (iy >= {nf2});  iy += {nf2} * (iy < 0);
    size_t base = ((size_t)ix * {nf2} + (size_t)iy) * {nf3};
    int iz0 = i1z[j];
    float fz = frz[j];
    for (int lz = 0; lz < {w}; ++lz) {{
        float wz = k1_es((float)lz - fz);
        int iz = iz0 + lz;
        iz -= {nf3} * (iz >= {nf3});  iz += {nf3} * (iz < 0);
        size_t cell = (base + (size_t)iz) * {nch};
        for (int ch = 0; ch < {nch}; ++ch) {{
            atomic_fetch_add_explicit(&grid[(cell + ch) * 2],
                                      cre[ch] * wz, memory_order_relaxed);
            atomic_fetch_add_explicit(&grid[(cell + ch) * 2 + 1],
                                      cim[ch] * wz, memory_order_relaxed);
        }}
    }}
"""
        spread = mx.fast.metal_kernel(
            name=f"t3spreadb{nch}",
            input_names=["cj", "i1x", "i1y", "i1z", "frx", "fry", "frz"],
            output_names=["grid"], header="#include <metal_math>\n" + es1,
            source=spread_src, atomic_outputs=True)

        padz_src = f"""
    uint r2 = thread_position_in_grid.x;
    uint r1 = thread_position_in_grid.y;
    if (r2 >= {nu2}u || r1 >= {nu1}u) return;
    int q1 = (int)r1;  q1 -= {nu1} * (q1 >= {nu1 - nf1 // 2});
    int q2 = (int)r2;  q2 -= {nu2} * (q2 >= {nu2 - nf2 // 2});
    bool inband = (q1 >= {-(nf1 // 2)} && q1 < {nf1 // 2}
                && q2 >= {-(nf2 // 2)} && q2 < {nf2 // 2});
    if (!inband) {{
        for (int ch = 0; ch < {nch}; ++ch)
            for (int l3 = 0; l3 < {nu3}; ++l3) {{
                size_t dst = (((size_t)ch * {nu3} + l3) * {nu1} + r1)
                             * {nu2} + r2;
                Z[2*dst] = 0.0f; Z[2*dst+1] = 0.0f;
            }}
        return;
    }}
    int m1 = q1 + {nf1 // 2}, m2 = q2 + {nf2 // 2};
    float d12 = dec1[m1] * dec2[m2];
    size_t src0 = ((size_t)m1 * {nf2} + m2) * {nf3};
    for (int ch = 0; ch < {nch}; ++ch) {{
        float bre[{nf3}], bim[{nf3}];
        for (int m3 = 0; m3 < {nf3}; ++m3) {{
            float d = dec3[m3];
            bre[m3] = bf[((src0 + m3) * {nch} + ch) * 2] * d;
            bim[m3] = bf[((src0 + m3) * {nch} + ch) * 2 + 1] * d;
        }}
        for (int l3 = 0; l3 < {nu3}; ++l3) {{
            float accre = 0.0f, accim = 0.0f;
            for (int m3 = 0; m3 < {nf3}; ++m3) {{
                float twre = tw[2*(l3*{nf3} + m3)];
                float twim = tw[2*(l3*{nf3} + m3) + 1];
                accre = metal::fma(bre[m3], twre,
                                   metal::fma(-bim[m3], twim, accre));
                accim = metal::fma(bre[m3], twim,
                                   metal::fma(bim[m3], twre, accim));
            }}
            size_t dst = (((size_t)ch * {nu3} + l3) * {nu1} + r1)
                         * {nu2} + r2;
            Z[2*dst]   = accre * d12;
            Z[2*dst+1] = accim * d12;
        }}
    }}
"""
        padz = mx.fast.metal_kernel(
            name=f"t3padzb{nch}",
            input_names=["bf", "dec1", "dec2", "dec3", "tw"],
            output_names=["Z"], source=padz_src)

        gslab_src = f"""
    uint kk = thread_position_in_grid.x;
    if (kk >= {self.M}u) return;
    int kz = kzbuf[0];
    int z0 = ti1z[kk];
    int l = kz - z0;
    l += {nu3} * (l < 0); l -= {nu3} * (l >= {nu3});
    if (l < 0 || l >= {w2}) {{
        for (int ch = 0; ch < {nch}; ++ch) {{
            accout[(kk*{nch}+ch)*2]   = accin[(kk*{nch}+ch)*2];
            accout[(kk*{nch}+ch)*2+1] = accin[(kk*{nch}+ch)*2+1];
        }}
        return;
    }}
    float wz = k2_es((float)l - tfrz[kk]);
    float wx[{w2}], wy[{w2}];
    int jx[{w2}], jy[{w2}];
    float fx = tfrx[kk], fy = tfry[kk];
    int x0 = ti1x[kk], y0 = ti1y[kk];
    for (int t = 0; t < {w2}; ++t) {{
        wx[t] = k2_es((float)t - fx);
        wy[t] = k2_es((float)t - fy);
        int a = x0 + t; a -= {nu1} * (a >= {nu1}); a += {nu1} * (a < 0);
        jx[t] = (a % {self.scramA[0]}) * {self.scramA[1]} + a / {self.scramA[0]};
        int b = y0 + t; b -= {nu2} * (b >= {nu2}); b += {nu2} * (b < 0);
        jy[t] = (b % {self.scramB[0]}) * {self.scramB[1]} + b / {self.scramB[0]};
    }}
    float sre[{nch}], sim[{nch}];
    for (int ch = 0; ch < {nch}; ++ch) {{ sre[ch] = 0.0f; sim[ch] = 0.0f; }}
    for (int lx = 0; lx < {w2}; ++lx) {{
        for (int ly = 0; ly < {w2}; ++ly) {{
            float wv = wx[lx] * wy[ly];
            size_t idx = (size_t)jx[lx] * {nu2} + (size_t)jy[ly];
            for (int ch = 0; ch < {nch}; ++ch) {{
                size_t off = ((size_t)ch * {nu1} * {nu2} + idx) * 2;
                sre[ch] = metal::fma(vk[off], wv, sre[ch]);
                sim[ch] = metal::fma(vk[off + 1], wv, sim[ch]);
            }}
        }}
    }}
    for (int ch = 0; ch < {nch}; ++ch) {{
        accout[(kk*{nch}+ch)*2]   = accin[(kk*{nch}+ch)*2]   + sre[ch] * wz;
        accout[(kk*{nch}+ch)*2+1] = accin[(kk*{nch}+ch)*2+1] + sim[ch] * wz;
    }}
"""
        gslab = mx.fast.metal_kernel(
            name=f"t3gatherslabb{nch}",
            input_names=["vk", "accin", "kzbuf", "ti1x", "ti1y", "ti1z",
                         "tfrx", "tfry", "tfrz"],
            output_names=["accout"], header="#include <metal_math>\n" + es2,
            source=gslab_src)
        return dict(spread=spread, padz=padz, gslab=gslab)

    def execute_batch(self, cs, return_np=True):
        """Batched transform of nch strength channels through one plan.

        cs: (nch, P) complex. Returns (nch, M). Slab-mode plans only
        (multiple strength vectors over one fixed geometry); per-channel
        results agree with separate execute() calls within the run-to-run
        atomic noise.
        """
        if not self.slab_mode:
            raise NotImplementedError("execute_batch requires slab mode")
        cs = np.asarray(cs)
        nch, P = cs.shape
        assert P == self.P
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up
        # The fused batch kernels carry an extra factor nch in every buffer
        # (spread nf-grid and the z-major Z), so they cross the metal_kernel
        # 2**31-element limit at far smaller grids than single execute. When
        # either fused buffer would overflow, fall back to per-channel
        # execute() — which is z-chunked and large-grid-safe — and stack. The
        # fused path gives no measured speedup over this anyway (bandwidth-
        # bound; see ACCEPTANCE), so the fallback costs nothing.
        if (nf1 * nf2 * nf3 * nch * 2 > _MK_MAX_ELEMS
                or nu3 * nu1 * nu2 * nch * 2 > _MK_MAX_ELEMS):
            outs = [self.execute(cs[ch], return_np=False) for ch in range(nch)]
            res = mx.stack(outs)
            mx.eval(res)
            return np.array(res) if return_np else res
        if not hasattr(self, "_batch_kernels"):
            self._batch_kernels = {}
        if nch not in self._batch_kernels:
            self._batch_kernels[nch] = self._build_batch_kernels(nch)
        K = self._batch_kernels[nch]
        w = self.w
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up

        def _trim():
            if self.low_mem:
                mx.clear_cache()

        C = mx.array(cs.astype(np.complex64))                  # (nch, P)
        Cp = mx.take(C, self.mx_perm, axis=1) * self.mx_pre[None, :]
        cpf = mx.view(mx.contiguous(mx.transpose(Cp, (1, 0))),
                      dtype=mx.float32).reshape(-1)            # (P,nch,2)
        del C, Cp

        bf = K["spread"](
            inputs=[cpf, self.mx_i1[0], self.mx_i1[1], self.mx_i1[2],
                    self.mx_fr[0], self.mx_fr[1], self.mx_fr[2]],
            output_shapes=[(nf1 * nf2 * nf3 * nch * 2,)],
            output_dtypes=[mx.float32],
            grid=(w * w, self.P, 1), threadgroup=(w * w, 1024 // (w * w), 1),
            init_value=0)[0]
        del cpf
        mx.eval(bf)
        _trim()

        Z = K["padz"](
            inputs=[bf, self.mx_dec[0], self.mx_dec[1], self.mx_dec[2],
                    self.mx_tw.reshape(-1)],
            output_shapes=[(nch * nu3 * nu1 * nu2 * 2,)],
            output_dtypes=[mx.float32],
            grid=(nu2, nu1, 1), threadgroup=(min(nu2, 256), 1, 1))[0]
        del bf
        mx.eval(Z)
        _trim()

        Zc = mx.view(Z, dtype=mx.complex64).reshape(nch, nu3, nu1, nu2)
        acc = mx.zeros((self.M * nch * 2,), dtype=mx.float32)
        mx.eval(acc)
        inv = self.isign > 0
        for kz in range(nu3):
            vk = Zc[:, kz]                          # (nch, nu1, nu2)
            vk, sB = fft_axis_scrambled(vk, 2, inverse=inv,
                                        twiddle_cache=self._twiddles)
            assert sB == self.scramB
            vk = fft_axis(vk, 1, inverse=inv,
                          twiddle_cache=self._twiddles)
            vkf = mx.view(mx.contiguous(vk), dtype=mx.float32).reshape(-1)
            kzbuf = mx.array(np.array([kz], dtype=np.int32))
            acc = K["gslab"](
                inputs=[vkf, acc, kzbuf, self.mx_ti1[0], self.mx_ti1[1],
                        self.mx_ti1[2], self.mx_tfr[0], self.mx_tfr[1],
                        self.mx_tfr[2]],
                output_shapes=[(self.M * nch * 2,)],
                output_dtypes=[mx.float32],
                grid=(self.M, 1, 1), threadgroup=(256, 1, 1))[0]
            del vk, vkf
            mx.eval(acc)
            if self.low_mem and kz % 6 == 5:
                mx.clear_cache()
        del Zc, Z

        out = mx.view(acc.reshape(self.M, nch, 2),
                      dtype=mx.complex64).reshape(self.M, nch)
        res = mx.transpose(out * self.mx_post_c[:, None], (1, 0))
        mx.eval(res)
        if return_np:
            return np.array(res)
        return res

    # ------------------------------------------------------------------
    def _stages(self, c):
        """Lazy mx graph for one transform; yields (label, array) stages."""
        w = self.w
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up

        cmx = mx.array(np.asarray(c).astype(np.complex64)) \
            if not isinstance(c, mx.array) else c
        assert cmx.size == self.P, \
            f"c.size ({cmx.size}) must equal number of sources ({self.P})"
        cp = mx.take(cmx, self.mx_perm) * self.mx_pre
        cpf = mx.view(cp, dtype=mx.float32)
        yield "prephase", cpf

        bf = self._spread(
            inputs=[cpf, self.mx_i1[0], self.mx_i1[1], self.mx_i1[2],
                    self.mx_fr[0], self.mx_fr[1], self.mx_fr[2]],
            output_shapes=[(nf1 * nf2 * nf3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(w * w, self.P, 1), threadgroup=(w * w, 1024 // (w * w), 1),
            init_value=0)[0]
        yield "spread", bf

        Hf = self._pad(
            inputs=[bf, self.mx_dec[0], self.mx_dec[1], self.mx_dec[2]],
            output_shapes=[(nu1 * nu2 * nu3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(nu3, nu2, nu1), threadgroup=(nu3 if nu3 <= 32 else 32,
                                               1024 // min(nu3, 32), 1))[0]
        yield "pad+deconv", Hf

        H = mx.view(Hf, dtype=mx.complex64).reshape(nu1, nu2, nu3)
        # axis-by-axis keeps peak memory at ~2x the inner grid
        for ax in (2, 1, 0):
            H = fft_axis(H, ax, inverse=self.isign > 0,
                         twiddle_cache=self._twiddles)
        vf = mx.view(H, dtype=mx.float32).reshape(-1)
        yield "fft", vf

        out = self._gather(
            inputs=[vf, self.mx_ti1[0], self.mx_ti1[1], self.mx_ti1[2],
                    self.mx_tfr[0], self.mx_tfr[1], self.mx_tfr[2],
                    self.mx_post.reshape(-1)],
            output_shapes=[(self.M * 2,)],
            output_dtypes=[mx.float32],
            grid=(self.M, 1, 1), threadgroup=(256, 1, 1))[0]
        res = mx.view(out, dtype=mx.complex64)
        yield "gather", res

    def _get_padz_chunk(self, kzc):
        """padz variant that writes only z-slabs [kz0, kz0+kzc) into a
        (kzc, nu1, nu2) buffer, with kz0 passed at call time. Identical
        arithmetic to the full padz (same dec/twiddle/inband math); only the
        z-loop range and the destination offset differ, so the slab pipeline
        stays under the metal_kernel 2**31-element limit at large lateral
        grids. Cached per chunk size."""
        k = self._padz_chunks.get(kzc)
        if k is not None:
            return k
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up
        src = f"""
    uint r2 = thread_position_in_grid.x;
    uint r1 = thread_position_in_grid.y;
    if (r2 >= {nu2}u || r1 >= {nu1}u) return;
    device float* Zf = (device float*) Z;     // complex64 out, write interleaved
    int kz0 = kz0buf[0];
    int q1 = (int)r1;  q1 -= {nu1} * (q1 >= {nu1 - nf1 // 2});
    int q2 = (int)r2;  q2 -= {nu2} * (q2 >= {nu2 - nf2 // 2});
    bool inband = (q1 >= {-(nf1 // 2)} && q1 < {nf1 // 2}
                && q2 >= {-(nf2 // 2)} && q2 < {nf2 // 2});
    if (!inband) {{
        for (int l3c = 0; l3c < {kzc}; ++l3c) {{
            size_t dst = ((size_t)l3c * {nu1} + r1) * {nu2} + r2;
            Zf[2*dst] = 0.0f; Zf[2*dst+1] = 0.0f;
        }}
        return;
    }}
    int m1 = q1 + {nf1 // 2}, m2 = q2 + {nf2 // 2};
    float d12 = dec1[m1] * dec2[m2];
    size_t src0 = ((size_t)m1 * {nf2} + m2) * {nf3};
    float bre[{nf3}], bim[{nf3}];
    for (int m3 = 0; m3 < {nf3}; ++m3) {{
        float d = dec3[m3];
        bre[m3] = bf[src0 + m3].real * d;
        bim[m3] = bf[src0 + m3].imag * d;
    }}
    for (int l3c = 0; l3c < {kzc}; ++l3c) {{
        int l3 = kz0 + l3c;
        float accre = 0.0f, accim = 0.0f;
        for (int m3 = 0; m3 < {nf3}; ++m3) {{
            float twre = tw[2*(l3*{nf3} + m3)];
            float twim = tw[2*(l3*{nf3} + m3) + 1];
            accre = metal::fma(bre[m3], twre, metal::fma(-bim[m3], twim, accre));
            accim = metal::fma(bre[m3], twim, metal::fma( bim[m3], twre, accim));
        }}
        size_t dst = ((size_t)l3c * {nu1} + r1) * {nu2} + r2;
        Zf[2*dst]   = accre * d12;
        Zf[2*dst+1] = accim * d12;
    }}
"""
        k = mx.fast.metal_kernel(
            name=f"t3padz_chunk{kzc}",
            input_names=["bf", "dec1", "dec2", "dec3", "tw", "kz0buf"],
            output_names=["Z"], source=src)
        self._padz_chunks[kzc] = k
        return k

    def _execute_slab(self, c, return_np=True):
        """Slab pipeline: fused pad+zDFT, per-z-slab lateral FFTs, gather
        accumulated over slabs. Peak memory ~ nf-grid + z-major inner grid
        + one slab, instead of 2x the full inner grid. At large lateral grids
        the z-major grid exceeds the metal_kernel 2**31-element limit, so padz
        is produced in z-chunks (each under the limit); results are identical
        to the single-call path at sizes where the latter fits."""
        w = self.w
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up

        cmx = mx.array(np.asarray(c).astype(np.complex64)) \
            if not isinstance(c, mx.array) else c
        assert cmx.size == self.P, \
            f"c.size ({cmx.size}) must equal number of sources ({self.P})"
        cp = mx.take(cmx, self.mx_perm) * self.mx_pre
        cpf = mx.view(cp, dtype=mx.float32)

        # complex64-typed spread grid: nf1*nf2*nf3 elements (not 2x), so the
        # nf grid stays under the metal_kernel int32 cap to ~n_up 13.6k/axis.
        bf = self._spread_c(
            inputs=[cpf, self.mx_i1[0], self.mx_i1[1], self.mx_i1[2],
                    self.mx_fr[0], self.mx_fr[1], self.mx_fr[2]],
            output_shapes=[(nf1 * nf2 * nf3,)],
            output_dtypes=[mx.complex64],
            grid=(w * w, self.P, 1), threadgroup=(w * w, 1024 // (w * w), 1),
            init_value=0)[0]
        del cp, cpf
        mx.eval(bf)
        if self.low_mem:
            mx.clear_cache()

        acc = mx.zeros((self.M * 2,), dtype=mx.float32)
        mx.eval(acc)
        inv = self.isign > 0

        vkfft = self.fft_backend == "vkfft"
        if vkfft:
            from . import vkfft_backend as _vk
            vk_inv = 1 if inv else -1          # MLX ifft convention <-> VkFFT
            vk_norm = 1 if inv else 0

        def _gather_one(vk, kz, acc):
            # vk already FFT'd (natural order); gather only.
            vkf = mx.view(vk, dtype=mx.float32).reshape(-1)
            kzbuf = mx.array(np.array([kz], dtype=np.int32))
            return self._gather_slab(
                inputs=[vkf, acc, kzbuf, self.mx_ti1[0], self.mx_ti1[1],
                        self.mx_ti1[2], self.mx_tfr[0], self.mx_tfr[1],
                        self.mx_tfr[2]],
                output_shapes=[(self.M * 2,)],
                output_dtypes=[mx.float32],
                grid=(self.M, 1, 1), threadgroup=(256, 1, 1))[0]

        def _slab(vk, kz, acc):
            vk, sB = fft_axis_scrambled(vk, 1, inverse=inv,
                                        twiddle_cache=self._twiddles)
            assert sB == self.scramB
            vk = fft_axis(vk, 0, inverse=inv, twiddle_cache=self._twiddles)
            return _gather_one(vk, kz, acc)

        def _fft_block_vkfft(Zblk):
            # in-place batched 2D FFT over (nu1,nu2) of a (kc,nu1,nu2) block,
            # natural order (gather uses identity scramble for the vkfft path)
            mx.eval(Zblk)
            _vk.fft2_inplace(Zblk, vk_inv, vk_norm)

        dec = [self.mx_dec[0], self.mx_dec[1], self.mx_dec[2]]
        twf = self.mx_tw.reshape(-1)
        if nu3 * nu1 * nu2 <= _MK_MAX_ELEMS:
            # single-call padz: Z is complex64 (nu3*nu1*nu2 elems), so this
            # full path now reaches ~n_up 9460/axis (was ~6700 with float Z),
            # covering 4K-class grids without z-chunking.
            Z = self._padz(
                inputs=[bf, dec[0], dec[1], dec[2], twf],
                output_shapes=[(nu3 * nu1 * nu2,)],
                output_dtypes=[mx.complex64],
                grid=(nu2, nu1, 1),
                threadgroup=(min(nu2, 256), 1, 1))[0]
            del bf
            mx.eval(Z)
            if self.low_mem:
                mx.clear_cache()
            Zc = Z.reshape(nu3, nu1, nu2)
            if vkfft:
                _fft_block_vkfft(Zc)             # all nu3 slabs at once, in place
                for kz in range(nu3):            # chain gathers; eval once (the
                    acc = _gather_one(Zc[kz], kz, acc)   # VkFFT sync already
                mx.eval(acc)                     # drained the queue)
            else:
                for kz in range(nu3):
                    acc = _slab(Zc[kz], kz, acc)
                    mx.eval(acc)
                    if self.low_mem and kz % 6 == 5:
                        mx.clear_cache()
            del Zc, Z
        else:
            # z-chunked padz (complex64): each call writes <= _MK_MAX_ELEMS
            # complex elements.
            per_slab = nu1 * nu2
            maxkzc = max(1, _MK_MAX_ELEMS // per_slab)
            nchunks = (nu3 + maxkzc - 1) // maxkzc
            kzc = (nu3 + nchunks - 1) // nchunks      # balanced chunk size
            for kz0 in range(0, nu3, kzc):
                kc = min(kzc, nu3 - kz0)
                pk = self._get_padz_chunk(kc)
                Zk = pk(
                    inputs=[bf, dec[0], dec[1], dec[2], twf,
                            mx.array(np.array([kz0], dtype=np.int32))],
                    output_shapes=[(kc * nu1 * nu2,)],
                    output_dtypes=[mx.complex64],
                    grid=(nu2, nu1, 1),
                    threadgroup=(min(nu2, 256), 1, 1))[0]
                mx.eval(Zk)
                Zkc = Zk.reshape(kc, nu1, nu2)
                if vkfft:
                    _fft_block_vkfft(Zkc)        # this chunk's slabs, in place
                    for j in range(kc):
                        acc = _gather_one(Zkc[j], kz0 + j, acc)
                    mx.eval(acc)                 # eval once per chunk
                else:
                    for j in range(kc):
                        acc = _slab(Zkc[j], kz0 + j, acc)
                        mx.eval(acc)
                del Zk, Zkc
                if self.low_mem:
                    mx.clear_cache()
            del bf

        res = mx.view(acc, dtype=mx.complex64) * self.mx_post_c
        mx.eval(res)
        if return_np:
            return np.array(res)
        return res

    def execute(self, c, return_np=True):
        """Eager per-stage execution with explicit frees (peak-memory aware)."""
        if self.slab_mode:
            return self._execute_slab(c, return_np=return_np)
        w = self.w
        nf1, nf2, nf3 = self.nf
        nu1, nu2, nu3 = self.n_up
        def _trim():
            if self.low_mem:
                mx.clear_cache()

        cmx = mx.array(np.asarray(c).astype(np.complex64)) \
            if not isinstance(c, mx.array) else c
        assert cmx.size == self.P, \
            f"c.size ({cmx.size}) must equal number of sources ({self.P})"
        cp = mx.take(cmx, self.mx_perm) * self.mx_pre
        cpf = mx.view(cp, dtype=mx.float32)

        bf = self._spread(
            inputs=[cpf, self.mx_i1[0], self.mx_i1[1], self.mx_i1[2],
                    self.mx_fr[0], self.mx_fr[1], self.mx_fr[2]],
            output_shapes=[(nf1 * nf2 * nf3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(w * w, self.P, 1), threadgroup=(w * w, 1024 // (w * w), 1),
            init_value=0)[0]
        del cp, cpf
        mx.eval(bf)
        _trim()

        H = self._pad(
            inputs=[bf, self.mx_dec[0], self.mx_dec[1], self.mx_dec[2]],
            output_shapes=[(nu1 * nu2 * nu3 * 2,)],
            output_dtypes=[mx.float32],
            grid=(nu3, nu2, nu1), threadgroup=(nu3 if nu3 <= 32 else 32,
                                               1024 // min(nu3, 32), 1))[0]
        del bf
        mx.eval(H)
        _trim()

        H = mx.view(H, dtype=mx.complex64).reshape(nu1, nu2, nu3)
        for ax in (2, 1, 0):
            Hn = fft_axis(H, ax, inverse=self.isign > 0,
                          twiddle_cache=self._twiddles)
            del H
            H = Hn
            del Hn
            mx.eval(H)
            _trim()
        vf = mx.view(H, dtype=mx.float32).reshape(-1)

        out = self._gather(
            inputs=[vf, self.mx_ti1[0], self.mx_ti1[1], self.mx_ti1[2],
                    self.mx_tfr[0], self.mx_tfr[1], self.mx_tfr[2],
                    self.mx_post.reshape(-1)],
            output_shapes=[(self.M * 2,)],
            output_dtypes=[mx.float32],
            grid=(self.M, 1, 1), threadgroup=(256, 1, 1))[0]
        del H, vf
        res = mx.view(out, dtype=mx.complex64)
        mx.eval(res)
        if return_np:
            return np.array(res)
        return res

    def execute_profiled(self, c):
        import time as _time
        times = {}
        mx.synchronize()
        t0 = _time.perf_counter()
        res = None
        for label, arr in self._stages(c):
            mx.eval(arr)
            mx.synchronize()
            t = _time.perf_counter()
            times[label] = t - t0
            t0 = t
            res = arr
        return np.array(res), times
