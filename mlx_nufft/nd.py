"""Dimension-templated type-1/2 NUFFT plans on Apple GPU (dims 1, 2, 3).

Generalization of types12.py (3D, even dims only) to 1D/2D/3D with even or
odd mode counts, mirroring FINUFFT conventions (modeord=0):

  type 1:  f[k] = sum_j c[j] exp(i*isign * k . x_j)
  type 2:  c[j] = sum_k f[k] exp(i*isign * k . x_j)

with k_d integer in [-(N_d//2), (N_d-1)//2] (even or odd N_d), x in
[-pi, pi)^d (any values accepted; folded mod 2pi in fp64 at plan time).
fp32 GPU pipeline; precision-critical coordinate handling in fp64 at plan
time ('crit64', default) exactly as in types12/gpu_t3.

Default upsampfac=2.0 (FINUFFT's default): at eps=1e-5 the ES kernel width
is w=6 vs w=9 at sigma=1.25, i.e. w^d-fold fewer spread/interp taps for
2^d-fold grid memory — the right trade on large-unified-memory machines.
Pass upsampfac=1.25 to recover the low-memory sizing.

The 3D specialization of this module reproduces types12.py's algorithm
exactly (same kernels modulo generation); types12.py remains untouched as
the validated reference implementation.
"""

import numpy as np
import mlx.core as mx

from .sizing import kernel_params, kernel_ft, next235even
from .gpu_t3 import _es_msl, fft_axis, _DF64_HDR

PI = np.pi

_AX = ("x", "y", "z")


def _linearize(idx_names, dims):
    """MSL expression linearizing C-order indices idx_names over dims."""
    expr = f"(size_t){idx_names[0]}"
    for name, n in zip(idx_names[1:], dims[1:]):
        expr = f"({expr} * {n} + (size_t){name})"
    return expr


def _build_t1_df64_setup_kernel(dim):
    """Per-point type-1 source setup in double-single (df64) arithmetic: the
    GPU analogue of _PointsND._compute_cells' host fp64 rescale (crit64 grade).

    Strictly simpler than the type-3 df64 setup (gpu_t3._build_df64_setup_kernel)
    — no source-centre C, no gamma, no prephase — but it ADDS a df64 Cody-Waite
    reduction mod(x, 2pi) matching np.mod(x, 2pi), which type-3 does not need
    (type-3 folds the coordinate via its source centre C instead). Forming
    r = x - floor(x/2pi)*2pi in df64 (never fp32) keeps ~fp64 relative accuracy
    for any |x| within the fp32 exponent range, so the ES cell index i1 is
    bit-exact against the host fp64 path and the fraction fr is correct to the
    fp32 representation floor.

    consts layout (2*dim + 4 floats): per dim d at offset 2d
    [invh_hi, invh_lo] with invh = n_up[d]/(2pi); tail at offset 2*dim
    [w/2, 2pi_hi, 2pi_lo, 1/2pi].
    """
    A = 2 * dim
    body = [f"""
    uint j = thread_position_in_grid.x;
    if (j >= (uint)P0[0]) return;
    df64 twopi = df_make(cst[{A + 1}], cst[{A + 2}]);
"""]
    for d in range(dim):
        body.append(f"""
    {{
        df64 x = df_make(xh{d}[j], xl{d}[j]);
        df64 q = df_mul(x, df_make(cst[{A + 3}], 0.0f));      // x / 2pi
        float kf = metal::floor(q.hi + q.lo);
        df64 r = df_add(x, df_mul(df_make(-kf, 0.0f), twopi));   // x - kf*2pi
        if (r.hi + r.lo < 0.0f)          r = df_add(r, twopi);
        if (r.hi + r.lo >= cst[{A + 1}]) r = df_add(r, df_make(-cst[{A + 1}], -cst[{A + 2}]));
        df64 xi = df_mul(r, df_make(cst[{2 * d}], cst[{2 * d + 1}]));   // r * invh
        df64 a = df_add(xi, df_make(-cst[{A}], 0.0f));                  // xi - w/2
        float i1f = metal::ceil(a.hi);
        if (a.hi - i1f + a.lo > 0.0f) i1f += 1.0f;                      // exact-tie ceil
        i1o{d}[j] = (int)i1f;
        fro{d}[j] = (a.hi - i1f) + a.lo + cst[{A}];                     // xi - i1
    }}
""")
    innames = []
    for d in range(dim):
        innames += [f"xh{d}", f"xl{d}"]
    innames += ["cst", "P0"]
    outnames = ([f"i1o{d}" for d in range(dim)]
                + [f"fro{d}" for d in range(dim)])
    return mx.fast.metal_kernel(
        name=f"t1df64setup{dim}d", input_names=innames,
        output_names=outnames, header=_DF64_HDR, source="".join(body))


class _PointsND:
    """Shared point/kernel setup for ND types 1 and 2."""

    def __init__(self, x, n_modes, eps, isign, upsampfac, prec, sort_points):
        if prec not in ("fp32", "crit64"):
            raise ValueError("prec must be 'fp32' or 'crit64'")
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = (x,)
        x = tuple(x)
        self.dim = len(x)
        if self.dim not in (1, 2, 3):
            raise ValueError("dims 1, 2, 3 supported")
        if np.isscalar(n_modes):
            n_modes = (int(n_modes),) * self.dim
        self.N = tuple(int(n) for n in n_modes)
        if len(self.N) != self.dim:
            raise ValueError(f"n_modes length {len(self.N)} must match "
                             f"dim {self.dim}")
        if any(n < 1 for n in self.N):
            raise ValueError("mode dims must be >= 1")
        if upsampfac in (None, 0, 0.0):       # finufft auto sentinel
            upsampfac = 2.0
        if not 1.0 < upsampfac <= 4.0:
            raise ValueError(f"upsampfac must be in (1, 4], got {upsampfac}")
        self.prec = prec
        self.isign = +1 if isign >= 0 else -1   # finufft: non-negative -> +
        self.eps = eps
        self.sigma = upsampfac
        self.w, self.beta = kernel_params(eps, upsampfac)
        w = self.w
        self.n_up = [next235even(max(2 * w, int(np.ceil(upsampfac * n))))
                     for n in self.N]

        x64 = [np.asarray(v, dtype=np.float64).ravel() for v in x]
        self.P = x64[0].size
        if self.P < 1:
            raise ValueError("at least one nonuniform point is required")
        if not all(v.size == self.P for v in x64):
            raise ValueError("coordinate arrays must have equal length")
        self._sort_points = bool(sort_points)
        self._set_points_arrays(x64)

        # mode deconvolution: 1/phihat(2 pi k / n_up); uncentered fine grid
        # (u_l = l*h, x folded into [0, 2pi)) => no (-1)^k half-grid factor.
        # FFT normalization for isign=+1 (mx ifft includes 1/n per axis)
        # folded into dim 0.
        decs = []
        for d in range(self.dim):
            k = np.arange(-(self.N[d] // 2),
                          self.N[d] - self.N[d] // 2, dtype=np.float64)
            ph = kernel_ft(2.0 * PI * k / self.n_up[d], self.beta, w)
            decs.append(1.0 / ph)
        if self.isign > 0:
            decs[0] = decs[0] * float(np.prod([float(n) for n in self.n_up]))
        self.decs = decs
        self.mx_dec = [mx.array(v.astype(np.float32)) for v in decs]
        self._twiddles = {}

    def _compute_cells(self, x64):
        """Host fp64/fp32 ES cell index + fraction per dim for the given fp64
        coordinates. No sort, no GPU upload — sets self.i1 / self.fr (numpy).
        Geometry/mode state (n_up, kernel, deconvolution, twiddles, compiled
        kernels) is untouched."""
        rdt = np.float64 if self.prec == "crit64" else np.float32
        w = self.w
        self.i1, self.fr = [], []
        for d in range(self.dim):
            h = 2.0 * PI / self.n_up[d]
            xi = (np.mod(x64[d], 2.0 * PI).astype(rdt) / rdt(h))
            ii = np.ceil(xi - rdt(w / 2.0)).astype(np.int32)
            self.i1.append(ii)
            self.fr.append((xi - ii.astype(rdt)).astype(np.float32))

    def _sort_and_upload(self):
        """Lateral-cell-key sort + mx upload (non-OD / GM path), from the
        host i1/fr in self.i1/self.fr. Sets perm/sorted and the mx arrays."""
        if self._sort_points and self.P > 0:
            if self.dim >= 2:
                key = self.i1[0].astype(np.int64) * self.n_up[1] + self.i1[1]
            else:
                key = self.i1[0].astype(np.int64)
            self.perm = np.argsort(key, kind="stable")
        else:
            self.perm = np.arange(self.P)
        self.sorted = self._sort_points
        self.mx_i1 = [mx.array(v[self.perm]) for v in self.i1]
        self.mx_fr = [mx.array(v[self.perm]) for v in self.fr]
        self.mx_perm = mx.array(self.perm.astype(np.uint32))

    def _set_points_arrays(self, x64):
        """Back-compat: compute cells then the lateral sort + upload (the
        host re-point used by __init__ and set_sources(backend="host"))."""
        self._compute_cells(x64)
        self._sort_and_upload()

    # -- GPU (df64) re-point: cells on a Metal kernel, sort via mx.argsort ---

    def _compute_cells_gpu(self, x64):
        """df64 GPU analogue of _compute_cells: returns (i1_g, fr_g) lists of
        UNSORTED GPU arrays (int32 cell index, float32 fraction) per dim,
        crit64-grade (i1 bit-exact vs host fp64, fr to the fp32 floor). Clears
        the numpy self.i1/self.fr mirrors (this path keeps them on-GPU)."""
        dim, w, P = self.dim, self.w, self.P
        if getattr(self, "_df64_setup", None) is None:
            self._df64_setup = _build_t1_df64_setup_kernel(dim)
        A = 2 * dim
        cst = np.zeros(A + 4, dtype=np.float32)
        for d in range(dim):
            invh = self.n_up[d] / (2.0 * PI)
            cst[2 * d] = np.float32(invh)
            cst[2 * d + 1] = np.float32(invh - np.float64(cst[2 * d]))
        cst[A] = np.float32(w / 2.0)
        cst[A + 1] = np.float32(2.0 * PI)
        cst[A + 2] = np.float32(2.0 * PI - np.float64(cst[A + 1]))
        cst[A + 3] = np.float32(1.0 / (2.0 * PI))
        ins = []
        for d in range(dim):
            hi = x64[d].astype(np.float32)
            lo = (x64[d] - hi).astype(np.float32)
            ins += [mx.array(hi), mx.array(lo)]
        ins += [mx.array(cst), mx.array(np.array([P], dtype=np.int32))]
        outs = self._df64_setup(
            inputs=ins,
            output_shapes=[(P,)] * (2 * dim),
            output_dtypes=[mx.int32] * dim + [mx.float32] * dim,
            grid=(P, 1, 1), threadgroup=(256, 1, 1))
        i1_g, fr_g = list(outs[:dim]), list(outs[dim:])
        self.i1 = self.fr = None
        return i1_g, fr_g

    def _sort_and_upload_gpu(self, i1_g, fr_g):
        """GPU lateral-key sort (mx.argsort) + perm-indexed take, from the
        GPU cells (non-OD / GM path). Materializes the numpy perm mirror."""
        P, dim = self.P, self.dim
        if self._sort_points and P > 0:
            key = i1_g[0]
            if dim >= 2:
                key = i1_g[0] * self.n_up[1] + i1_g[1]
            perm = mx.argsort(key)
        else:
            perm = mx.arange(P, dtype=mx.uint32)
        self.mx_perm = perm.astype(mx.uint32)
        self.mx_i1 = [mx.take(v, self.mx_perm) for v in i1_g]
        self.mx_fr = [mx.take(v, self.mx_perm) for v in fr_g]
        mx.eval(self.mx_perm, *self.mx_i1, *self.mx_fr)
        self.perm = np.array(self.mx_perm).astype(np.intp)
        self.sorted = self._sort_points

    # -- kernel-source builders (shared by both plan types) ---------------

    def _wrap_lines(self, var, base, off, nu):
        """MSL: var = base + off wrapped periodically into [0, nu)."""
        return (f"    int {var} = {base} + {off};  "
                f"{var} -= {nu} * ({var} >= {nu});  "
                f"{var} += {nu} * ({var} < 0);\n")

    def _tg_for(self, gx):
        tx = min(int(gx), 256)
        return (tx, max(1, 256 // tx), 1)

    # -- output-driven (OD) binning: cuFINUFFT-style subproblems -----------
    #
    # Points are bin-sorted; one 256-thread threadgroup processes one
    # subproblem (<= _OD_MSUB points of one bin) against a padded tile of
    # the fine grid staged in threadgroup memory (32 KB on Apple GPUs).
    # Spread: threads parallelize over each point's w^d taps -> tile
    # accumulation needs no atomics (barrier per point); one global atomic
    # add per tile cell at flush. Interp: tile is loaded once, then each
    # thread gathers whole points from threadgroup memory.

    _OD_MSUB = 1024
    _OD_TG = 256

    def _od_tile_dims(self):
        """Per-dim bin sizes m_d; padded tile p_d = m_d + w must fit 28 KB."""
        w = self.w
        if self.dim == 1:
            m = [2048]
        elif self.dim == 2:
            m = [48 - w] * 2
        else:
            m = [15 - w] * 3
        p = [mi + w for mi in m]
        ptot = int(np.prod(p))
        if w > 8 or ptot * 8 > 28 * 1024 or any(mi < 1 for mi in m) \
                or any(nu < pi for nu, pi in zip(self.n_up, p)):
            return None
        return m, p, ptot

    def _od_prepare(self, msub):
        """Bin-sort points (host) and build subproblem tables (subproblems
        capped at msub points each); returns False if the OD path is not
        applicable. Spread uses msub ~1e3 (load balance across threadgroups);
        interp uses whole bins (msub large) so each tile loads exactly once."""
        dims = self._od_tile_dims()
        if dims is None or self.P < 20000:
            return False
        m, p, ptot = dims
        w2 = self.w // 2
        b = [((self.i1[d].astype(np.int64) + w2) // m[d])
             for d in range(self.dim)]
        nb = [int(bd.max()) + 1 if bd.size else 1 for bd in b]
        key = b[0]
        for d in range(1, self.dim):
            key = key * nb[d] + b[d]
        perm = np.argsort(key, kind="stable")
        self.perm = perm
        self.mx_perm = mx.array(perm.astype(np.uint32))
        self.mx_i1 = [mx.array(v[perm]) for v in self.i1]
        self.mx_fr = [mx.array(v[perm]) for v in self.fr]
        self._od_finish(key[perm], msub, nb, m, p, ptot, w2)
        return True

    def _od_prepare_gpu(self, msub, i1_g, fr_g):
        """OD bin-sort on the GPU: df64 cells -> per-dim bins -> mx.argsort,
        perm-indexed take of cells. The subproblem-table build stays host-side
        (it is over bins not points, and _od_nsub is needed CPU-side to size
        the launch grid) from ONE P-length pull of the sorted bin keys — the
        sole device->host sync. Returns False if OD is not applicable."""
        dims = self._od_tile_dims()
        if dims is None or self.P < 20000:
            return False
        m, p, ptot = dims
        w2 = self.w // 2
        b_g = [(i1_g[d] + w2) // m[d] for d in range(self.dim)]
        nb = [int(np.array(mx.max(bd))) + 1 for bd in b_g]
        key_g = b_g[0]
        for d in range(1, self.dim):
            key_g = key_g * nb[d] + b_g[d]
        self.mx_perm = mx.argsort(key_g).astype(mx.uint32)
        self.mx_i1 = [mx.take(v, self.mx_perm) for v in i1_g]
        self.mx_fr = [mx.take(v, self.mx_perm) for v in fr_g]
        key_s = np.array(mx.take(key_g, self.mx_perm))     # one P-length sync
        mx.eval(self.mx_perm, *self.mx_i1, *self.mx_fr)
        self.perm = np.array(self.mx_perm).astype(np.intp)
        self._od_finish(key_s, msub, nb, m, p, ptot, w2)
        return True

    def _od_finish(self, key_s, msub, nb, m, p, ptot, w2):
        """Common OD subproblem-table build from the SORTED bin-key array
        (vectorized over bins, not points — replaces the old per-bin Python
        loop with identical output). Sets self.sorted + the OD launch tables."""
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        counts = np.diff(np.r_[starts, key_s.size])
        nsub_per_bin = (counts + msub - 1) // msub
        total = int(nsub_per_bin.sum())
        bin_id = np.repeat(np.arange(starts.size), nsub_per_bin)
        seg_beg = np.cumsum(nsub_per_bin) - nsub_per_bin
        off = (np.arange(total) - np.repeat(seg_beg, nsub_per_bin)) * msub
        sub_start = (starts[bin_id] + off).astype(np.int32)
        sub_count = np.minimum(msub, counts[bin_id] - off).astype(np.int32)
        sub_key = key_s[starts][bin_id].astype(np.int64)
        # decode bin origin Delta_d = b_d*m_d - floor(w/2) per subproblem
        origins, rem = [], sub_key
        for d in range(self.dim - 1, -1, -1):
            bd = rem % nb[d]
            rem = rem // nb[d]
            origins.insert(0, (bd * m[d] - w2).astype(np.int32))
        self.sorted = True
        self._od_m, self._od_p, self._od_ptot = m, p, ptot
        self._od_nsub = total
        self._mx_sub_start = mx.array(sub_start)
        self._mx_sub_count = mx.array(sub_count)
        self._mx_sub_o = [mx.array(o) for o in origins]

    def _fft_grid(self, Hf):
        H = mx.view(Hf, dtype=mx.complex64).reshape(*self.n_up)
        for ax in range(self.dim - 1, -1, -1):
            Hn = fft_axis(H, ax, inverse=self.isign > 0,
                          twiddle_cache=self._twiddles)
            del H
            H = Hn
            del Hn
        mx.eval(H)
        return H


class Type1PlanND(_PointsND):
    """f[k] = sum_j c[j] exp(i*isign * k . x_j), modeord=0 box, dims 1-3."""

    def __init__(self, x, n_modes, eps=1e-6, isign=+1, upsampfac=2.0,
                 prec="crit64", sort_points=True, spread_method="auto"):
        super().__init__(x, n_modes, eps, isign, upsampfac, prec, sort_points)
        dim, w, P = self.dim, self.w, self.P
        nu = self.n_up
        N = self.N
        es = _es_msl("k1", w, self.beta)

        assert spread_method in ("auto", "od", "gm")
        # OD pays when the tap count keeps a 256-thread group busy (w >= 5);
        # at w <= 4 the direct atomic spread is equally fast and simpler.
        # Preference order: exclusive-ownership OD (no global atomics, no
        # output zero-init) -> padded-tile OD -> GM.
        want_od = (spread_method == "od"
                   or (spread_method == "auto" and w >= 5))
        modx = self._odx_dims() if (want_od and dim == 1) else None
        self._od_ex = modx is not None and self._odx_prepare(modx)
        self._od = ((not self._od_ex) and want_od
                    and self._od_prepare(self._OD_MSUB))
        if spread_method == "od" and not (self._od or self._od_ex):
            raise ValueError("OD spreading not applicable to this geometry")
        if self._od_ex:
            self._spread_ex = self._build_od_spread_ex(es)
        elif self._od:
            self._spread_od = self._build_od_spread(es)

        # ---- spread: lane covers (lx[,ly]) taps; dim 3 loops lz ----------
        self._es = es
        self._lanes = w * w if dim >= 2 else w
        # GM spread is the sliceable path (its baked P is an upper guard, so it
        # runs over any point-subset launched with fewer grid rows) -> reused by
        # execute_disjoint. Built eagerly only when it is the chosen spreader.
        self._spread = None if (self._od or self._od_ex) \
            else self._build_gm_spread()

        # in-band FFT-order rows per axis, in modeord order (negative block
        # first): gather indices for the progressive crop (axes 1..dim-1;
        # axis 0 is cropped by the fused _crop kernel below).
        self._mx_cropidx = [
            mx.array(np.concatenate([
                np.arange(nu[d] - N[d] // 2, nu[d]),
                np.arange(N[d] - N[d] // 2)]).astype(np.int32))
            for d in range(dim)]

        # ---- final axis-0 crop + deconvolve (fused) ----------------------
        # _fft_modes delivers the grid as (B, N[1], .., N[dim-1], nu[0]):
        # every axis but 0 already cropped to modeord order, axis 0 still
        # full-length FFT-order on the contiguous last position. One pass
        # gathers the in-band axis-0 rows and applies the separable mode
        # deconvolution. The batch index rides the spare grid slot (dim 3
        # folds b*N[0]+m1 into z), so one kernel serves any batch size.
        if dim == 3:
            src = f"""
    uint m3 = thread_position_in_grid.x;
    uint m2 = thread_position_in_grid.y;
    uint zz = thread_position_in_grid.z;
    if (m3 >= {N[2]}u || m2 >= {N[1]}u) return;
    uint b = zz / {N[0]}u;  uint m1 = zz % {N[0]}u;
"""
        elif dim == 2:
            src = f"""
    uint m2 = thread_position_in_grid.x;
    uint m1 = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    if (m2 >= {N[1]}u || m1 >= {N[0]}u) return;
"""
        else:
            src = f"""
    uint m1 = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (m1 >= {N[0]}u) return;
"""
        src += (f"    int q1 = (int)m1 - {N[0] // 2};  "
                f"int r1 = q1 + {nu[0]} * (q1 < 0);\n")
        vnames = ["b"] + [f"m{d + 1}" for d in range(1, dim)] + ["r1"]
        vdims = [0] + [N[d] for d in range(1, dim)] + [nu[0]]
        src += f"    size_t src = {_linearize(vnames, vdims)};\n"
        src += (f"    size_t dst = "
                f"{_linearize(['b'] + [f'm{d + 1}' for d in range(dim)], [0] + list(N))};\n")
        src += ("    float d = "
                + " * ".join(f"dec{d + 1}[m{d + 1}]" for d in range(dim))
                + ";\n")
        src += """    fk[2*dst]   = v[2*src] * d;
    fk[2*dst+1] = v[2*src+1] * d;
"""
        self._crop = mx.fast.metal_kernel(
            name=f"t1crop{dim}d",
            input_names=["v"] + [f"dec{d + 1}" for d in range(dim)],
            output_names=["fk"], source=src)

    def _build_gm_spread(self):
        """Global-memory (sorted-atomic) spread kernel. The baked point count
        is an upper guard only, so the same kernel spreads any point-subset
        launched with grid rows = subset size — the basis of execute_disjoint."""
        dim, w, P, nu = self.dim, self.w, self.P, self.n_up
        lanes = self._lanes
        src = f"""
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    if (lane >= {lanes}u || j >= {P}u) return;
"""
        if dim == 1:
            src += """    int lx = (int)lane;
    float wgt = k1_es((float)lx - frx[j]);
"""
        else:
            src += f"""    int lx = (int)(lane / {w}u), ly = (int)(lane % {w}u);
    float wgt = k1_es((float)lx - frx[j]) * k1_es((float)ly - fry[j]);
"""
        src += """    float cre = cj[2*j] * wgt, cim = cj[2*j+1] * wgt;
"""
        src += self._wrap_lines("ix", "i1x[j]", "lx", nu[0])
        if dim >= 2:
            src += self._wrap_lines("iy", "i1y[j]", "ly", nu[1])
        if dim == 1:
            src += """    size_t cell = (size_t)ix;
    atomic_fetch_add_explicit(&grid[2*cell],   cre, memory_order_relaxed);
    atomic_fetch_add_explicit(&grid[2*cell+1], cim, memory_order_relaxed);
"""
        elif dim == 2:
            src += f"""    size_t cell = (size_t)ix * {nu[1]} + (size_t)iy;
    atomic_fetch_add_explicit(&grid[2*cell],   cre, memory_order_relaxed);
    atomic_fetch_add_explicit(&grid[2*cell+1], cim, memory_order_relaxed);
"""
        else:
            src += f"""    size_t base = ((size_t)ix * {nu[1]} + (size_t)iy) * {nu[2]};
    int iz0 = i1z[j];
    float fz = frz[j];
    for (int lz = 0; lz < {w}; ++lz) {{
        float wz = k1_es((float)lz - fz);
        int iz = iz0 + lz;
        iz -= {nu[2]} * (iz >= {nu[2]});  iz += {nu[2]} * (iz < 0);
        atomic_fetch_add_explicit(&grid[2*(base + (size_t)iz)],
                                  cre * wz, memory_order_relaxed);
        atomic_fetch_add_explicit(&grid[2*(base + (size_t)iz) + 1],
                                  cim * wz, memory_order_relaxed);
    }}
"""
        innames = (["cj"] + [f"i1{_AX[d]}" for d in range(dim)]
                   + [f"fr{_AX[d]}" for d in range(dim)])
        return mx.fast.metal_kernel(
            name=f"t1spread{dim}d", input_names=innames,
            output_names=["grid"], header="#include <metal_math>\n" + self._es,
            source=src, atomic_outputs=True)

    # points per batch in the OD spread inner loop; staging costs
    # B*(dim*w + dim + 2) threadgroup words on top of the 2*ptot-word tile
    _OD_SPREAD_B = {1: 32, 2: 32, 3: 16}

    # threadgroup float add: bit-cast CAS on atomic_uint (Metal has no
    # threadgroup-scope atomic_float); contention is low — concurrent
    # lanes mostly hold distinct taps
    _TG_FADD = """#include <metal_math>
inline void tg_fadd(threadgroup metal::atomic_uint *a, float v) {
    uint prev = atomic_load_explicit(a, metal::memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        a, &prev, as_type<uint>(as_type<float>(prev) + v),
        metal::memory_order_relaxed, metal::memory_order_relaxed)) {}
}
"""

    def _build_od_spread(self, es):
        """Output-driven spread: one threadgroup per subproblem, padded tile
        in threadgroup memory. Points are processed in batches of B: lanes
        0..bn stage their point's strength (perm-gather) + tile cell offsets,
        all lanes cooperatively stage the per-dim 1D ES weights (layout
        wts[b][d][l]); after one barrier a flat loop over the batch's tap
        tasks accumulates into the tile with bit-cast CAS float adds
        (tg_fadd) — two barriers per batch instead of one per point, and all
        lanes stay busy. In dims 1-2 a task is one (point, tap); in 3D a
        task is (point, ax, ay) and the z taps ride a register loop over
        contiguous tile cells (one decode + xy-weight load per w adds).
        Flush: one global atomic add per tile cell, periodic wrap."""
        dim, w = self.dim, self.w
        nu = self.n_up
        m, p, ptot = self._od_m, self._od_p, self._od_ptot
        TG = self._OD_TG
        taps = w ** dim
        B = self._OD_SPREAD_B[dim]
        dw = dim * w
        src = f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sub = threadgroup_position_in_grid.y;
    threadgroup atomic_uint tile[{2 * ptot}];
    threadgroup float wts[{B * dw}];
    threadgroup float str[{2 * B}];
    threadgroup int loff[{B * dim}];
    for (uint q = tid; q < {2 * ptot}u; q += {TG}u)
        atomic_store_explicit(&tile[q], 0u, memory_order_relaxed);
    uint s0 = (uint)sub_start[sub];
    uint cnt = (uint)sub_count[sub];
"""
        for d in range(dim):
            src += f"    int o{_AX[d]} = sub_o{_AX[d]}[sub];\n"
        src += f"""    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t0 = 0; t0 < cnt; t0 += {B}u) {{
        uint bn = metal::min({B}u, cnt - t0);
        uint base = s0 + t0;
        if (tid < bn) {{
            uint jp = perm[base + tid];
            str[2*tid]   = cj[2*jp];
            str[2*tid+1] = cj[2*jp+1];
"""
        for d in range(dim):
            src += (f"            loff[{dim}u*tid + {d}u] = "
                    f"i1{_AX[d]}[base + tid] - o{_AX[d]};\n")
        src += "        }\n"
        for d in range(dim):
            src += f"""        for (uint q = tid; q < bn * {w}u; q += {TG}u) {{
            uint b = q / {w}u, l = q % {w}u;
            wts[b * {dw}u + {d * w}u + l] =
                k1_es((float)l - fr{_AX[d]}[base + b]);
        }}
"""
        src += "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        if dim == 1:
            src += f"""        for (uint q = tid; q < bn * {taps}u; q += {TG}u) {{
            uint b = q / {taps}u;
            uint ax = q % {taps}u;
            float wgt = wts[b * {dw}u + ax];
            uint cell = (uint)(loff[b] + (int)ax);
            tg_fadd(&tile[2*cell],   str[2*b]   * wgt);
            tg_fadd(&tile[2*cell+1], str[2*b+1] * wgt);
        }}
"""
        elif dim == 2:
            src += f"""        for (uint q = tid; q < bn * {taps}u; q += {TG}u) {{
            uint b = q / {taps}u;
            uint tap = q % {taps}u;
            uint ax = tap / {w}u, ay = tap % {w}u;
            float wgt = wts[b * {dw}u + ax] * wts[b * {dw}u + {w}u + ay];
            uint cell = (uint)(loff[2u*b] + (int)ax) * {p[1]}u
                      + (uint)(loff[2u*b+1u] + (int)ay);
            tg_fadd(&tile[2*cell],   str[2*b]   * wgt);
            tg_fadd(&tile[2*cell+1], str[2*b+1] * wgt);
        }}
"""
        else:
            src += f"""        for (uint q = tid; q < bn * {w * w}u; q += {TG}u) {{
            uint b = q / {w * w}u;
            uint rem = q % {w * w}u;
            uint ax = rem / {w}u, ay = rem % {w}u;
            float wxy = wts[b * {dw}u + ax] * wts[b * {dw}u + {w}u + ay];
            float cre = str[2*b] * wxy, cim = str[2*b+1] * wxy;
            uint cell0 = ((uint)(loff[3u*b] + (int)ax) * {p[1]}u
                        + (uint)(loff[3u*b+1u] + (int)ay)) * {p[2]}u
                       + (uint)loff[3u*b+2u];
            for (uint az = 0; az < {w}u; ++az) {{
                float wz = wts[b * {dw}u + {2 * w}u + az];
                tg_fadd(&tile[2*(cell0 + az)],   cre * wz);
                tg_fadd(&tile[2*(cell0 + az)+1], cim * wz);
            }}
        }}
"""
        src += """        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
        # flush padded tile to the global fine grid with periodic wrap
        src += f"    for (uint q = tid; q < {ptot}u; q += {TG}u) {{\n"
        if dim == 1:
            src += "        int px = (int)q;\n"
        elif dim == 2:
            src += (f"        int px = (int)(q / {p[1]}u), "
                    f"py = (int)(q % {p[1]}u);\n")
        else:
            src += (f"        int px = (int)(q / {p[1] * p[2]}u);\n"
                    f"        uint qr = q % {p[1] * p[2]}u;\n"
                    f"        int py = (int)(qr / {p[2]}u), "
                    f"pz = (int)(qr % {p[2]}u);\n")
        for d in range(dim):
            a = _AX[d]
            src += (f"        int g{a} = o{a} + p{a};  "
                    f"g{a} -= {nu[d]} * (g{a} >= {nu[d]});  "
                    f"g{a} += {nu[d]} * (g{a} < 0);\n")
        gexpr = _linearize([f"g{_AX[d]}" for d in range(dim)], nu)
        src += f"""        size_t cell = {gexpr};
        atomic_fetch_add_explicit(&grid[2*cell],
            as_type<float>(atomic_load_explicit(&tile[2*q],
                                                memory_order_relaxed)),
            memory_order_relaxed);
        atomic_fetch_add_explicit(&grid[2*cell+1],
            as_type<float>(atomic_load_explicit(&tile[2*q+1],
                                                memory_order_relaxed)),
            memory_order_relaxed);
    }}
"""
        innames = (["cj", "perm"] + [f"i1{_AX[d]}" for d in range(dim)]
                   + [f"fr{_AX[d]}" for d in range(dim)]
                   + ["sub_start", "sub_count"]
                   + [f"sub_o{_AX[d]}" for d in range(dim)])
        return mx.fast.metal_kernel(
            name=f"t1spread{dim}d_od", input_names=innames,
            output_names=["grid"], header=self._TG_FADD + es,
            source=src, atomic_outputs=True)

    # -- exclusive-ownership OD spread (gather formulation) ---------------
    #
    # One threadgroup OWNS one m^d region of the fine grid outright (no
    # guard padding): it is the only writer of those cells, so the flush is
    # plain coalesced stores — no global atomics and no output zero-init
    # (the region boxes tile the grid exactly; top-edge remainder regions
    # clip their extent). Candidate points live in the region's own bin
    # plus the 3^d-1 neighbouring bins (bin size m >= w//2 keeps every
    # support within one bin of its region, wrap included); each candidate
    # is clip-tested against the region box and simd-compacted before the
    # tap pipeline.
    #
    # Auto-selected for dim 1 ONLY, where it measures ~2x faster than the
    # padded-tile kernel (M5 Max, M=1e7, nu=2e6: 4.0 -> 2.0 ms). In dim 2 it
    # ties (11.5 vs 11.2 ms at M=1e7, nu=4096^2) and in dim 3 it loses
    # (75 -> 132 ms at M=1e7, nu=512^3): each point is scanned from 3^d
    # neighbour bins and staged in ~(1+(w-1)/m)^d regions, and the smaller
    # unpadded tile roughly doubles the threadgroup-CAS retry cost — in 3D
    # those overheads exceed the padded kernel's whole flush+init budget
    # (~10 ms measured). The kernel builder stays dim-general for retuning.

    _ODX_TG = 256
    # candidate windows in flight per simdgroup (overlaps the per-candidate
    # device-load latency); >1 costs registers — a loss for dims 2-3
    _ODX_ILP = {1: 4, 2: 1, 3: 1}
    _ODX_M = {1: 512, 2: 28, 3: 12}
    _ODX_MAXBIN = 65536              # clustering guard: 1 threadgroup/region

    def _odx_dims(self):
        """Region sizes m_d for the exclusive spread, or None if the
        geometry does not admit it (padded-tile OD is the fallback). Per
        dim: >= 3 regions (the neighbour scan needs distinct bins) and a
        top remainder region absent or >= w//2 wide (wrapped support from
        bin 0 must not reach below the last region); tile + compaction
        staging within the 32 KB threadgroup budget."""
        w2 = self.w // 2
        m = []
        for d in range(self.dim):
            nu, m0 = self.n_up[d], self._ODX_M[self.dim]
            md = next((mc for mc in range(m0, max(w2, m0 - 8) - 1, -1)
                       if nu >= 3 * mc
                       and (nu % mc == 0 or nu % mc >= w2)),
                      None)
            if md is None:
                return None
            m.append(md)
        tg_bytes = (2 * int(np.prod(m)) * 4
                    + self._ODX_TG * (2 * self.dim + 2) * 4
                    + (2 * 3 ** self.dim + 1) * 4 + 16)
        return m if tg_bytes <= 32 * 1024 else None

    def _odx_fold(self, i1, nu, w2):
        """Fold i1 so the support CENTRE cell i1 + w2 lies in [0, nu) — the
        bin the point resides in. Only even w at the top grid edge can push
        the centre to nu; shifting i1 by -nu is a no-op for the periodic
        tap arithmetic (same cells, same offsets)."""
        return i1 - (nu * ((i1 + w2) >= nu)).astype(i1.dtype)

    def _odx_prepare(self, m):
        """Host bin-sort onto the exclusive region grid + region CSR;
        mirrors _od_prepare."""
        if self.P < 20000:
            return False
        w2 = self.w // 2
        nb = [-(-self.n_up[d] // m[d]) for d in range(self.dim)]
        i1f = [self._odx_fold(self.i1[d].astype(np.int64), self.n_up[d], w2)
               for d in range(self.dim)]
        key = (i1f[0] + w2) // m[0]
        for d in range(1, self.dim):
            key = key * nb[d] + (i1f[d] + w2) // m[d]
        perm = np.argsort(key, kind="stable")
        if not self._odx_finish(key[perm], nb, m):
            return False
        self.perm = perm
        self.mx_perm = mx.array(perm.astype(np.uint32))
        self.mx_i1 = [mx.array(v[perm].astype(np.int32)) for v in i1f]
        self.mx_fr = [mx.array(v[perm]) for v in self.fr]
        return True

    def _odx_prepare_gpu(self, m, i1_g, fr_g):
        """GPU bin-sort for the exclusive spread; the CSR build stays host-
        side from ONE P-length pull of the sorted keys (cf. _od_prepare_gpu)."""
        if self.P < 20000:
            return False
        w2 = self.w // 2
        nb = [-(-self.n_up[d] // m[d]) for d in range(self.dim)]
        i1f = [i1_g[d] - self.n_up[d] * ((i1_g[d] + w2) >= self.n_up[d])
               for d in range(self.dim)]
        key_g = (i1f[0] + w2) // m[0]
        for d in range(1, self.dim):
            key_g = key_g * nb[d] + (i1f[d] + w2) // m[d]
        perm_g = mx.argsort(key_g).astype(mx.uint32)
        key_s = np.array(mx.take(key_g, perm_g))       # one P-length sync
        if not self._odx_finish(key_s, nb, m):
            return False
        self.mx_perm = perm_g
        self.mx_i1 = [mx.take(v, perm_g) for v in i1f]
        self.mx_fr = [mx.take(v, perm_g) for v in fr_g]
        mx.eval(self.mx_perm, *self.mx_i1, *self.mx_fr)
        self.perm = np.array(self.mx_perm).astype(np.intp)
        return True

    def _odx_finish(self, key_s, nb, m):
        """Region CSR (bin_start, over ALL regions incl. empty — every grid
        cell must be written by exactly one region) from the sorted bin
        keys. False under extreme clustering: a region cannot be split
        across threadgroups without losing the exclusive plain flush."""
        nreg = int(np.prod(nb))
        starts = np.searchsorted(key_s, np.arange(nreg + 1)).astype(np.int32)
        if int(np.diff(starts).max(initial=0)) > self._ODX_MAXBIN:
            return False
        self.sorted = True
        self._odx_m, self._odx_nb, self._odx_nreg = m, nb, nreg
        self._mx_bin_start = mx.array(starts)
        return True

    def _build_od_spread_ex(self, es):
        """Exclusive-ownership spread kernel, simdgroup-autonomous: at ~28 KB
        of threadgroup memory only one threadgroup is resident per core, so
        threadgroup barriers stall the whole core — the pipeline therefore
        runs per SIMDGROUP (lockstep, simdgroup_barrier only) with no
        threadgroup barrier between the initial range-table build and the
        final flush. A shared work counter deals 32-candidate windows of the
        3^d neighbour-bin ranges to simdgroups; each lane maps its virtual
        index to (range, point) via the prefixed range table, clip-tests the
        point's tap window against the owned box (wrap bins carry a +-nu
        cell-offset adjustment), simd-compacts the survivors into the
        simdgroup's staging slots, and the simdgroup spreads them with
        on-the-fly ES weights (fast exp2) and tg_fadd CAS adds into the
        UNPADDED m^d tile. Flush: plain stores to the owned cells."""
        dim, w = self.dim, self.w
        nu = self.n_up
        m, nb = self._odx_m, self._odx_nb
        TG, ilp = self._ODX_TG, self._ODX_ILP[self.dim]
        nr = 3 ** dim
        mtot = int(np.prod(m))
        src = f"""
    uint tid = thread_position_in_threadgroup.x;
    uint rid = threadgroup_position_in_grid.y;
    uint lane = tid & 31u;
    threadgroup atomic_uint tile[{2 * mtot}];
"""
        for d in range(dim):
            a = _AX[d]
            src += (f"    threadgroup int so{a}[{TG}];  "
                    f"threadgroup float sf{a}[{TG}];\n")
        src += f"""    threadgroup float ssr[{TG}], ssi[{TG}];
    threadgroup uint bst[{nr}], cum[{nr + 1}];
    threadgroup atomic_uint wk;
"""
        if dim == 3:
            src += f"""    uint rx = rid / {nb[1] * nb[2]}u;
    uint rr = rid % {nb[1] * nb[2]}u;
    uint ry = rr / {nb[2]}u, rz = rr % {nb[2]}u;
"""
        elif dim == 2:
            src += f"    uint rx = rid / {nb[1]}u, ry = rid % {nb[1]}u;\n"
        else:
            src += "    uint rx = rid;\n"
        for d in range(dim):
            a = _AX[d]
            src += (f"    int g0{a} = (int)r{a} * {m[d]};  "
                    f"int ms{a} = metal::min({m[d]}, {nu[d]} - g0{a});\n")
        src += f"""    for (uint q = tid; q < {2 * mtot}u; q += {TG}u)
        atomic_store_explicit(&tile[q], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&wk, 0u, memory_order_relaxed);
    if (tid < {nr}u) {{
"""
        # neighbour-bin decode for range r = tid (fastest axis fastest)
        div = 1
        for d in range(dim - 1, -1, -1):
            a = _AX[d]
            src += (f"        int b{a} = (int)r{a} + (int)(tid / {div}u) "
                    f"% 3 - 1;\n"
                    f"        b{a} += {nb[d]} * (b{a} < 0);  "
                    f"b{a} -= {nb[d]} * (b{a} >= {nb[d]});\n")
            div *= 3
        bexpr = f"(uint)b{_AX[0]}"
        for d in range(1, dim):
            bexpr = f"({bexpr} * {nb[d]}u + (uint)b{_AX[d]})"
        src += f"""        uint bi = {bexpr};
        bst[tid] = (uint)bin_start[bi];
        cum[tid] = (uint)bin_start[bi + 1u] - bst[tid];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {{
        uint acc = 0u;
        for (uint r = 0u; r < {nr}u; ++r)
            {{ uint cc = cum[r]; cum[r] = acc; acc += cc; }}
        cum[{nr}] = acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = cum[{nr}];
    uint v0 = 0u;
    if (metal::simd_is_first())
        v0 = atomic_fetch_add_explicit(&wk, {32 * ilp}u, memory_order_relaxed);
    v0 = metal::simd_broadcast_first(v0);
    while (v0 < total) {{
        // {ilp} windows in flight: the per-candidate device loads (i1) are
        // issued together so their latency overlaps instead of chaining
        bool keep[{ilp}];
        uint j[{ilp}];
"""
        src += ("        int "
                + ", ".join(f"o{_AX[d]}[{ilp}]" for d in range(dim)) + ";\n")
        src += f"""        for (uint u = 0u; u < {ilp}u; ++u) {{
            keep[u] = false;
            uint v = v0 + u * 32u + lane;
            if (v < total) {{
                uint r = 0u;
"""
        step = 1
        while step * 2 <= nr - 1:
            step *= 2
        while step:
            src += (f"                if (r + {step}u <= {nr - 1}u && "
                    f"cum[r + {step}u] <= v) r += {step}u;\n")
            step //= 2
        src += "                j[u] = bst[r] + (v - cum[r]);\n"
        div = 1
        for d in range(dim - 1, -1, -1):
            a = _AX[d]
            src += (f"                int b{a} = (int)r{a} + "
                    f"(int)(r / {div}u) % 3 - 1;\n"
                    f"                int adj{a} = (b{a} < 0) ? -{nu[d]} : "
                    f"((b{a} >= {nb[d]}) ? {nu[d]} : 0);\n"
                    f"                o{a}[u] = i1{a}[j[u]] + adj{a} - g0{a};\n")
            div *= 3
        src += ("                keep[u] = "
                + "\n                    && ".join(
                    f"(metal::max(0, -o{_AX[d]}[u]) < "
                    f"metal::min({w}, ms{_AX[d]} - o{_AX[d]}[u]))"
                    for d in range(dim))
                + ";\n            }\n        }\n")
        src += """        for (uint u = 0u; u < {ILP}u; ++u) {
        uint flag = keep[u] ? 1u : 0u;
        uint pos = metal::simd_prefix_exclusive_sum(flag);
        uint nk = metal::simd_sum(flag);
        if (nk == 0u) continue;
        if (keep[u]) {
            uint s = (tid & ~31u) + pos;
""".replace("{ILP}", str(ilp))
        for d in range(dim):
            a = _AX[d]
            src += (f"            so{a}[s] = o{a}[u];  "
                    f"sf{a}[s] = fr{a}[j[u]];\n")
        src += f"""            uint jp = perm[j[u]];
            ssr[s] = cj[2u*jp];  ssi[s] = cj[2u*jp+1u];
        }}
        metal::simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = lane; t < nk * {w}u; t += 32u) {{
            uint b = (tid & ~31u) + t / {w}u;
            int ax = (int)(t % {w}u);
            int ux = so{_AX[0]}[b] + ax;
            if (ux >= 0 && ux < ms{_AX[0]}) {{
                float wx = k1_es((float)ax - sf{_AX[0]}[b]);
"""
        if dim == 1:
            src += """                tg_fadd(&tile[2u*(uint)ux],    ssr[b] * wx);
                tg_fadd(&tile[2u*(uint)ux+1u], ssi[b] * wx);
"""
        elif dim == 2:
            src += f"""                float cre = ssr[b] * wx, cim = ssi[b] * wx;
                int oy2 = soy[b];  float fy = sfy[b];
                uint rowb = (uint)ux * {m[1]}u;
                for (int ly = 0; ly < {w}; ++ly) {{
                    int uy = oy2 + ly;
                    if (uy < 0 || uy >= msy) continue;
                    float wy = k1_es((float)ly - fy);
                    uint cell = rowb + (uint)uy;
                    tg_fadd(&tile[2u*cell],    cre * wy);
                    tg_fadd(&tile[2u*cell+1u], cim * wy);
                }}
"""
        else:
            src += f"""                float cre = ssr[b] * wx, cim = ssi[b] * wx;
                int oy2 = soy[b], oz2 = soz[b];
                float fy = sfy[b], fz = sfz[b];
                int ly0 = metal::max(0, -oy2);
                int ly1 = metal::min({w}, msy - oy2);
                uint rowb = (uint)ux * {m[1]}u;
                for (int ly = ly0; ly < ly1; ++ly) {{
                    float wy = k1_es((float)ly - fy);
                    float yre = cre * wy, yim = cim * wy;
                    uint rowc = (rowb + (uint)(oy2 + ly)) * {m[2]}u;
                    for (int lz = 0; lz < {w}; ++lz) {{
                        int uz = oz2 + lz;
                        if (uz < 0 || uz >= msz) continue;
                        float wz = k1_es((float)lz - fz);
                        uint cell = rowc + (uint)uz;
                        tg_fadd(&tile[2u*cell],    yre * wz);
                        tg_fadd(&tile[2u*cell+1u], yim * wz);
                    }}
                }}
"""
        src += f"""            }}
        }}
        metal::simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (metal::simd_is_first())
            v0 = atomic_fetch_add_explicit(&wk, {32 * ilp}u,
                                           memory_order_relaxed);
        v0 = metal::simd_broadcast_first(v0);
    }}
"""
        # flush: plain stores to the exclusively-owned cells
        src += f"""    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < {mtot}u; q += {TG}u) {{
"""
        if dim == 3:
            src += (f"        int ux = (int)(q / {m[1] * m[2]}u);\n"
                    f"        uint qr = q % {m[1] * m[2]}u;\n"
                    f"        int uy = (int)(qr / {m[2]}u), "
                    f"uz = (int)(qr % {m[2]}u);\n")
        elif dim == 2:
            src += (f"        int ux = (int)(q / {m[1]}u), "
                    f"uy = (int)(q % {m[1]}u);\n")
        else:
            src += "        int ux = (int)q;\n"
        src += ("        if ("
                + " || ".join(f"u{_AX[d]} >= ms{_AX[d]}" for d in range(dim))
                + ") continue;\n")
        gexpr = _linearize([f"(g0{_AX[d]} + u{_AX[d]})" for d in range(dim)],
                           nu)
        src += f"""        size_t cell = {gexpr};
        grid[2*cell]   = as_type<float>(atomic_load_explicit(&tile[2u*q],
                                            memory_order_relaxed));
        grid[2*cell+1] = as_type<float>(atomic_load_explicit(&tile[2u*q+1u],
                                            memory_order_relaxed));
    }}
"""
        innames = (["cj", "perm"] + [f"i1{_AX[d]}" for d in range(dim)]
                   + [f"fr{_AX[d]}" for d in range(dim)] + ["bin_start"])
        return mx.fast.metal_kernel(
            name=f"t1spread{dim}d_odx", input_names=innames,
            output_names=["grid"],
            header="#include <metal_simdgroup>\n" + self._TG_FADD + es,
            source=src)

    def _launch_spread_ex(self, cmx):
        """Launch the exclusive-ownership spread (no init_value: the region
        boxes cover every grid cell exactly once)."""
        nu_tot = int(np.prod(self.n_up))
        cpf = mx.view(cmx, dtype=mx.float32)
        return self._spread_ex(
            inputs=[cpf, self.mx_perm] + self.mx_i1 + self.mx_fr
                   + [self._mx_bin_start],
            output_shapes=[(nu_tot * 2,)],
            output_dtypes=[mx.float32],
            grid=(self._ODX_TG, self._odx_nreg, 1),
            threadgroup=(self._ODX_TG, 1, 1))[0]

    def _crop_launch(self, B):
        N = self.N
        if self.dim == 3:
            g = (N[2], N[1], B * N[0])
        elif self.dim == 2:
            g = (N[1], N[0], B)
        else:
            g = (N[0], B, 1)
        return g, self._tg_for(g[0])

    def _fft_modes(self, H):
        """Progressive FFT+crop: (*n_up) or (B, *n_up) grid -> (*N) / (B, *N)
        deconvolved modes (lazy).

        After axis d's FFT only its N[d] in-band rows survive, so crop axis d
        before FFT-ing the next axis — FFT work drops to
        (1 + 1/sigma + ... + 1/sigma^(dim-1))/dim of the full-grid chain. Each
        FFT runs on the contiguous LAST axis (MLX's native path); the crop
        gather also cycles the axes so the next axis lands last, replacing
        MLX's internal strided-axis transpose round-trips. The axis-0 crop is
        folded into the deconvolution kernel (_crop)."""
        dim = self.dim
        inv = self.isign > 0
        nb = H.ndim - dim                     # 0 or 1 leading batch axes
        B = int(H.shape[0]) if nb else 1
        cyc = tuple(range(nb)) + (nb + dim - 1,) \
            + tuple(range(nb, nb + dim - 1))
        H = fft_axis(H, H.ndim - 1, inverse=inv, twiddle_cache=self._twiddles)
        for ax in range(dim - 1, 0, -1):
            H = mx.take(mx.transpose(H, cyc), self._mx_cropidx[ax], axis=nb)
            H = fft_axis(H, H.ndim - 1, inverse=inv,
                         twiddle_cache=self._twiddles)
        vf = mx.view(H, dtype=mx.float32).reshape(-1)
        N_tot = int(np.prod(self.N))
        g, tg = self._crop_launch(B)
        fk = self._crop(
            inputs=[vf] + self.mx_dec,
            output_shapes=[(B * N_tot * 2,)],
            output_dtypes=[mx.float32],
            grid=g, threadgroup=tg)[0]
        out = mx.view(fk, dtype=mx.complex64)
        return out.reshape(B, *self.N) if nb else out.reshape(*self.N)

    def execute(self, c, return_np=True):
        nu_tot = int(np.prod(self.n_up))
        # asarray(dtype=...) is a no-op for complex64 input (astype copies)
        cmx = mx.array(np.asarray(c, dtype=np.complex64)) \
            if not isinstance(c, mx.array) else c
        if cmx.size != self.P:
            raise ValueError(f"c.size ({cmx.size}) must equal the number of "
                             f"nonuniform points ({self.P})")
        # OD-family kernels fold the permutation in (perm-indexed reads)
        if self._od_ex:
            bf = self._launch_spread_ex(cmx)
        elif self._od:
            cpf = mx.view(cmx, dtype=mx.float32)
            bf = self._spread_od(
                inputs=[cpf, self.mx_perm] + self.mx_i1 + self.mx_fr
                       + [self._mx_sub_start, self._mx_sub_count]
                       + self._mx_sub_o,
                output_shapes=[(nu_tot * 2,)],
                output_dtypes=[mx.float32],
                grid=(self._OD_TG, self._od_nsub, 1),
                threadgroup=(self._OD_TG, 1, 1),
                init_value=0)[0]
        else:
            cpf = mx.view(mx.take(cmx, self.mx_perm), dtype=mx.float32)
            lanes = self._lanes
            bf = self._spread(
                inputs=[cpf] + self.mx_i1 + self.mx_fr,
                output_shapes=[(nu_tot * 2,)],
                output_dtypes=[mx.float32],
                grid=(lanes, max(self.P, 1), 1),
                threadgroup=(lanes, max(1, 1024 // lanes), 1),
                init_value=0)[0]
        res = self._fft_modes(          # chained: spread evals with the FFT
            mx.view(bf, dtype=mx.complex64).reshape(*self.n_up))
        del bf
        mx.eval(res)
        return np.array(res) if return_np else res

    def set_sources(self, x, backend="host"):
        """Cheaply re-point the plan to new nonuniform coordinates, reusing the
        compiled kernels, mode deconvolution and FFT setup — only the
        point-dependent cell indices/fractions/sort (and OD bin tables) are
        rebuilt. For repeated workloads where the MODE box is fixed but the
        points move each call (the type-1 analogue of Type3Plan.set_sources).
        P may change; kernels that bake the point count are rebuilt then.

        backend="host": fp64 numpy rescale + sort (crit64 grade); CPU-argsort
                        bound (~hundreds of ms at P~1e6).
        backend="gpu":  df64 (double-single) Metal rescale + mx.argsort sort,
                        crit64 grade, ~tens of ms — the type-1 analogue of
                        Type3Plan.set_sources(backend="gpu"). For OD plans it
                        also skips the redundant lateral-key sort the host path
                        computes-then-discards.
        Returns self."""
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = (x,)
        x = tuple(x)
        if len(x) != self.dim:
            raise ValueError(f"expected {self.dim} coordinate arrays, "
                             f"got {len(x)}")
        if backend not in ("host", "gpu"):
            raise ValueError(f"unknown backend {backend!r}")
        x64 = [np.asarray(v, dtype=np.float64).ravel() for v in x]
        P_new = x64[0].size
        if P_new < 1 or not all(v.size == P_new for v in x64):
            raise ValueError("coordinate arrays must be non-empty, equal length")
        p_changed = (P_new != self.P)
        self.P = P_new
        self._spread_batch = {}                 # baked P -> invalidate cache
        if p_changed and self._spread is not None:
            self._spread = self._build_gm_spread()   # GM spread bakes P
        # Compute cells once (no sort); then EITHER the OD bin-sort OR the
        # non-OD lateral sort — never both (the host path's wasted first sort
        # is discarded by the OD bin-sort; here it is simply never run).
        if backend == "gpu":
            i1_g, fr_g = self._compute_cells_gpu(x64)
            if self._od_ex and \
                    not self._odx_prepare_gpu(self._odx_m, i1_g, fr_g):
                self._od_ex = False   # no longer eligible (P, clustering)
                self._od = self._od_prepare_gpu(self._OD_MSUB, i1_g, fr_g)
                if self._od:
                    self._spread_od = self._build_od_spread(self._es)
            elif self._od and \
                    not self._od_prepare_gpu(self._OD_MSUB, i1_g, fr_g):
                self._od = False            # no longer OD-eligible (e.g. P)
            if not (self._od_ex or self._od):
                if self._spread is None:
                    self._spread = self._build_gm_spread()
                self._sort_and_upload_gpu(i1_g, fr_g)
        else:
            self._compute_cells(x64)
            if self._od_ex and not self._odx_prepare(self._odx_m):
                self._od_ex = False
                self._od = self._od_prepare(self._OD_MSUB)
                if self._od:
                    self._spread_od = self._build_od_spread(self._es)
            elif self._od and not self._od_prepare(self._OD_MSUB):
                self._od = False
            if not (self._od_ex or self._od):
                if self._spread is None:
                    self._spread = self._build_gm_spread()
                self._sort_and_upload()
        return self

    def _get_batch_spread(self, B):
        """Batched GM spread: one kernel computes the ES kernel weight once per
        (point, tap) and atomic-scatters all B strength vectors into B separate
        grids — the 'one shared spread, B FFTs' path. cs laid out (P, B, 2)."""
        if not hasattr(self, "_spread_batch"):
            self._spread_batch = {}
        k = self._spread_batch.get(B)
        if k is not None:
            return k
        dim, w, P, nu = self.dim, self.w, self.P, self.n_up
        lanes = self._lanes
        nutot = int(np.prod(nu))
        src = f"""
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    if (lane >= {lanes}u || j >= {P}u) return;
"""
        if dim == 1:
            src += "    int lx = (int)lane;\n    float wgt = k1_es((float)lx - frx[j]);\n"
        else:
            src += (f"    int lx = (int)(lane / {w}u), ly = (int)(lane % {w}u);\n"
                    f"    float wgt = k1_es((float)lx - frx[j]) * k1_es((float)ly - fry[j]);\n")
        src += self._wrap_lines("ix", "i1x[j]", "lx", nu[0])
        if dim >= 2:
            src += self._wrap_lines("iy", "i1y[j]", "ly", nu[1])
        if dim <= 2:
            cellL = "(size_t)ix" if dim == 1 else f"(size_t)ix * {nu[1]} + (size_t)iy"
            src += f"""    size_t cellL = {cellL};
    for (uint b = 0; b < {B}u; ++b) {{
        size_t s = 2*((size_t)j*{B}u + b);
        float cre = cj[s] * wgt, cim = cj[s+1] * wgt;
        size_t go = (size_t)b * {nutot} + cellL;
        atomic_fetch_add_explicit(&grid[2*go],   cre, memory_order_relaxed);
        atomic_fetch_add_explicit(&grid[2*go+1], cim, memory_order_relaxed);
    }}
"""
        else:
            src += f"""    size_t base = ((size_t)ix * {nu[1]} + (size_t)iy) * {nu[2]};
    int iz0 = i1z[j];  float fz = frz[j];
    for (int lz = 0; lz < {w}; ++lz) {{
        float wz = k1_es((float)lz - fz);
        int iz = iz0 + lz;  iz -= {nu[2]} * (iz >= {nu[2]});  iz += {nu[2]} * (iz < 0);
        size_t cellL = base + (size_t)iz;
        for (uint b = 0; b < {B}u; ++b) {{
            size_t s = 2*((size_t)j*{B}u + b);
            float cre = cj[s] * wgt * wz, cim = cj[s+1] * wgt * wz;
            size_t go = (size_t)b * {nutot} + cellL;
            atomic_fetch_add_explicit(&grid[2*go],   cre, memory_order_relaxed);
            atomic_fetch_add_explicit(&grid[2*go+1], cim, memory_order_relaxed);
        }}
    }}
"""
        innames = (["cj"] + [f"i1{_AX[d]}" for d in range(dim)]
                   + [f"fr{_AX[d]}" for d in range(dim)])
        k = mx.fast.metal_kernel(
            name=f"t1spread{dim}d_b{B}", input_names=innames,
            output_names=["grid"], header="#include <metal_math>\n" + self._es,
            source=src, atomic_outputs=True)
        self._spread_batch[B] = k
        return k

    def execute_batch(self, cs, return_np=True):
        """Batched multi-strength type-1: transform B strength vectors over the
        SAME points through ONE shared spread (ES kernel weights computed once
        per point/tap, applied to all B) + B FFTs + B crops. cs: (B, P).
        Returns (B, *N). Useful when many strength vectors share one point
        set. Output grids must fit the metal_kernel int32 cap, so B is
        processed in chunks when needed."""
        cs = np.asarray(cs)
        if cs.ndim != 2 or cs.shape[1] != self.P:
            raise ValueError(f"cs must have shape (B, {self.P})")
        Btot = int(cs.shape[0])
        nu_tot = int(np.prod(self.n_up))
        lanes = self._lanes
        Bchunk = max(1, (2**31 - 1) // (nu_tot * 2))     # grid output int32 cap
        Cm = mx.array(np.asarray(cs, dtype=np.complex64))              # (Btot, P)
        Csort = mx.take(Cm, self.mx_perm, axis=1)        # plan (sort) order
        outs = []
        for b0 in range(0, Btot, Bchunk):
            B = min(Bchunk, Btot - b0)
            # layout (P, B, 2): per point, the B strengths are contiguous
            cpf = mx.view(mx.contiguous(mx.transpose(Csort[b0:b0 + B], (1, 0))),
                          dtype=mx.float32).reshape(-1)
            spread_b = self._get_batch_spread(B)
            bf = spread_b(
                inputs=[cpf] + self.mx_i1 + self.mx_fr,
                output_shapes=[(B * nu_tot * 2,)],
                output_dtypes=[mx.float32],
                grid=(lanes, max(self.P, 1), 1),
                threadgroup=(lanes, max(1, 1024 // lanes), 1),
                init_value=0)[0]
            mx.eval(bf)
            # whole-chunk per-axis FFTs (leading batch dim) + progressive crop
            ob = self._fft_modes(
                mx.view(bf, dtype=mx.complex64).reshape(B, *self.n_up))
            del bf
            mx.eval(ob)
            outs.append(ob)
        res = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        mx.eval(res)
        return np.array(res) if return_np else res

    def execute_disjoint(self, c, groups, return_np=True):
        """Disjoint-support batch: G subsets of the SAME point set, each
        transformed independently (coefficients outside the subset zeroed),
        through ONE shared plan. Each point is spread exactly once total (into
        its subset's grid) rather than once per subset, so the spread stage is
        shared across the G transforms; the per-subset FFT and crop are not
        shareable (each subset yields different modes). Returns f of shape
        (G, *N).

        Realized speedup vs independent transforms is largest where spreading
        dominates (high point density); for FFT-dominated geometries it tends
        to the plan-amortization the shared plan already gives. `groups` is an
        int array of length P assigning each point to a subset 0..G-1."""
        groups = np.asarray(groups)
        if groups.shape != (self.P,):
            raise ValueError(f"groups must have shape ({self.P},)")
        cmx = mx.array(np.asarray(c, dtype=np.complex64)) \
            if not isinstance(c, mx.array) else c
        if cmx.size != self.P:
            raise ValueError(f"c.size ({cmx.size}) must equal the number of "
                             f"nonuniform points ({self.P})")
        if self._spread is None:                 # OD plan: build GM on demand
            self._spread = self._build_gm_spread()
        G = int(groups.max()) + 1 if groups.size else 0
        dim, lanes = self.dim, self._lanes
        nu_tot = int(np.prod(self.n_up))
        N_tot = int(np.prod(self.N))
        gsort = groups[self.perm]                # labels in plan (perm) order
        c_sorted = mx.take(cmx, self.mx_perm)
        outs = []
        for g in range(G):
            pos = np.flatnonzero(gsort == g).astype(np.uint32)
            if pos.size == 0:
                outs.append(mx.zeros((N_tot,), dtype=mx.complex64)
                            .reshape(*self.N))
                continue
            px = mx.array(pos)
            i1g = [mx.take(self.mx_i1[d], px) for d in range(dim)]
            frg = [mx.take(self.mx_fr[d], px) for d in range(dim)]
            cg = mx.view(mx.take(c_sorted, px), dtype=mx.float32)
            bf = self._spread(
                inputs=[cg] + i1g + frg,
                output_shapes=[(nu_tot * 2,)], output_dtypes=[mx.float32],
                grid=(lanes, int(pos.size), 1),
                threadgroup=(lanes, max(1, 1024 // lanes), 1),
                init_value=0)[0]
            mx.eval(bf)
            fk = self._fft_modes(
                mx.view(bf, dtype=mx.complex64).reshape(*self.n_up))
            del bf
            mx.eval(fk)
            outs.append(fk)
        res = mx.stack(outs)
        mx.eval(res)
        return np.array(res) if return_np else res


class Type2PlanND(_PointsND):
    """c[j] = sum_k f[k] exp(i*isign * k . x_j), modeord=0 box, dims 1-3."""

    def __init__(self, x, n_modes, eps=1e-6, isign=-1, upsampfac=2.0,
                 prec="crit64", sort_points=False, spread_method="auto"):
        super().__init__(x, n_modes, eps, isign, upsampfac, prec, sort_points)
        dim, w, P = self.dim, self.w, self.P
        nu = self.n_up
        N = self.N
        es = _es_msl("k2", w, self.beta)

        assert spread_method in ("auto", "od", "gm")
        # interp has no atomics/barrier-per-point: whole bins as subproblems
        # so each bin's tile is loaded from the fine grid exactly once
        self._od = (spread_method != "gm") and self._od_prepare(1 << 22)
        if spread_method == "od" and not self._od:
            raise ValueError("OD interp not applicable to this geometry")
        if self._od:
            self._gather_od = self._build_od_gather(es)

        # ---- axis-0 pad + deconvolve (fused) -----------------------------
        # _modes_to_grid FFTs axis 0 first, on a grid still cropped in every
        # other axis: this kernel scatters the deconvolved modes into the
        # axis-0 FFT-order rows (zeros out of band), with axis 0 cycled to
        # the contiguous last position -> output (N[1], .., N[dim-1], nu[0]).
        if dim == 3:
            src = f"""
    uint r1 = thread_position_in_grid.x;
    uint m3 = thread_position_in_grid.y;
    uint m2 = thread_position_in_grid.z;
    if (r1 >= {nu[0]}u || m3 >= {N[2]}u) return;
"""
        elif dim == 2:
            src = f"""
    uint r1 = thread_position_in_grid.x;
    uint m2 = thread_position_in_grid.y;
    if (r1 >= {nu[0]}u || m2 >= {N[1]}u) return;
"""
        else:
            src = f"""
    uint r1 = thread_position_in_grid.x;
    if (r1 >= {nu[0]}u) return;
"""
        dnames = [f"m{d + 1}" for d in range(1, dim)] + ["r1"]
        ddims = [N[d] for d in range(1, dim)] + [nu[0]]
        src += f"    size_t dst = {_linearize(dnames, ddims)};\n"
        src += (f"    int q1 = (int)r1;  "
                f"q1 -= {nu[0]} * (q1 >= {nu[0] - N[0] // 2});\n")
        src += (f"    bool inband = (q1 >= {-(N[0] // 2)} && "
                f"q1 < {N[0] - N[0] // 2});\n")
        src += """    if (!inband) { H[2*dst] = 0.0f; H[2*dst+1] = 0.0f; return; }
"""
        src += f"    int m1 = q1 + {N[0] // 2};\n"
        src += (f"    size_t src = "
                f"{_linearize([f'm{d + 1}' for d in range(dim)], N)};\n")
        src += ("    float d = "
                + " * ".join(f"dec{d + 1}[m{d + 1}]" for d in range(dim))
                + ";\n")
        src += """    H[2*dst]   = fk[2*src] * d;
    H[2*dst+1] = fk[2*src+1] * d;
"""
        self._pad = mx.fast.metal_kernel(
            name=f"t2pad{dim}d",
            input_names=["fk"] + [f"dec{d + 1}" for d in range(dim)],
            output_names=["H"], source=src)

        # ---- gather/interp ------------------------------------------------
        src = f"""
    uint kk = thread_position_in_grid.x;
    if (kk >= {P}u) return;
"""
        for d in range(dim):
            a = _AX[d]
            src += f"""    float w{a}[{w}];
    int j{a}[{w}];
    float f{a} = tfr{a}[kk];
    int {a}0 = ti1{a}[kk];
    for (int l = 0; l < {w}; ++l) {{
        w{a}[l] = k2_es((float)l - f{a});
        int t = {a}0 + l; t -= {nu[d]} * (t >= {nu[d]}); t += {nu[d]} * (t < 0);
        j{a}[l] = t;
    }}
"""
        src += "    float accre = 0.0f, accim = 0.0f;\n"
        if dim == 1:
            src += f"""    for (int lx = 0; lx < {w}; ++lx) {{
        size_t idx = (size_t)jx[lx];
        accre = metal::fma(v[2*idx],   wx[lx], accre);
        accim = metal::fma(v[2*idx+1], wx[lx], accim);
    }}
"""
        elif dim == 2:
            src += f"""    for (int lx = 0; lx < {w}; ++lx) {{
        size_t base = (size_t)jx[lx] * {nu[1]};
        float sre = 0.0f, sim = 0.0f;
        for (int ly = 0; ly < {w}; ++ly) {{
            float wv = wy[ly];
            size_t idx = base + (size_t)jy[ly];
            sre = metal::fma(v[2*idx],   wv, sre);
            sim = metal::fma(v[2*idx+1], wv, sim);
        }}
        accre = metal::fma(sre, wx[lx], accre);
        accim = metal::fma(sim, wx[lx], accim);
    }}
"""
        else:
            src += f"""    for (int lx = 0; lx < {w}; ++lx) {{
        for (int ly = 0; ly < {w}; ++ly) {{
            float wxy = wx[lx] * wy[ly];
            size_t base = ((size_t)jx[lx] * {nu[1]} + (size_t)jy[ly]) * {nu[2]};
            float sre = 0.0f, sim = 0.0f;
            for (int lz = 0; lz < {w}; ++lz) {{
                float wv = wz[lz];
                size_t idx = base + (size_t)jz[lz];
                sre = metal::fma(v[2*idx],   wv, sre);
                sim = metal::fma(v[2*idx+1], wv, sim);
            }}
            accre = metal::fma(sre, wxy, accre);
            accim = metal::fma(sim, wxy, accim);
        }}
    }}
"""
        src += """    out[2*kk]   = accre;
    out[2*kk+1] = accim;
"""
        innames = (["v"] + [f"ti1{_AX[d]}" for d in range(dim)]
                   + [f"tfr{_AX[d]}" for d in range(dim)])
        self._gather = None if self._od else mx.fast.metal_kernel(
            name=f"t2gather{dim}d", input_names=innames,
            output_names=["out"], header="#include <metal_math>\n" + es,
            source=src)

    def _build_od_gather(self, es):
        """Tiled interp: one threadgroup per subproblem loads the padded
        tile from the fine grid once (periodic wrap), then each thread
        gathers whole points from threadgroup memory — no atomics, one
        barrier per subproblem."""
        dim, w = self.dim, self.w
        nu = self.n_up
        m, p, ptot = self._od_m, self._od_p, self._od_ptot
        TG = self._OD_TG
        src = f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sub = threadgroup_position_in_grid.y;
    threadgroup float tile[{2 * ptot}];
"""
        for d in range(dim):
            src += f"    int o{_AX[d]} = sub_o{_AX[d]}[sub];\n"
        src += f"    for (uint q = tid; q < {ptot}u; q += {TG}u) {{\n"
        if dim == 1:
            src += "        int px = (int)q;\n"
        elif dim == 2:
            src += f"        int px = (int)(q / {p[1]}u), py = (int)(q % {p[1]}u);\n"
        else:
            src += (f"        int px = (int)(q / {p[1] * p[2]}u);\n"
                    f"        uint qr = q % {p[1] * p[2]}u;\n"
                    f"        int py = (int)(qr / {p[2]}u), "
                    f"pz = (int)(qr % {p[2]}u);\n")
        for d in range(dim):
            a = _AX[d]
            src += (f"        int g{a} = o{a} + p{a};  "
                    f"g{a} -= {nu[d]} * (g{a} >= {nu[d]});  "
                    f"g{a} += {nu[d]} * (g{a} < 0);\n")
        gexpr = _linearize([f"g{_AX[d]}" for d in range(dim)], nu)
        src += f"""        size_t cell = {gexpr};
        tile[2*q]   = v[2*cell];
        tile[2*q+1] = v[2*cell+1];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s0 = (uint)sub_start[sub];
    uint cnt = (uint)sub_count[sub];
    for (uint t = tid; t < cnt; t += {TG}u) {{
        uint j = s0 + t;
        uint jp = perm[j];
"""
        for d in range(dim):
            a = _AX[d]
            src += f"""        float w{a}[{w}];
        float f{a} = tfr{a}[j];
        int l0{a} = ti1{a}[j] - o{a};
        for (int l = 0; l < {w}; ++l) w{a}[l] = k2_es((float)l - f{a});
"""
        src += "        float accre = 0.0f, accim = 0.0f;\n"
        if dim == 1:
            src += """        for (int lx = 0; lx < %d; ++lx) {
            uint idx = (uint)(l0x + lx);
            accre = metal::fma(tile[2*idx],   wx[lx], accre);
            accim = metal::fma(tile[2*idx+1], wx[lx], accim);
        }
""" % w
        elif dim == 2:
            src += f"""        for (int lx = 0; lx < {w}; ++lx) {{
            uint base = (uint)(l0x + lx) * {p[1]}u;
            float sre = 0.0f, sim = 0.0f;
            for (int ly = 0; ly < {w}; ++ly) {{
                uint idx = base + (uint)(l0y + ly);
                sre = metal::fma(tile[2*idx],   wy[ly], sre);
                sim = metal::fma(tile[2*idx+1], wy[ly], sim);
            }}
            accre = metal::fma(sre, wx[lx], accre);
            accim = metal::fma(sim, wx[lx], accim);
        }}
"""
        else:
            src += f"""        for (int lx = 0; lx < {w}; ++lx) {{
            for (int ly = 0; ly < {w}; ++ly) {{
                float wxy = wx[lx] * wy[ly];
                uint base = ((uint)(l0x + lx) * {p[1]}u + (uint)(l0y + ly))
                          * {p[2]}u;
                float sre = 0.0f, sim = 0.0f;
                for (int lz = 0; lz < {w}; ++lz) {{
                    uint idx = base + (uint)(l0z + lz);
                    sre = metal::fma(tile[2*idx],   wz[lz], sre);
                    sim = metal::fma(tile[2*idx+1], wz[lz], sim);
                }}
                accre = metal::fma(sre, wxy, accre);
                accim = metal::fma(sim, wxy, accim);
            }}
        }}
"""
        src += """        out[2*jp]   = accre;
        out[2*jp+1] = accim;
    }
"""
        innames = (["v", "perm"] + [f"ti1{_AX[d]}" for d in range(dim)]
                   + [f"tfr{_AX[d]}" for d in range(dim)]
                   + ["sub_start", "sub_count"]
                   + [f"sub_o{_AX[d]}" for d in range(dim)])
        return mx.fast.metal_kernel(
            name=f"t2gather{dim}d_od", input_names=innames,
            output_names=["out"], header="#include <metal_math>\n" + es,
            source=src)

    def _modes_to_grid(self, fkf):
        """Progressive pad+FFT (mirror of Type1PlanND._fft_modes): flat
        float32 view of the modeord box -> FFT'd fine grid (*n_up, natural
        axis order, evaluated).

        Each axis is zero-padded to nu_d immediately BEFORE its FFT, so
        early FFTs run on the still-cropped box. The axis-0 pad and the
        deconvolution ride one kernel (_pad); later pads are slice+
        concatenate writes that also cycle the padded axis to the contiguous
        last position, so every FFT is contiguous-last-axis."""
        dim, N, nu = self.dim, self.N, self.n_up
        inv = self.isign > 0
        d0 = tuple(N[d] for d in range(1, dim)) + (nu[0],)
        if dim == 3:
            g = (nu[0], N[2], N[1])
        elif dim == 2:
            g = (nu[0], N[1], 1)
        else:
            g = (nu[0], 1, 1)
        Hf = self._pad(
            inputs=[fkf] + self.mx_dec,
            output_shapes=[(int(np.prod(d0)) * 2,)],
            output_dtypes=[mx.float32],
            grid=g, threadgroup=self._tg_for(g[0]))[0]
        H = mx.view(Hf, dtype=mx.complex64).reshape(*d0)
        H = fft_axis(H, dim - 1, inverse=inv, twiddle_cache=self._twiddles)
        cyc = tuple(range(1, dim)) + (0,)
        for d in range(1, dim):
            T = mx.transpose(H, cyc)          # axis d -> contiguous last
            parts = [T[..., N[d] // 2:]]      # k = 0 .. N-N//2-1
            if nu[d] > N[d]:
                zshape = tuple(T.shape[:-1]) + (nu[d] - N[d],)
                parts.append(mx.zeros(zshape, dtype=mx.complex64))
            if N[d] // 2 > 0:
                parts.append(T[..., :N[d] // 2])   # k = -N//2 .. -1
            H = mx.concatenate(parts, axis=dim - 1)
            H = fft_axis(H, dim - 1, inverse=inv,
                         twiddle_cache=self._twiddles)
        mx.eval(H)
        return H

    def execute(self, fk, return_np=True):
        # asarray(dtype=...) is a no-op for complex64 input (astype copies);
        # mx.array materializes a contiguous buffer either way
        fmx = mx.array(np.asarray(fk, dtype=np.complex64)) \
            if not isinstance(fk, mx.array) else fk
        if fmx.size != int(np.prod(self.N)):
            raise ValueError(f"f.size ({fmx.size}) must equal the mode-box "
                             f"size {tuple(self.N)}")
        fkf = mx.view(fmx.reshape(-1), dtype=mx.float32)
        H = self._modes_to_grid(fkf)
        vf = mx.view(H, dtype=mx.float32).reshape(-1)
        if self._od:
            # output written perm-indexed -> already in caller order
            out = self._gather_od(
                inputs=[vf, self.mx_perm] + self.mx_i1 + self.mx_fr
                       + [self._mx_sub_start, self._mx_sub_count]
                       + self._mx_sub_o,
                output_shapes=[(self.P * 2,)],
                output_dtypes=[mx.float32],
                grid=(self._OD_TG, self._od_nsub, 1),
                threadgroup=(self._OD_TG, 1, 1))[0]
        else:
            out = self._gather(
                inputs=[vf] + self.mx_i1 + self.mx_fr,
                output_shapes=[(self.P * 2,)],
                output_dtypes=[mx.float32],
                grid=(max(self.P, 1), 1, 1), threadgroup=(256, 1, 1))[0]
        del H, vf
        res = mx.view(out, dtype=mx.complex64)
        if self.sorted and not self._od:
            # restore caller point order (GM path computed in sorted order)
            if not hasattr(self, "_mx_inv"):
                inv = np.empty_like(self.perm)
                inv[self.perm] = np.arange(self.P)
                self._mx_inv = mx.array(inv.astype(np.uint32))
            res = mx.take(res, self._mx_inv)
        mx.eval(res)
        return np.array(res) if return_np else res
