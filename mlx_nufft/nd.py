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

    def _grid_index_guard(self, names, dims):
        """Thread-position reads + bounds guard for an ND output box.

        Metal grid axes are (x, y, z) = (last dim, ..., first dim) so the
        fastest-varying output index rides the x lane.
        """
        s = ""
        gax = ["x", "y", "z"]
        for i, name in enumerate(reversed(names)):
            s += (f"    uint {name} = "
                  f"thread_position_in_grid.{gax[i]};\n")
        conds = " || ".join(f"{n} >= {d}u" for n, d in zip(names, dims))
        s += f"    if ({conds}) return;\n"
        return s

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
        want_od = (spread_method == "od"
                   or (spread_method == "auto" and w >= 5))
        self._od = want_od and self._od_prepare(self._OD_MSUB)
        if spread_method == "od" and not self._od:
            raise ValueError("OD spreading not applicable to this geometry")
        if self._od:
            self._spread_od = self._build_od_spread(es)

        # ---- spread: lane covers (lx[,ly]) taps; dim 3 loops lz ----------
        self._es = es
        self._lanes = w * w if dim >= 2 else w
        # GM spread is the sliceable path (its baked P is an upper guard, so it
        # runs over any point-subset launched with fewer grid rows) -> reused by
        # execute_disjoint. Built eagerly only when it is the chosen spreader.
        self._spread = None if self._od else self._build_gm_spread()

        # ---- crop FFT-order fine grid to mode box + deconvolve -----------
        mnames = [f"m{d + 1}" for d in range(dim)]
        src = self._grid_index_guard(mnames, N)
        for d in range(dim):
            src += (f"    int q{d + 1} = (int)m{d + 1} - {N[d] // 2};  "
                    f"int r{d + 1} = q{d + 1} + {nu[d]} * (q{d + 1} < 0);\n")
        src += (f"    size_t src = "
                f"{_linearize([f'r{d + 1}' for d in range(dim)], nu)};\n")
        src += (f"    size_t dst = "
                f"{_linearize(mnames, N)};\n")
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

    def _build_od_spread(self, es):
        """Output-driven spread: one threadgroup per subproblem, padded tile
        in threadgroup memory, tap-parallel accumulation (no atomics inside
        the tile), one global atomic add per tile cell at flush."""
        dim, w = self.dim, self.w
        nu = self.n_up
        m, p, ptot = self._od_m, self._od_p, self._od_ptot
        TG = self._OD_TG
        taps = w ** dim
        src = f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sub = threadgroup_position_in_grid.y;
    threadgroup float tile[{2 * ptot}];
    for (uint q = tid; q < {2 * ptot}u; q += {TG}u) tile[q] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s0 = (uint)sub_start[sub];
    uint cnt = (uint)sub_count[sub];
"""
        for d in range(dim):
            src += f"    int o{_AX[d]} = sub_o{_AX[d]}[sub];\n"
        src += """    for (uint t = 0; t < cnt; ++t) {
        uint j = s0 + t;
        uint jp = perm[j];
        float cre = cj[2*jp], cim = cj[2*jp+1];
"""
        for d in range(dim):
            a = _AX[d]
            src += (f"        int l0{a} = i1{a}[j] - o{a};  "
                    f"float f{a} = fr{a}[j];\n")
        src += f"""        for (uint tap = tid; tap < {taps}u; tap += {TG}u) {{
"""
        if dim == 1:
            src += """            int ax = (int)tap;
            float wgt = k1_es((float)ax - fx);
            uint cell = (uint)(l0x + ax);
"""
        elif dim == 2:
            src += f"""            int ax = (int)(tap / {w}u), ay = (int)(tap % {w}u);
            float wgt = k1_es((float)ax - fx) * k1_es((float)ay - fy);
            uint cell = (uint)(l0x + ax) * {p[1]}u + (uint)(l0y + ay);
"""
        else:
            src += f"""            int ax = (int)(tap / {w * w}u);
            uint rem = tap % {w * w}u;
            int ay = (int)(rem / {w}u), az = (int)(rem % {w}u);
            float wgt = k1_es((float)ax - fx) * k1_es((float)ay - fy)
                      * k1_es((float)az - fz);
            uint cell = ((uint)(l0x + ax) * {p[1]}u + (uint)(l0y + ay))
                      * {p[2]}u + (uint)(l0z + az);
"""
        src += """            tile[2*cell]   += cre * wgt;
            tile[2*cell+1] += cim * wgt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
        # flush padded tile to the global fine grid with periodic wrap
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
        atomic_fetch_add_explicit(&grid[2*cell],   tile[2*q],
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&grid[2*cell+1], tile[2*q+1],
                                  memory_order_relaxed);
    }}
"""
        innames = (["cj", "perm"] + [f"i1{_AX[d]}" for d in range(dim)]
                   + [f"fr{_AX[d]}" for d in range(dim)]
                   + ["sub_start", "sub_count"]
                   + [f"sub_o{_AX[d]}" for d in range(dim)])
        return mx.fast.metal_kernel(
            name=f"t1spread{dim}d_od", input_names=innames,
            output_names=["grid"], header="#include <metal_math>\n" + es,
            source=src, atomic_outputs=True)

    def execute(self, c, return_np=True):
        dim = self.dim
        nu_tot = int(np.prod(self.n_up))
        N_tot = int(np.prod(self.N))
        cmx = mx.array(np.asarray(c).astype(np.complex64)) \
            if not isinstance(c, mx.array) else c
        if cmx.size != self.P:
            raise ValueError(f"c.size ({cmx.size}) must equal the number of "
                             f"nonuniform points ({self.P})")
        if self._od:
            # permutation is folded into the kernel (perm-indexed reads)
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
        mx.eval(bf)
        H = self._fft_grid(bf)
        del bf
        vf = mx.view(H, dtype=mx.float32).reshape(-1)
        gdims = tuple(reversed(self.N)) + (1,) * (3 - dim)
        fk = self._crop(
            inputs=[vf] + self.mx_dec,
            output_shapes=[(N_tot * 2,)],
            output_dtypes=[mx.float32],
            grid=gdims, threadgroup=self._tg_for(gdims[0]))[0]
        del H, vf
        res = mx.view(fk, dtype=mx.complex64).reshape(*self.N)
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
            if self._od:
                if not self._od_prepare_gpu(self._OD_MSUB, i1_g, fr_g):
                    self._od = False            # no longer OD-eligible (e.g. P)
                    self._spread = self._build_gm_spread()
                    self._sort_and_upload_gpu(i1_g, fr_g)
            else:
                self._sort_and_upload_gpu(i1_g, fr_g)
        else:
            self._compute_cells(x64)
            if self._od:
                if not self._od_prepare(self._OD_MSUB):
                    self._od = False
                    self._spread = self._build_gm_spread()
                    self._sort_and_upload()
            else:
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
        dim = self.dim
        nu_tot = int(np.prod(self.n_up))
        N_tot = int(np.prod(self.N))
        lanes = self._lanes
        gdims = tuple(reversed(self.N)) + (1,) * (3 - dim)
        Bchunk = max(1, (2**31 - 1) // (nu_tot * 2))     # grid output int32 cap
        Cm = mx.array(np.ascontiguousarray(cs).astype(np.complex64))   # (Btot, P)
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
            bfc = mx.view(bf, dtype=mx.complex64).reshape(B, *self.n_up)
            del bf
            for b in range(B):
                H = bfc[b]
                for ax in range(dim - 1, -1, -1):
                    H = fft_axis(H, ax, inverse=self.isign > 0,
                                 twiddle_cache=self._twiddles)
                vf = mx.view(H, dtype=mx.float32).reshape(-1)
                fk = self._crop(
                    inputs=[vf] + self.mx_dec,
                    output_shapes=[(N_tot * 2,)], output_dtypes=[mx.float32],
                    grid=gdims, threadgroup=self._tg_for(gdims[0]))[0]
                outs.append(mx.view(fk, dtype=mx.complex64).reshape(*self.N))
            del bfc
        res = mx.stack(outs)
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
        cmx = mx.array(np.asarray(c).astype(np.complex64)) \
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
        gdims = tuple(reversed(self.N)) + (1,) * (3 - dim)
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
            H = self._fft_grid(bf)
            del bf
            vf = mx.view(H, dtype=mx.float32).reshape(-1)
            fk = self._crop(
                inputs=[vf] + self.mx_dec,
                output_shapes=[(N_tot * 2,)], output_dtypes=[mx.float32],
                grid=gdims, threadgroup=self._tg_for(gdims[0]))[0]
            del H, vf
            outs.append(mx.view(fk, dtype=mx.complex64).reshape(*self.N))
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

        # ---- pad mode box into FFT-order fine grid + deconvolve ----------
        rnames = [f"r{d + 1}" for d in range(dim)]
        src = self._grid_index_guard(rnames, nu)
        src += (f"    size_t dst = "
                f"{_linearize(rnames, nu)};\n")
        for d in range(dim):
            src += (f"    int q{d + 1} = (int)r{d + 1};  "
                    f"q{d + 1} -= {nu[d]} * "
                    f"(q{d + 1} >= {nu[d] - N[d] // 2});\n")
        conds = " && ".join(
            f"(q{d + 1} >= {-(N[d] // 2)} && q{d + 1} < {N[d] - N[d] // 2})"
            for d in range(dim))
        src += f"""    bool inband = {conds};
    if (!inband) {{ H[2*dst] = 0.0f; H[2*dst+1] = 0.0f; return; }}
"""
        for d in range(dim):
            src += f"    int m{d + 1} = q{d + 1} + {N[d] // 2};\n"
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

    def execute(self, fk, return_np=True):
        nu_tot = int(np.prod(self.n_up))
        fmx = mx.array(np.ascontiguousarray(fk).astype(np.complex64)) \
            if not isinstance(fk, mx.array) else fk
        if fmx.size != int(np.prod(self.N)):
            raise ValueError(f"f.size ({fmx.size}) must equal the mode-box "
                             f"size {tuple(self.N)}")
        fkf = mx.view(fmx.reshape(-1), dtype=mx.float32)
        gdims = tuple(reversed(self.n_up)) + (1,) * (3 - self.dim)
        Hf = self._pad(
            inputs=[fkf] + self.mx_dec,
            output_shapes=[(nu_tot * 2,)],
            output_dtypes=[mx.float32],
            grid=gdims, threadgroup=self._tg_for(gdims[0]))[0]
        mx.eval(Hf)
        H = self._fft_grid(Hf)
        del Hf
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
