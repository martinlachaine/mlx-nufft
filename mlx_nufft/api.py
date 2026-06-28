"""finufft-compatible API for mlx_nufft.

Drop-in mirror of the `finufft` Python package's interface, so existing
callers switch with one import line:

    import mlx_nufft as finufft
    fk = finufft.nufft2d1(x, y, c, (N1, N2), eps=1e-6)
    plan = finufft.Plan(1, (N1, N2), n_trans=8, eps=1e-6)
    plan.setpts(x, y)
    fk = plan.execute(c)

Semantics mirrored from finufft 2.x:
  - mode boxes are modeord=0: k_d integer in [-(N_d//2), (N_d-1)//2],
    even or odd N_d;
  - isign: non-negative means +i in the exponential (type-1/3 default +1,
    type-2 default -1);
  - multi-vector inputs: leading n_trans axis on strengths/mode arrays;
  - out= arrays are filled in place when supplied;
  - x (etc.) in [-pi, pi), folded otherwise.

Differences (documented, not silent):
  - computation is the validated fp32 GPU pipeline with fp64-critical
    setup ('crit64'); requesting eps below 1e-6 clamps to 1e-6 with a
    warning (the fp32 accuracy envelope: see ACCEPTANCE.md);
  - complex128 inputs are accepted and returned as complex128, but the
    transform itself is fp32-grade;
  - modeord=1 (FFT ordering) is not implemented (raises);
  - 1D/2D type 3 currently run as degenerate slices of the validated 3D
    type-3 kernel (functional; not speed-tuned).
"""

import warnings

import numpy as np

from .nd import Type1PlanND, Type2PlanND
from .gpu_t3 import GpuT3Plan

_EPS_FLOOR = 1e-6
_IGNORED_OPTS = {
    "nthreads", "debug", "spread_debug", "showwarn", "fftw", "spread_sort",
    "spread_kerevalmeth", "spread_kerpad", "chkbnds", "maxbatchsize",
    "spread_thread", "spread_nthr_atomic", "spread_max_sp_size",
}


def _check_opts(kwargs):
    opts = dict(kwargs)
    if opts.pop("modeord", 0) not in (0,):
        raise NotImplementedError("modeord=1 (FFT ordering) not implemented")
    upsampfac = opts.pop("upsampfac", None)
    if upsampfac in (0, 0.0):                 # finufft auto sentinel
        upsampfac = None
    prec = opts.pop("prec", "crit64")
    fft_backend = opts.pop("fft_backend", "mlx")   # type-3 slab only
    for k in list(opts):
        if k in _IGNORED_OPTS:
            opts.pop(k)
    if opts:
        warnings.warn(f"mlx-nufft: ignoring unknown options {sorted(opts)}")
    return upsampfac, prec, fft_backend


def _norm_eps(eps):
    eps = float(eps)
    if eps < _EPS_FLOOR:
        warnings.warn(
            f"mlx-nufft: eps={eps:g} is below the fp32 pipeline floor; "
            f"clamping to {_EPS_FLOOR:g} (see ACCEPTANCE.md accuracy notes)")
        eps = _EPS_FLOOR
    return eps


def _norm_isign(isign, default):
    if isign is None:
        return default
    return +1 if isign >= 0 else -1


def _out_dtype(*arrays):
    for a in arrays:
        if np.asarray(a).dtype in (np.complex128, np.float64):
            return np.complex128
    return np.complex64


def _vec_shape(data, inner_ndim, inner_shape=None):
    """Split data shape into (n_trans, inner shape); inner_ndim trailing.
    If inner_shape is given, the trailing dims must match it exactly
    (mirrors FINUFFT's strict size checks — no silent truncation)."""
    data = np.asarray(data)
    if data.ndim == inner_ndim:
        out = 1, data[None, ...]
    elif data.ndim == inner_ndim + 1:
        out = data.shape[0], data
    else:
        raise ValueError(f"data must have {inner_ndim} or {inner_ndim + 1} "
                         f"dims, got shape {data.shape}")
    if inner_shape is not None and out[1].shape[1:] != tuple(inner_shape):
        raise ValueError(f"data inner shape {out[1].shape[1:]} must be "
                         f"{tuple(inner_shape)}")
    return out


def _fill_out(out, res, dtype):
    res = res.astype(dtype, copy=False)
    if out is not None:
        # exact shape, or the (1, ...) stacked form when n_trans == 1
        if out.shape != res.shape and out.shape != (1,) + res.shape \
                and (1,) + out.shape != res.shape:
            raise ValueError(f"out.shape {out.shape} does not match result "
                             f"shape {res.shape}")
        np.copyto(out, res.reshape(out.shape))
        return out
    return res


def _embed3(arrs, dim):
    """Zero-pad a dim<3 coordinate tuple to 3 components for GpuT3Plan."""
    arrs = [np.asarray(a, dtype=np.float64).ravel() for a in arrs]
    z = np.zeros(arrs[0].size)
    return tuple(arrs) + (z,) * (3 - dim)


def _modes_tuple(n_modes, dim, out, out_offset=0):
    if n_modes is None:
        if out is None:
            raise ValueError("either n_modes or out must be supplied")
        shape = out.shape[out_offset:]
        if len(shape) != dim:
            raise ValueError(f"out shape {out.shape} does not match dim {dim}")
        return tuple(int(n) for n in shape)
    if np.isscalar(n_modes):
        return (int(n_modes),) * dim
    return tuple(int(n) for n in n_modes)


# ---------------------------------------------------------------------------
# type 1: nonuniform -> uniform


def _warn_no_vkfft(fft_backend, what):
    if fft_backend != "mlx":
        warnings.warn(f"mlx-nufft: fft_backend={fft_backend!r} applies only "
                      f"to 3D type-3 (slab); ignored for {what}")


def _nufft_t1(dim, coords, c, n_modes, out, eps, isign, kwargs):
    upsampfac, prec, fft_backend = _check_opts(kwargs)
    _warn_no_vkfft(fft_backend, "type-1")
    eps = _norm_eps(eps)
    isign = _norm_isign(isign, +1)
    dtype = _out_dtype(c)
    M = np.asarray(coords[0]).size
    n_tr, cv = _vec_shape(c, 1, inner_shape=(M,))
    if out is not None and n_modes is None:
        n_modes = _modes_tuple(None, dim, out, out_offset=(1 if n_tr > 1 else 0))
    N = _modes_tuple(n_modes, dim, out)
    kw = {} if upsampfac is None else {"upsampfac": upsampfac}
    plan = Type1PlanND(coords, N, eps=eps, isign=isign, prec=prec, **kw)
    res = np.stack([plan.execute(cv[t]) for t in range(n_tr)])
    if n_tr == 1 and (np.asarray(c).ndim == 1):
        res = res[0]
    return _fill_out(out, res, dtype)


def nufft1d1(x, c, n_modes=None, out=None, eps=1e-6, isign=1, **kwargs):
    """1D type-1: f[k] = sum_j c[j] exp(+/-i k x(j))."""
    return _nufft_t1(1, (x,), c, n_modes, out, eps, isign, kwargs)


def nufft2d1(x, y, c, n_modes=None, out=None, eps=1e-6, isign=1, **kwargs):
    """2D type-1: f[k1,k2] = sum_j c[j] exp(+/-i (k1 x(j) + k2 y(j)))."""
    return _nufft_t1(2, (x, y), c, n_modes, out, eps, isign, kwargs)


def nufft3d1(x, y, z, c, n_modes=None, out=None, eps=1e-6, isign=1, **kwargs):
    """3D type-1: f[k1,k2,k3] = sum_j c[j] exp(+/-i k . x_j)."""
    return _nufft_t1(3, (x, y, z), c, n_modes, out, eps, isign, kwargs)


# ---------------------------------------------------------------------------
# type 2: uniform -> nonuniform


def _nufft_t2(dim, coords, f, out, eps, isign, kwargs):
    upsampfac, prec, fft_backend = _check_opts(kwargs)
    _warn_no_vkfft(fft_backend, "type-2")
    eps = _norm_eps(eps)
    isign = _norm_isign(isign, -1)
    dtype = _out_dtype(f)
    n_tr, fv = _vec_shape(f, dim)
    N = fv.shape[1:]
    kw = {} if upsampfac is None else {"upsampfac": upsampfac}
    plan = Type2PlanND(coords, N, eps=eps, isign=isign, prec=prec, **kw)
    res = np.stack([plan.execute(fv[t]) for t in range(n_tr)])
    if n_tr == 1 and (np.asarray(f).ndim == dim):
        res = res[0]
    return _fill_out(out, res, dtype)


def nufft1d2(x, f, out=None, eps=1e-6, isign=-1, **kwargs):
    """1D type-2: c[j] = sum_k f[k] exp(+/-i k x(j))."""
    return _nufft_t2(1, (x,), f, out, eps, isign, kwargs)


def nufft2d2(x, y, f, out=None, eps=1e-6, isign=-1, **kwargs):
    """2D type-2: c[j] = sum_{k1,k2} f[k1,k2] exp(+/-i (k1 x(j) + k2 y(j)))."""
    return _nufft_t2(2, (x, y), f, out, eps, isign, kwargs)


def nufft3d2(x, y, z, f, out=None, eps=1e-6, isign=-1, **kwargs):
    """3D type-2: c[j] = sum_k f[k] exp(+/-i k . x_j)."""
    return _nufft_t2(3, (x, y, z), f, out, eps, isign, kwargs)


# ---------------------------------------------------------------------------
# type 3: nonuniform -> nonuniform


def _nufft_t3(dim, src, c, trg, out, eps, isign, kwargs):
    upsampfac, prec, fft_backend = _check_opts(kwargs)
    if upsampfac is not None and upsampfac != 1.25:
        warnings.warn("mlx-nufft: type-3 runs the validated sigma=1.25 "
                      "pipeline; upsampfac ignored")
    eps = _norm_eps(eps)
    isign = _norm_isign(isign, +1)
    dtype = _out_dtype(c)
    M = np.asarray(src[0]).size
    n_tr, cv = _vec_shape(c, 1, inner_shape=(M,))
    plan = GpuT3Plan(_embed3(src, dim), _embed3(trg, dim),
                     eps=eps, isign=isign, prec=prec, fft_backend=fft_backend)
    res = np.stack([plan.execute(cv[t]) for t in range(n_tr)])
    if n_tr == 1 and (np.asarray(c).ndim == 1):
        res = res[0]
    return _fill_out(out, res, dtype)


def nufft1d3(x, c, s, out=None, eps=1e-6, isign=1, **kwargs):
    """1D type-3: f[k] = sum_j c[j] exp(+/-i s[k] x[j])."""
    return _nufft_t3(1, (x,), c, (s,), out, eps, isign, kwargs)


def nufft2d3(x, y, c, s, t, out=None, eps=1e-6, isign=1, **kwargs):
    """2D type-3: f[k] = sum_j c[j] exp(+/-i (s[k] x[j] + t[k] y[j]))."""
    return _nufft_t3(2, (x, y), c, (s, t), out, eps, isign, kwargs)


def nufft3d3(x, y, z, c, s, t, u, out=None, eps=1e-6, isign=1, **kwargs):
    """3D type-3: f[k] = sum_j c[j] exp(+/-i (s,t,u)_k . (x,y,z)_j)."""
    return _nufft_t3(3, (x, y, z), c, (s, t, u), out, eps, isign, kwargs)


# ---------------------------------------------------------------------------
# Plan interface


class Plan:
    """finufft.Plan-compatible plan/setpts/execute interface.

    Plan(nufft_type, n_modes_or_dim, n_trans=1, eps=1e-6, isign=None,
         dtype='complex128', **kwargs)

    For types 1/2, n_modes_or_dim is the mode tuple (dim inferred from its
    length). For type 3 it is the dimension (1, 2 or 3). setpts() builds the
    GPU plan (points are part of plan state, as in cu/FINUFFT); execute()
    runs each of n_trans vectors through the cached plan.
    """

    def __init__(self, nufft_type, n_modes_or_dim, n_trans=1, eps=1e-6,
                 isign=None, dtype="complex128", **kwargs):
        if nufft_type not in (1, 2, 3):
            raise ValueError("nufft_type must be 1, 2 or 3")
        self.type = int(nufft_type)
        self.n_trans = int(n_trans)
        self.eps = _norm_eps(eps)
        self.isign = _norm_isign(isign, -1 if self.type == 2 else +1)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.complex64, np.complex128):
            raise ValueError("dtype must be complex64 or complex128")
        self._upsampfac, self._prec, self._fft_backend = _check_opts(kwargs)
        if self._fft_backend != "mlx" and self.type != 3:
            _warn_no_vkfft(self._fft_backend, f"type-{self.type}")
            self._fft_backend = "mlx"
        if self.type == 3:
            self.dim = int(n_modes_or_dim)
            self.n_modes = None
        else:
            if np.isscalar(n_modes_or_dim):
                n_modes_or_dim = (n_modes_or_dim,)
            self.n_modes = tuple(int(n) for n in n_modes_or_dim)
            self.dim = len(self.n_modes)
        if self.dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2 or 3")
        self._plan = None
        self._adjoint = None
        self._n_targets = None

    def setpts(self, x=None, y=None, z=None, s=None, t=None, u=None):
        coords = [v for v in (x, y, z) if v is not None]
        if len(coords) != self.dim:
            raise ValueError(f"expected {self.dim} coordinate arrays, "
                             f"got {len(coords)}")
        kw = {} if self._upsampfac is None else {"upsampfac": self._upsampfac}
        self._adjoint = None
        if self.type == 1:
            self._plan = Type1PlanND(tuple(coords), self.n_modes,
                                     eps=self.eps, isign=self.isign,
                                     prec=self._prec, **kw)
        elif self.type == 2:
            self._plan = Type2PlanND(tuple(coords), self.n_modes,
                                     eps=self.eps, isign=self.isign,
                                     prec=self._prec, **kw)
        else:
            if self._upsampfac is not None and self._upsampfac != 1.25:
                warnings.warn("mlx-nufft: type-3 runs the validated "
                              "sigma=1.25 pipeline; upsampfac ignored")
            trg = [v for v in (s, t, u) if v is not None]
            if len(trg) != self.dim:
                raise ValueError(f"expected {self.dim} target arrays, "
                                 f"got {len(trg)}")
            self._plan = GpuT3Plan(_embed3(coords, self.dim),
                                   _embed3(trg, self.dim),
                                   eps=self.eps, isign=self.isign,
                                   prec=self._prec, fft_backend=self._fft_backend)
            self._n_targets = np.asarray(trg[0]).size
        self._coords = [np.asarray(v) for v in coords]
        self._targets = None if self.type != 3 else \
            [np.asarray(v) for v in (s, t, u) if v is not None]

    def execute(self, data, out=None):
        if self._plan is None:
            raise RuntimeError("setpts() must be called before execute()")
        if self.type == 2:
            inner, ishape = self.dim, self.n_modes
        else:
            inner, ishape = 1, (self._plan.P,)
        n_tr, dv = _vec_shape(data, inner, inner_shape=ishape)
        if n_tr != self.n_trans:
            raise ValueError(f"data has {n_tr} vectors, plan has "
                             f"n_trans={self.n_trans}")
        res = np.stack([self._plan.execute(dv[k]) for k in range(self.n_trans)])
        if self.n_trans == 1 and np.asarray(data).ndim == inner:
            res = res[0]
        return _fill_out(out, res, self.dtype)

    def execute_adjoint(self, data, out=None):
        """Apply the adjoint of the planned transform (finufft 2.5 API).

        Type-1 plan adjoint maps modes -> points; type-2 adjoint maps
        points -> modes; type-3 adjoint maps targets -> sources. Implemented
        as the sibling transform with isign negated (the exact adjoint of
        the NUFFT matrix; the validated type-3 adjoint identity)."""
        if self._plan is None:
            raise RuntimeError("setpts() must be called before "
                               "execute_adjoint()")
        if self._adjoint is None:
            kw = {} if self._upsampfac is None \
                else {"upsampfac": self._upsampfac}
            if self.type == 1:
                self._adjoint = Type2PlanND(tuple(self._coords), self.n_modes,
                                            eps=self.eps, isign=-self.isign,
                                            prec=self._prec, **kw)
            elif self.type == 2:
                self._adjoint = Type1PlanND(tuple(self._coords), self.n_modes,
                                            eps=self.eps, isign=-self.isign,
                                            prec=self._prec, **kw)
            else:
                self._adjoint = GpuT3Plan(_embed3(self._targets, self.dim),
                                          _embed3(self._coords, self.dim),
                                          eps=self.eps, isign=-self.isign,
                                          prec=self._prec,
                                          fft_backend=self._fft_backend)
        if self.type == 1:
            inner, ishape = self.dim, self.n_modes
        elif self.type == 2:
            inner, ishape = 1, (self._plan.P,)
        else:
            inner, ishape = 1, (self._n_targets,)
        n_tr, dv = _vec_shape(data, inner, inner_shape=ishape)
        if n_tr != self.n_trans:
            raise ValueError(f"data has {n_tr} vectors, plan has "
                             f"n_trans={self.n_trans}")
        res = np.stack([self._adjoint.execute(dv[k])
                        for k in range(self.n_trans)])
        if self.n_trans == 1 and np.asarray(data).ndim == inner:
            res = res[0]
        return _fill_out(out, res, self.dtype)
