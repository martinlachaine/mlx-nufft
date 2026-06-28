"""Synthetic type-3 test-problem generators and an exact direct-sum oracle.

Two families:
  gen_anisotropic - a hard, anisotropic instance with a large coordinate
      dynamic range (source coordinates ~1e7 on two axes, a thin third
      axis), which stresses fp32 coordinate handling and produces a large,
      anisotropic type-3 fine grid. This is the demanding accuracy case.
  gen_generic     - an isotropic random type-3 with modest extents, small
      enough that the CPU fp64 oracle runs at full size.
"""

import numpy as np


def gen_anisotropic(N=1024, P=100_000, lat=1.5, seed=0):
    """Anisotropic type-3 instance with a large coordinate dynamic range.

    Source coordinates reach order 1e7 on two axes (scaled by `lat`) while
    the matching target extents are order 1e-4, so the product that sets the
    type-3 fine grid is large on the two "wide" axes; the third axis is a
    thin slab (a quadratic-form coordinate), giving an anisotropic grid
    (lateral ~thousands of cells, third axis ~tens). The constants below are
    fixed and carry no physical meaning here -- they set the per-axis dynamic
    ranges that make this the worst case for single-precision coordinates.
    """
    rng = np.random.default_rng(seed)
    s0 = 520e-9                      # fixed scale constant
    k0 = 2.0 * np.pi / s0            # ~1.21e7: sets the source-coord range
    A = 7.17e-4                      # target half-width on the two wide axes
    u = rng.uniform(-lat, lat, P)
    v = rng.uniform(-lat, lat, P)
    d = rng.uniform(1.2, 6.0, P)     # per-source range factor in [1.2, 6]
    tau = 1.0 / d
    amp = rng.uniform(0.2, 1.0, P)
    th = rng.uniform(0, 2 * np.pi, P)
    c = (amp * np.exp(1j * th) / (1j * s0 * d)
         * np.exp(1j * 0.5 * k0 * tau * (u * u + v * v))).astype(np.complex128)
    # source coordinates: two wide axes (x1, x2) + a thin third axis (x3)
    x1 = -k0 * tau * v
    x2 = -k0 * tau * u
    x3 = 0.5 * k0 * tau
    # target coordinates: an N x N lattice on the two wide axes; the third
    # axis is the quadratic form of the first two (a thin slab in extent)
    p = A / N
    r = (np.arange(N) - N / 2) * p
    GY, GX = np.meshgrid(r, r, indexing="ij")
    S1 = GY.ravel()
    S2 = GX.ravel()
    S3 = (GX * GX + GY * GY).ravel()
    return dict(x=(x1, x2, x3), c=c, s=(S1, S2, S3),
                meta=dict(N=N, P=P, lat=lat, seed=seed, geom="anisotropic"))


def gen_generic(N=1024, P=100_000, seed=0, X=30.0, S=20.0):
    """Generic isotropic random 3D type-3: P sources in [-X,X]^3, M=N^2
    targets in [-S,S]^3. Default X*S keeps the fine grid ~500^3, small
    enough that the CPU fp64 oracle runs at full size."""
    rng = np.random.default_rng(seed)
    M = N * N
    x = tuple(rng.uniform(-X, X, P) for _ in range(3))
    s = tuple(rng.uniform(-S, S, M) for _ in range(3))
    a = rng.uniform(0.2, 1.0, P)
    th = rng.uniform(0, 2 * np.pi, P)
    c = a * np.exp(1j * th)
    return dict(x=x, c=c, s=s,
                meta=dict(N=N, P=P, X=X, S=S, seed=seed, geom="generic"))


def uniform_points(dim, N, rho=1.0, dist="rand", sigma=2.0, seed=0):
    """Nonuniform points for a uniform-mode (type 1/2) problem, cuFINUFFT-style.

    Returns (x, n_modes, M) where x is a list of `dim` fp64 coordinate arrays in
    [-pi, pi), n_modes = (N,)*dim, and M = round(rho * sigma^dim * N^dim) is set
    so the density rho = M / (sigma^dim * prod(N)) matches cuFINUFFT's definition.

    dist="rand":    points uniform on [-pi, pi)^dim.
    dist="cluster": points uniform in a small box of side 8h at the domain
                    corner, h = 2*pi/(sigma*N) the fine-grid spacing (the
                    clustered worst case from the cuFINUFFT paper).
    """
    rng = np.random.default_rng(seed)
    M = int(round(rho * (sigma ** dim) * (N ** dim)))
    if dist == "rand":
        x = [rng.uniform(-np.pi, np.pi, M) for _ in range(dim)]
    elif dist == "cluster":
        h = 2.0 * np.pi / (sigma * N)
        x = [(-np.pi + rng.uniform(0.0, 8.0 * h, M)) for _ in range(dim)]
    else:
        raise ValueError(f"unknown dist {dist!r}")
    return [v.astype(np.float64) for v in x], (N,) * dim, M


def direct_sum(x, c, s, isign=+1, idx=None, chunk=None):
    """Exact fp64 direct summation oracle: f[k] = sum_j c_j e^{i*isign*s_k.x_j}.

    idx: optional target subset indices (for the subset oracle at full size).
    chunk: target rows per block; default keeps the chunk x P phase matrix
    around ~160 MB so workers stay memory-light at P=1e6.
    """
    s1, s2, s3 = s
    if idx is not None:
        s1, s2, s3 = s1[idx], s2[idx], s3[idx]
    x1, x2, x3 = (np.asarray(v, dtype=np.float64) for v in x)
    c = np.asarray(c, dtype=np.complex128)
    M = s1.size
    if chunk is None:
        chunk = max(8, int(2e7 / max(x1.size, 1)))
    out = np.empty(M, dtype=np.complex128)
    for a0 in range(0, M, chunk):
        sl = slice(a0, min(a0 + chunk, M))
        ph = (np.outer(s1[sl], x1) + np.outer(s2[sl], x2)
              + np.outer(s3[sl], x3))
        out[sl] = np.exp(1j * isign * ph) @ c
    return out


def rel_l2(u, ref):
    u = np.asarray(u).ravel()
    ref = np.asarray(ref).ravel()
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def _ds_worker(args):
    x, c, s, isign, idx = args
    return direct_sum(x, c, s, isign=isign, idx=idx)


def direct_sum_mp(x, c, s, isign=+1, idx=None, workers=7):
    """Parallel exact fp64 direct sum over a target subset."""
    from concurrent.futures import ProcessPoolExecutor
    if idx is None:
        idx = np.arange(s[0].size)
    chunks = np.array_split(idx, workers)
    args = [(x, c, s, isign, ch) for ch in chunks if ch.size]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_ds_worker, args))
    return np.concatenate(parts)
