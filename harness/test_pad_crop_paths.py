"""Bit-identity of the ND type-2 pad paths (nd.PAD_PATH "fused" vs "v020")
and of the type-1 crop chain under the same switch.

"fused" replaces the 0.2.0 pads of Type2PlanND._modes_to_grid (mx.zeros +
mx.concatenate of two transposed slices per axis, and in dims 2/3 the
per-thread axis-0 pad + deconvolution kernel) with one tiled pass per axis
(nd._pad_kernel). That is pure data movement plus the same deconvolution
product, so the gate is exact: every bit of Type2PlanND.execute's complex64
output must agree between the two paths, in dims 1/2/3, at eps 1e-3 and
1e-5, for odd and even mode counts, both isigns, sigma 2 and 1.25, mode and
grid extents that are not multiples of the 32-wide tiles, N = 1 and N = 2
axes (no negative block, one-row blocks) and a plan re-executed in place.
One 3D case is also held against CPU FINUFFT. A structural check confirms
the switch selects different kernels (no Concatenate / Full in the fused
stage graph, no custom kernel in the v020 one), and subprocesses check the
MLX_NUFFT_PAD_PATH environment default.

Type 1 does not read the switch (its crop gathers are already one pass
each, riding the transposes MLX's last-axis FFT needs); its rows pin that
down for execute and for the batched execute_batch behind Plan(n_trans=4).
They run on lattice points spaced wider than the kernel and clear of the
periodic seam, so every fine cell receives at most one spread contribution
and the atomic spread is order-independent: exact equality is meaningful
there, and the batched result is also held bit for bit against the
per-vector loop.
"""

import os
import re
import sys
import pathlib
import tempfile
import subprocess

import numpy as np
import mlx.core as mx
import finufft as cpu

ROOT = pathlib.Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT))
import mlx_nufft as gpu                                  # noqa: E402
import mlx_nufft.nd as nd                                # noqa: E402
from mlx_nufft.nd import Type1PlanND, Type2PlanND        # noqa: E402
from harness.gen import rel_l2                           # noqa: E402

rng = np.random.default_rng(41)
PATHS = ("v020", "fused")
FAILS = []

# (N, eps, isign, upsampfac); P = 3000 random targets each
T2_CASES = [
    ((1000,), 1e-3, -1, 2.0),
    ((1001,), 1e-5, +1, 2.0),
    ((2 ** 16,), 1e-3, -1, 2.0),
    ((64, 45), 1e-3, -1, 2.0),
    ((63, 44), 1e-5, +1, 2.0),
    ((33, 17), 1e-5, -1, 1.25),
    ((24, 33, 20), 1e-3, -1, 2.0),
    ((25, 32, 21), 1e-5, +1, 2.0),
    ((64, 60, 56), 1e-5, -1, 1.25),
    ((128, 96, 80), 1e-3, -1, 2.0),
    ((1, 5, 7), 1e-3, -1, 2.0),
    ((2, 1, 3), 1e-5, +1, 2.0),
]
# (N, eps): type 1 on lattice points, GM spread (P < 20000), n_trans=4
T1_CASES = [
    ((4097,), 1e-3),
    ((4096,), 1e-5),
    ((33, 32), 1e-3),
    ((32, 31, 30), 1e-5),
]


def check_true(label, ok):
    print(f"  {'PASS' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)


def bits(a):
    """Raw bit pattern of a complex64 array (np.array_equal on the values
    would let +0 and -0 agree)."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.complex64)
                                ).view(np.uint32)


def same_bits(a, b):
    return np.shape(a) == np.shape(b) and np.array_equal(bits(a), bits(b))


def with_path(path, fn):
    saved = nd.PAD_PATH
    nd.PAD_PATH = path
    try:
        return fn()
    finally:
        nd.PAD_PATH = saved


def pts(dim, n):
    return [rng.uniform(-np.pi, np.pi, n) for _ in range(dim)]


def cplx(shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex64)


def lattice(n_up, s=8):
    """One point at the centre of every s-th fine cell per axis, from cell 4
    to nu - 5: supports of width w <= s never overlap and never wrap."""
    axes = [(np.arange(4, nu - 4, s) + 0.5) * (2.0 * np.pi / nu)
            for nu in n_up]
    return [m.ravel() for m in np.meshgrid(*axes, indexing="ij")]


def graph_prims(*arrs):
    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "g.dot"
        mx.export_to_dot(str(f), *arrs)
        return re.findall(r'label\s*="([^"]+)"', f.read_text())


if __name__ == "__main__":
    print("== 1. type 2: fused vs v020, bit for bit ==")
    for N, eps, isign, sigma in T2_CASES:
        dim = len(N)
        x = pts(dim, 3000)
        plan = Type2PlanND(x, N, eps=eps, isign=isign, upsampfac=sigma)
        fk = cplx(N)
        out = {p: with_path(p, lambda: plan.execute(fk)) for p in PATHS}
        again = with_path("fused", lambda: plan.execute(fk))
        tag = (f"t2 {dim}d N={N} nu={tuple(plan.n_up)} eps={eps:.0e} "
               f"isign={isign:+d} sigma={sigma}")
        check_true(f"{tag}: fused == v020",
                   same_bits(out["fused"], out["v020"]))
        check_true(f"{tag}: fused re-execute identical",
                   same_bits(out["fused"], again))
        check_true(f"{tag}: finite", bool(np.all(np.isfinite(out["fused"]))))
        if N == (24, 33, 20):
            ref = cpu.nufft3d2(*x, fk.astype(np.complex128), eps=1e-9,
                               isign=isign)
            err = rel_l2(out["fused"], ref)
            check_true(f"{tag}: vs cpu rel_l2={err:.2e}",
                       err <= 4 * eps + 2e-4)

    print("== 2. the switch selects different kernels ==")
    plan = Type2PlanND(pts(3, 500), (24, 33, 20), eps=1e-3)
    fkm = mx.array(cplx(plan.N))
    mx.eval(fkm)
    fkf = mx.view(fkm.reshape(-1), dtype=mx.float32)
    for p in PATHS:
        H0 = with_path(p, lambda: plan._pad_axis(fkf, 0))
        p0 = graph_prims(H0)
        mx.eval(H0)
        p1 = graph_prims(with_path(p, lambda: plan._pad_axis(H0, 1)))
        if p == "fused":
            ok = ("CustomKernel" in p0 and "CustomKernel" in p1
                  and "Concatenate" not in p1 and "Full" not in p1)
        else:
            ok = ("CustomKernel" in p0 and "Concatenate" in p1
                  and "Full" in p1 and "CustomKernel" not in p1)
        check_true(f"{p}: axis-0 {p0} / axis-1 {p1}", ok)
    try:
        nd.PAD_PATH = "bogus"
        plan._pad_axis(fkf, 0)
        check_true("PAD_PATH='bogus' raises ValueError", False)
    except ValueError:
        check_true("PAD_PATH='bogus' raises ValueError", True)
    finally:
        nd.PAD_PATH = "fused"

    print("== 3. MLX_NUFFT_PAD_PATH sets the module default ==")
    for val in (None, "v020", "fused"):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        env.pop("MLX_NUFFT_PAD_PATH", None)
        if val is not None:
            env["MLX_NUFFT_PAD_PATH"] = val
        got = subprocess.run(
            [sys.executable, "-c",
             "import mlx_nufft.nd as nd; print(nd.PAD_PATH)"],
            env=env, capture_output=True, text=True).stdout.strip()
        want = "fused" if val is None else val
        check_true(f"env {'unset' if val is None else val} -> {got}",
                   got == want)

    print("== 4. type 1 (crop chain, unswitched): execute and "
          "Plan(n_trans=4) bit for bit ==")
    for N, eps in T1_CASES:
        dim = len(N)
        probe = Type1PlanND(pts(dim, 16), N, eps=eps)
        x = lattice(probe.n_up)
        P = x[0].size
        cs = cplx((4, P))
        plan = Type1PlanND(x, N, eps=eps)
        tag = f"t1 {dim}d N={N} nu={tuple(plan.n_up)} eps={eps:.0e} P={P}"
        check_true(f"{tag}: GM spread path", plan._spread is not None)
        one = {p: with_path(p, lambda: plan.execute(cs[0])) for p in PATHS}
        check_true(f"{tag}: execute fused == v020",
                   same_bits(one["fused"], one["v020"]))
        check_true(f"{tag}: execute re-run identical (lattice spread)",
                   same_bits(one["fused"], plan.execute(cs[0])))
        pub = gpu.Plan(1, N, n_trans=4, eps=eps, isign=+1, dtype="complex64")
        pub.setpts(*x)
        bat = {p: with_path(p, lambda: pub.execute(cs)) for p in PATHS}
        check_true(f"{tag}: Plan(n_trans=4) fused == v020",
                   same_bits(bat["fused"], bat["v020"]))
        check_true(f"{tag}: Plan ran execute_batch (B=4 spread kernel built)",
                   4 in getattr(pub._plan, "_spread_batch", {}))
        loop = np.stack([pub._plan.execute(cs[b]) for b in range(4)])
        check_true(f"{tag}: Plan(n_trans=4) == per-vector loop",
                   same_bits(bat["fused"], loop))
        ref = getattr(cpu, f"nufft{dim}d1")(*x, cs[0].astype(np.complex128),
                                            N, eps=1e-9, isign=+1)
        err = rel_l2(one["fused"], ref)
        check_true(f"{tag}: vs cpu rel_l2={err:.2e}", err <= 4 * eps + 2e-4)

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
