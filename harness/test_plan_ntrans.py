"""Plan n_trans batching: the finufft-compatible Plan sends its n_trans
type-1 strength vectors through Type1PlanND.execute_batch (one shared spread
launch per chunk, whole-chunk per-axis FFTs) instead of one execute() per
vector. This checks that the batched Plan is indistinguishable from the
per-vector loop it replaced, and keeps the finufft parity gate.

Gates (hard), dims 1/2/3, n_trans in {1, 2, 4, 8}, eps=1e-5:
  - Plan(n_trans=B).execute(cs)[b] == Plan(n_trans=1).execute(cs[b]) to fp32
    roundoff (rel_l2 <= 1e-6 per vector: identical ES weights, only the
    atomic accumulation order differs)
  - every vector ~= CPU FINUFFT fp64 at the same eps (thr = 4*eps + 2e-4)
  - out= is filled in place, complex128 is accepted and returned as
    complex128, and a wrong vector count / point count / out shape raise
    the same ValueErrors as before
  - the OD spread path (P >= 20000) through the Plan, the memory-chunked
    path (several execute_batch calls per execute), the type-2 Plan adjoint
    (type-1 machinery) and the functional nufft*d1 with stacked strengths
"""

import sys
import pathlib

import numpy as np
import finufft as cpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
import mlx_nufft as gpu                                  # noqa: E402
import mlx_nufft.api as api                              # noqa: E402
from harness.gen import rel_l2                           # noqa: E402

rng = np.random.default_rng(23)
EPS = 1e-5
LOOP_THR = 1e-6                  # batched vs looped: fp32 atomic-order noise
CPU_THR = 4 * EPS + 2e-4         # vs CPU FINUFFT fp64 (as test_api_parity)
FAILS = []


def check(label, err, thr):
    ok = err <= thr
    print(f"  {'PASS' if ok else 'FAIL'} {label}: rel_l2={err:.3e} "
          f"(thr {thr:.1e})")
    if not ok:
        FAILS.append(label)


def check_true(label, ok):
    print(f"  {'PASS' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)


def pts(dim, n):
    return [rng.uniform(-np.pi, np.pi, n) for _ in range(dim)]


def strengths(nt, n):
    return (rng.standard_normal((nt, n))
            + 1j * rng.standard_normal((nt, n))).astype(np.complex64)


def t1_plan(N, nt, dtype="complex64"):
    return gpu.Plan(1, N, n_trans=nt, eps=EPS, isign=+1, dtype=dtype)


def looped(N, x, cs):
    """The per-vector path the batched Plan replaced: an n_trans=1 Plan over
    the same points, one execute per strength vector."""
    p1 = t1_plan(N, 1)
    p1.setpts(*x)
    return np.stack([p1.execute(cs[b]) for b in range(cs.shape[0])])


def cpu_ref(N, x, cs):
    fn = getattr(cpu, f"nufft{len(x)}d1")
    return np.stack([fn(*x, cs[b].astype(np.complex128), N, eps=EPS, isign=+1)
                     for b in range(cs.shape[0])])


def max_vs(got, ref):
    """Worst per-vector rel_l2 over the leading n_trans axis."""
    return max(rel_l2(got[b], ref[b]) for b in range(got.shape[0]))


def raises(label, fn, needle):
    try:
        fn()
    except ValueError as e:
        check_true(f"{label}: ValueError({e})", needle in str(e))
    else:
        check_true(f"{label}: no ValueError raised", False)


if __name__ == "__main__":
    M = 4000
    print(f"== Plan type 1, eps={EPS:g}: batched vs per-vector loop, vs CPU ==")
    for dim, N in [(1, (90,)), (2, (48, 36)), (3, (24, 20, 16))]:
        x = pts(dim, M)
        for nt in (1, 2, 4, 8):
            cs = strengths(nt, M)
            p = t1_plan(N, nt)
            p.setpts(*x)
            got = p.execute(cs)
            assert got.shape == (nt,) + N and got.dtype == np.complex64
            tag = f"t1 {dim}d N={N} n_trans={nt}"
            check(f"{tag} vs loop", max_vs(got, looped(N, x, cs)), LOOP_THR)
            check(f"{tag} vs cpu", max_vs(got, cpu_ref(N, x, cs)), CPU_THR)

    print("== OD spread path (P >= 20000) through the Plan ==")
    P = 50_000
    for dim, N in [(1, (4096,)), (2, (128, 128)), (3, (32, 32, 32))]:
        x = pts(dim, P)
        cs = strengths(4, P)
        p = t1_plan(N, 4)
        p.setpts(*x)
        got = p.execute(cs)
        tag = f"t1 {dim}d N={N} P={P} n_trans=4"
        check(f"{tag} vs loop", max_vs(got, looped(N, x, cs)), LOOP_THR)
        check(f"{tag} vs cpu", max_vs(got, cpu_ref(N, x, cs)), CPU_THR)

    print("== memory-chunked batching: 3 vectors per execute_batch call ==")
    N = (48, 36)
    x = pts(2, M)
    cs = strengths(8, M)
    p = t1_plan(N, 8)
    p.setpts(*x)
    whole = p.execute(cs)
    chunk_fn = api._t1_batch_chunk
    api._t1_batch_chunk = lambda plan, n_tr: 3
    try:
        chunked = p.execute(cs)
    finally:
        api._t1_batch_chunk = chunk_fn
    check("t1 2d n_trans=8 chunks of 3 vs one call", max_vs(chunked, whole),
          LOOP_THR)

    print("== out=, complex128, shape errors ==")
    cs = strengths(4, M)
    p = t1_plan(N, 4)
    p.setpts(*x)
    ref = looped(N, x, cs)
    out = np.empty((4,) + N, dtype=np.complex64)
    got = p.execute(cs, out=out)
    assert got is out
    check("t1 2d n_trans=4 out= filled in place vs loop", max_vs(out, ref),
          LOOP_THR)
    p128 = t1_plan(N, 4, dtype="complex128")
    p128.setpts(*x)
    got128 = p128.execute(cs.astype(np.complex128))
    assert got128.dtype == np.complex128 and got128.shape == (4,) + N
    check("t1 2d n_trans=4 complex128 in/out vs loop", max_vs(got128, ref),
          LOOP_THR)
    out128 = np.empty((4,) + N, dtype=np.complex128)
    assert p128.execute(cs.astype(np.complex128), out=out128) is out128
    check("t1 2d n_trans=4 complex128 out= vs loop", max_vs(out128, ref),
          LOOP_THR)
    raises("wrong vector count", lambda: p.execute(cs[:3]), "n_trans=4")
    raises("wrong point count", lambda: p.execute(cs[:, :-1]), "inner shape")
    raises("wrong out shape",
           lambda: p.execute(cs, out=np.empty((3,) + N, dtype=np.complex64)),
           "out.shape")

    print("== type-2 Plan adjoint (type-1 machinery), functional nufft2d1 ==")
    p2 = gpu.Plan(2, N, n_trans=4, eps=EPS, isign=-1, dtype="complex64")
    p2.setpts(*x)
    adj = p2.execute_adjoint(cs)
    p21 = gpu.Plan(2, N, n_trans=1, eps=EPS, isign=-1, dtype="complex64")
    p21.setpts(*x)
    adj_loop = np.stack([p21.execute_adjoint(cs[b]) for b in range(4)])
    check("t2 2d n_trans=4 execute_adjoint vs loop", max_vs(adj, adj_loop),
          LOOP_THR)
    check("t2 2d n_trans=4 execute_adjoint vs cpu t1 isign=+1",
          max_vs(adj, cpu_ref(N, x, cs)), CPU_THR)
    fk = gpu.nufft2d1(*x, cs, N, eps=EPS, isign=+1)
    fk_loop = np.stack([gpu.nufft2d1(*x, cs[b], N, eps=EPS, isign=+1)
                        for b in range(4)])
    check("nufft2d1 stacked c (4, M) vs loop", max_vs(fk, fk_loop), LOOP_THR)

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(0 if not FAILS else 1)
