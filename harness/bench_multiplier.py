"""Accuracy/throughput benchmark, mirroring the cuFINUFFT paper (Fig 4-6).

The signature figure: "ns per nonuniform point" vs achieved relative-L2 error,
for mlx-nufft and same-machine CPU FINUFFT, over an eps sweep, types 1/2, dims
1/2/3, "rand" and "cluster" distributions. Also a density (rho) sweep (Fig-6
analogue). Achieved error is measured against a tight fp64 CPU reference
(eps=1e-12), as cuFINUFFT measures error against FINUFFT at tight tolerance.

Protocol: "exec" timing only (plan + setpts excluded on both arms); single
precision both arms; GPU data device-resident (unified memory, no transfer).
GPU = min of 3 warm executes, CPU = min of 2.

Usage: bench_multiplier.py [--quick] [--rho]
Writes results/multiplier.json.
"""

import sys
import json
import time
import pathlib
import platform
import subprocess

import numpy as np
import mlx.core as mx
import finufft as cpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
import mlx_nufft as gpu                                          # noqa: E402
from harness.gen import rel_l2, uniform_points                # noqa: E402

REF_EPS = 1e-12   # fp64 CPU reference tolerance ("ground truth")


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, {platform.platform()}", mem


def build_problem(dim, ntype, N, dist, rho, ram_gb, seed=11):
    """Build (x, data, fp64 ref) once; the ref is reused across the eps sweep."""
    Nt = tuple([N] * dim)
    x, _, M = uniform_points(dim, N, rho=rho, dist=dist, seed=seed)
    n_tot = int(np.prod(Nt))
    est_gib = (3 * (2.0 ** dim) * n_tot * 8 + M * dim * 20) / 2**30
    if est_gib > 0.5 * ram_gb:
        return {"skip": f"~{est_gib:.0f} GiB > 50% RAM"}
    rng = np.random.default_rng(seed + 1)
    c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)
    fk = (rng.standard_normal(Nt) + 1j * rng.standard_normal(Nt)).astype(np.complex64)
    data = c if ntype == 1 else fk
    isign = +1 if ntype == 1 else -1
    rp = cpu.Plan(ntype, Nt, eps=REF_EPS, isign=isign, dtype="complex128")
    rp.setpts(*x)
    ref = np.asarray(rp.execute(data.astype(np.complex128))).ravel()
    return dict(Nt=Nt, x=x, M=M, data=data, isign=isign, ref=ref)


def bench(dim, ntype, N, eps, dist, rho, ram_gb, prob, seed=11):
    Nt, x, M = prob["Nt"], prob["x"], prob["M"]
    data, isign, ref = prob["data"], prob["isign"], prob["ref"]

    # GPU arm (exec only; device-resident; min of 3)
    data_g = mx.array(data); mx.eval(data_g)
    if ntype == 1:
        gp = gpu.Type1PlanND(tuple(x), Nt, eps=eps, isign=isign)
    else:
        gp = gpu.Type2PlanND(tuple(x), Nt, eps=eps, isign=isign)
    run = lambda: gp.execute(data_g, return_np=False)        # noqa: E731
    out = run()
    ts = []
    for _ in range(3):
        mx.synchronize(); t0 = time.perf_counter(); out = run(); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    t_gpu = min(ts)
    out_g = np.asarray(out)
    if ntype == 2 and gp.sorted and not gp._od:
        inv = np.empty_like(gp.perm); inv[gp.perm] = np.arange(gp.P)
        out_g = out_g[inv]
    err_gpu = rel_l2(out_g.ravel(), ref)
    del gp, data_g, out; mx.clear_cache()

    # CPU fp32 arm (exec only; min of 2)
    cp = cpu.Plan(ntype, Nt, eps=eps, isign=isign, dtype="complex64")
    cp.setpts(*[v.astype(np.float32) for v in x])
    ts_c = []
    for _ in range(2):
        t0 = time.perf_counter(); out_c = cp.execute(data); ts_c.append(time.perf_counter() - t0)
    t_cpu = min(ts_c)
    err_cpu = rel_l2(np.asarray(out_c).ravel(), ref)

    return dict(dim=dim, type=ntype, N=N, M=M, eps=eps, dist=dist, rho=rho,
                t_gpu=t_gpu, t_cpu=t_cpu, mult=t_cpu / t_gpu,
                ns_gpu=t_gpu / M * 1e9, ns_cpu=t_cpu / M * 1e9,
                rel_l2_gpu=float(err_gpu), rel_l2_cpu=float(err_cpu))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    do_rho = "--rho" in sys.argv
    mstr, ram_gb = machine()
    print(f"machine: {mstr}")
    print(f"cpu finufft {cpu.__version__} c64 / gpu mlx-nufft {gpu.__version__} "
          f"crit64; ref fp64 eps={REF_EPS:g}", flush=True)

    # representative sizes per dim (M = rho * (2N)^dim at sigma=2)
    Nrep = {1: 2 ** 20, 2: 1024, 3: 96}
    epss = [1e-2, 1e-4, 1e-5, 1e-6] if quick else \
           [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    dims = [2, 3] if quick else [1, 2, 3]
    dists = ["rand"] if quick else ["rand", "cluster"]

    rows = []
    # E1: signature accuracy/throughput curves (ref built once per problem)
    for dim in dims:
        for ntype in (1, 2):
            for dist in dists:
                prob = build_problem(dim, ntype, Nrep[dim], dist, 1.0, ram_gb)
                if "skip" in prob:
                    print(f"SKIP {dim}d t{ntype} {dist}: {prob['skip']}", flush=True)
                    continue
                for eps in epss:
                    r = bench(dim, ntype, Nrep[dim], eps, dist, 1.0, ram_gb, prob)
                    print(f"{dim}d t{ntype} {dist} N={Nrep[dim]} eps={eps:g}: "
                          f"ns/pt gpu {r['ns_gpu']:.1f} cpu {r['ns_cpu']:.1f}"
                          f" -> {r['mult']:.1f}x | relL2 gpu {r['rel_l2_gpu']:.1e}"
                          f" cpu {r['rel_l2_cpu']:.1e}", flush=True)
                    rows.append(r)

    # E7: density sweep (Fig-6 analogue), one representative config
    if do_rho and not quick:
        for rho in (0.1, 1.0, 10.0):
            for dim in (2, 3):
                prob = build_problem(dim, 1, Nrep[dim], "rand", rho, ram_gb)
                if "skip" in prob:
                    print(f"SKIP rho={rho} {dim}d: {prob['skip']}", flush=True); continue
                r = bench(dim, 1, Nrep[dim], 1e-5, "rand", rho, ram_gb, prob)
                r["sweep"] = "rho"
                print(f"rho={rho} {dim}d t1: ns/pt gpu {r['ns_gpu']:.1f} "
                      f"cpu {r['ns_cpu']:.1f} -> {r['mult']:.1f}x", flush=True)
                rows.append(r)

    out = dict(machine=mstr, protocol="exec-only, fp32 both arms, "
               f"ref fp64 eps={REF_EPS:g}", ref_eps=REF_EPS, rows=rows)
    p = pathlib.Path(__file__).resolve().parents[1] / "results" / "multiplier.json"
    p.parent.mkdir(exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nsaved {p}  ({len(rows)} rows)", flush=True)
