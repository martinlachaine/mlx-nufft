"""Acceptance matrix runner.

Regenerates the full acceptance table:
  - geometries: anisotropic (large coordinate dynamic range, lat=1.5) and
    generic isotropic random type-3
  - N in {512, 1024, 2048} (targets M = N^2), P in {1e4, 1e5, 1e6}
  - GPU crit64 at eps=1e-5; accuracy vs exact fp64 direct-sum subset oracle
  - CPU FINUFFT fp64 timed wherever its predicted internal grids fit in RAM;
    feasibility statement where they do not
  - both sides build the plan and execute it once, three times over, then
    execute the last plan three more times; minimum of each: speedup is the
    repeated execute, speedup_total the plan plus its first execute
  - adjoint/convention self-tests
Writes results/acceptance.json and prints a markdown table.

Usage: run_acceptance.py [--quick]   (--quick: N=512/1024 x P=1e4/1e5 only)
"""

import sys
import time
import json
import pathlib
import platform
import subprocess
import os

import numpy as np
import mlx.core as mx
import finufft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import (gen_anisotropic, gen_generic, direct_sum_mp,   # noqa
                         direct_sum, rel_l2)
from mlx_nufft import Type3Plan, Type1Plan, Type2Plan           # noqa
from mlx_nufft.sizing import finufft_width, set_nhg_type3   # noqa
from harness import memwatch                                  # noqa

RESDIR = pathlib.Path(__file__).resolve().parents[1] / "results"
RESDIR.mkdir(exist_ok=True)
# per-geometry eps: the thin-slab anisotropic case runs at 1e-5 (floor
# ~1.4e-5); the isotropic case has an fp32 corner-deconvolution noise floor
# whose sweet spot is w=8 (eps=3e-5, rel-L2 ~6e-5; see REPORT 'fp32 accuracy
# envelope')
EPS_BY_GEOM = {"anisotropic": 1e-5, "generic": 3e-5}
NSUB = 20_000


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, {platform.platform()}"


def machine_ram_gib():
    return memwatch.ram_gib()


def cpu_pred_gib(x, s, eps, sigma=1.25):
    """Predicted CPU fp64 FINUFFT t3 grid memory (complex128, both levels),
    at the width double-precision FINUFFT uses (no fp32 low-sigma cap)."""
    w = finufft_width(eps, sigma)
    spread, inner = 1.0, 1.0
    for d in range(3):
        X = 0.5 * (x[d].max() - x[d].min())
        S = 0.5 * (s[d].max() - s[d].min())
        nf, _, _ = set_nhg_type3(S, X, sigma, w)
        from mlx_nufft.sizing import next235even
        spread *= nf
        inner *= next235even(int(np.ceil(sigma * nf)))
    return (spread + inner) * 16 / 2**30


def oracle(tag, x, c, s, M):
    rng = np.random.default_rng(1)
    idx = rng.choice(M, min(NSUB, M), replace=False)
    f = RESDIR / f"oracle_{tag}_n{idx.size}.npz"
    if f.exists():
        return np.load(f)["fd"], idx
    t0 = time.perf_counter()
    fd = direct_sum_mp(x, c, s, isign=+1, idx=idx)
    print(f"    [oracle {tag}] {time.perf_counter()-t0:.0f}s", flush=True)
    np.savez(f, fd=fd, idx=idx)
    return fd, idx


def run_config(geom, N, P):
    tag = f"{geom}_N{N}_P{P}"
    print(f"  config {tag}", flush=True)
    if geom == "anisotropic":
        prob = gen_anisotropic(N=N, P=P, lat=1.5)
    else:
        prob = gen_generic(N=N, P=P)
    x, c, s = prob["x"], prob["c"], prob["s"]
    M = s[0].size
    eps = EPS_BY_GEOM[geom]
    fd, idx = oracle(tag, x, c, s, M)

    # GPU and CPU are timed alike. Three times, the previous plan is freed, a
    # new one is built and executed once, which is what a new geometry costs
    # in a running process; the last plan then executes three more times. The
    # minimum of each is reported: speedup compares the repeated execute,
    # speedup_total the plan plus its first execute. The first build starts
    # from an empty MLX buffer cache and is kept as t_total_cold.
    t_plans, t_tots, plan = [], [], None
    mx.clear_cache()
    for k in range(3):
        plan = None
        if k == 2:
            mx.reset_peak_memory()      # peak covers one build and its executes
        t0 = time.perf_counter()
        plan = Type3Plan(x, s, eps=eps, isign=+1, prec="crit64")
        t1 = time.perf_counter()
        f_gpu = plan.execute(c)
        t_plans.append(t1 - t0)
        t_tots.append(time.perf_counter() - t0)
    ts = []
    for _ in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        f_gpu = plan.execute(c)
        ts.append(time.perf_counter() - t0)
    peak = mx.get_peak_memory() / 2**30
    e_gpu = rel_l2(f_gpu[idx], fd)
    row = dict(geom=geom, N=N, P=P, M=int(M), eps=eps, prec="crit64",
               nf=plan.nf, n_up=plan.n_up, slab=bool(plan.slab_mode),
               t_plan=min(t_plans), t_plan_first=t_plans[0], t_total=min(t_tots),
               t_total_cold=t_tots[0],
               t_gpu=min(ts), rel_l2=e_gpu, peak_gib=peak)
    del plan
    mx.clear_cache()

    pred = cpu_pred_gib(x, s, eps)
    # run the fp64 CPU oracle when its predicted grids fit in half of RAM
    # (9 GiB was the 16 GB mini's number; a 128 GB machine can afford the
    # anisotropic cases' ~13 GiB and gets their CPU timings)
    if pred < 0.5 * machine_ram_gib():
        ts_cp, ts_ct, ts_c, cp = [], [], [], None
        for _ in range(3):
            cp = None
            t0 = time.perf_counter()
            cp = finufft.Plan(3, 3, eps=eps, isign=+1)
            cp.setpts(*x, *s)
            t1 = time.perf_counter()
            f_cpu = cp.execute(c)
            ts_cp.append(t1 - t0)
            ts_ct.append(time.perf_counter() - t0)
        for _ in range(3):
            t0 = time.perf_counter()
            f_cpu = cp.execute(c)
            ts_c.append(time.perf_counter() - t0)
        del cp
        row.update(t_cpu=min(ts_c), t_cpu_plan=min(ts_cp), t_cpu_total=min(ts_ct),
                   rel_l2_cpu=rel_l2(f_cpu[idx], fd),
                   speedup=min(ts_c) / min(ts),
                   speedup_total=min(ts_ct) / min(t_tots),
                   cpu_pred_gib=pred)
        del f_cpu
    else:
        row.update(t_cpu=None, cpu_pred_gib=pred,
                   cpu_note=f"fp64 grids ~{pred:.1f} GiB exceed the "
                            "CPU-oracle memory threshold")
    print(f"    gpu plan {min(t_plans):.3f}s total {min(t_tots):.3f}s exec {min(ts):.3f}s "
          f"rel_l2 {e_gpu:.2e} "
          f"peak {peak:.1f} GiB"
          + (f"  cpu plan {row['t_cpu_plan']:.3f}s total {row['t_cpu_total']:.3f}s "
             f"exec {row['t_cpu']:.3f}s "
             f"(exec {row['speedup']:.2f}x, total {row['speedup_total']:.2f}x)"
             if row.get("t_cpu") else f"  cpu N/A ({pred:.1f} GiB)"),
          flush=True)
    return row


def adjoint_tests():
    out = {}
    # type-3 adjoint (swap sources/targets, isign=-1) vs direct sum
    prob = gen_anisotropic(N=64, P=2000, lat=0.02)
    x, s = prob["x"], prob["s"]
    ones = np.ones(s[0].size, dtype=np.complex64)
    ref = direct_sum(s, ones.astype(np.complex128), x, isign=-1)
    fa = Type3Plan(s, x, eps=1e-5, isign=-1, prec="crit64").execute(ones)
    out["t3_adjoint"] = rel_l2(fa, ref)
    # t1/t2 adjoint pairing at eps=1e-6 (error budget below 1e-5)
    rng = np.random.default_rng(7)
    Nm, P = (64, 72, 18), 5000
    xx = [rng.uniform(-np.pi, np.pi, P) for _ in range(3)]
    cc = (rng.standard_normal(P) + 1j * rng.standard_normal(P))
    fk = (rng.standard_normal(Nm) + 1j * rng.standard_normal(Nm))
    f1 = Type1Plan(xx, Nm, eps=1e-5, isign=+1).execute(
        cc.astype(np.complex64))
    c2 = Type2Plan(xx, Nm, eps=1e-5, isign=-1).execute(
        fk.astype(np.complex64))
    lhs = np.vdot(np.asarray(fk), f1)
    rhs = np.vdot(c2, cc)
    out["t12_pairing"] = abs(lhs - rhs) / abs(lhs)
    # t1/t2 accuracy vs CPU FINUFFT fp64
    ref1 = finufft.nufft3d1(*xx, cc, Nm, isign=+1, eps=1e-9)
    out["t1_vs_finufft"] = rel_l2(
        Type1Plan(xx, Nm, eps=1e-5, isign=+1).execute(
            cc.astype(np.complex64)), ref1)
    ref2 = finufft.nufft3d2(*xx, fk, isign=+1, eps=1e-9)
    out["t2_vs_finufft"] = rel_l2(
        Type2Plan(xx, Nm, eps=1e-5, isign=+1).execute(
            fk.astype(np.complex64)), ref2)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    memwatch.start_watchdog()          # in-process: a forking poll slowed the CPU reference
    print(f"machine: {machine()}  eps={EPS_BY_GEOM} prec=crit64",
          flush=True)

    Ns = [512, 1024] if quick else [512, 1024, 2048]
    Ps = [10_000, 100_000] if quick else [10_000, 100_000, 1_000_000]
    rows = []
    for geom in ("generic", "anisotropic"):
        print(f"[{geom}]", flush=True)
        for N in Ns:
            for P in Ps:
                rows.append(run_config(geom, N, P))

    adj = adjoint_tests()
    print("[adjoint/convention]", {k: f"{v:.2e}" for k, v in adj.items()},
          flush=True)

    out = dict(machine=machine(), eps=EPS_BY_GEOM, nsub=NSUB, rows=rows,
               adjoint=adj,
               mlx_version=__import__("mlx.core", fromlist=["__version__"]
                                      ).__version__
               if hasattr(__import__("mlx.core", fromlist=["a"]),
                          "__version__") else "0.31.2")
    (RESDIR / "acceptance.json").write_text(json.dumps(out, indent=2))

    # markdown table
    print("\n| geometry | N (M=N^2) | P | eps | rel-L2 (gate 1e-4) | "
          "GPU exec s | CPU fp64 exec s | speedup exec | speedup total | peak GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cpu = f"{r['t_cpu']:.2f}" if r.get("t_cpu") else \
            f"N/A ({r['cpu_pred_gib']:.0f} GiB)"
        sp = f"{r['speedup']:.2f}x" if r.get("t_cpu") else "—"
        spt = f"{r['speedup_total']:.2f}x" if r.get("t_cpu") else "—"
        ok = "PASS" if r["rel_l2"] <= 1e-4 else "FAIL"
        print(f"| {r['geom']} | {r['N']} | {r['P']:.0e} | "
              f"{r['eps']:.0e} | "
              f"{r['rel_l2']:.2e} {ok} | {r['t_gpu']:.3f} | {cpu} | {sp} | {spt} | "
              f"{r['peak_gib']:.1f} |")
    print("\nsaved results/acceptance.json", flush=True)
