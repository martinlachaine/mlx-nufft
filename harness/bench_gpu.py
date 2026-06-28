"""GPU t3 benchmark: accuracy vs oracles + warm timing.

Usage: bench_gpu.py [lat] [N] [P] [eps] [prec] [reps] [--cpu] [--nsub n]
"""

import sys
import time
import json
import pathlib
import platform
import subprocess

import os
import threading
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum, rel_l2   # noqa: E402
from mlx_nufft.gpu_t3 import GpuT3Plan                  # noqa: E402


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, {platform.platform()}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lat = float(args[0]) if len(args) > 0 else 0.5
    N = int(args[1]) if len(args) > 1 else 1024
    P = int(args[2]) if len(args) > 2 else 100_000
    eps = float(args[3]) if len(args) > 3 else 1e-5
    prec = args[4] if len(args) > 4 else "crit64"
    reps = int(args[5]) if len(args) > 5 else 5
    do_cpu = "--cpu" in sys.argv
    nsub = 20_000
    if "--nsub" in sys.argv:
        nsub = int(sys.argv[sys.argv.index("--nsub") + 1])
    sigma_inner = None
    if "--sin" in sys.argv:
        sigma_inner = float(sys.argv[sys.argv.index("--sin") + 1])

    print(f"machine: {machine()}", flush=True)

    def watchdog(limit_gib=None):
        import subprocess as sp
        import time as _t
        if limit_gib is None:   # ~72% of machine RAM
            ram = int(sp.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) / 2**30
            limit_gib = max(11.5, 0.72 * ram)
        pid = os.getpid()
        while True:
            rss = int(sp.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True).stdout or 0)
            gib = max(rss * 1024, mx.get_active_memory()) / 2**30
            if gib > limit_gib:
                print(f"WATCHDOG: {gib:.1f} GiB > {limit_gib} GiB"
                      " — aborting to protect machine", flush=True)
                os._exit(3)
            _t.sleep(0.5)

    threading.Thread(target=watchdog, daemon=True).start()
    prob = gen_anisotropic(N=N, P=P, lat=lat)
    x, c, s = prob["x"], prob["c"], prob["s"]
    M = s[0].size
    print(f"lat={lat} N={N} (M={M}) P={P} eps={eps:.0e} prec={prec}")

    resdir = pathlib.Path(__file__).resolve().parents[1] / "results"
    resdir.mkdir(exist_ok=True)
    okey = resdir / f"oracle_lat{lat}_N{N}_P{P}_n{nsub}.npz"
    rng = np.random.default_rng(1)
    idx = rng.choice(M, min(nsub, M), replace=False)
    if okey.exists():
        fd = np.load(okey)["fd"]
        print(f"[oracle] loaded cached subset oracle {okey.name}", flush=True)
    else:
        t0 = time.perf_counter()
        fd = direct_sum(x, c, s, isign=+1, idx=idx, chunk=128)
        print(f"[oracle] direct subset n={idx.size}: "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        np.savez(okey, fd=fd, idx=idx)

    t0 = time.perf_counter()
    plan = GpuT3Plan(x, s, eps=eps, isign=+1, prec=prec,
                     sigma_inner=sigma_inner)
    t_plan = time.perf_counter() - t0
    gb = lambda nfs: np.prod([float(v) for v in nfs]) * 8 / 1e9  # noqa: E731
    print(f"plan: {t_plan:.2f}s  nf={plan.nf} n_up={plan.n_up} "
          f"w={plan.w}/{plan.w2}  "
          f"grids c64: {gb(plan.nf):.2f} + {gb(plan.n_up):.2f} GB",
          flush=True)

    # warmup (compiles kernels, allocates pools)
    f_gpu = plan.execute(c)
    # timed
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        f_gpu = plan.execute(c)
        ts.append(time.perf_counter() - t0)
    t_gpu_min, t_gpu_avg = min(ts), float(np.mean(ts))
    peak = mx.get_peak_memory() / 2**30
    print(f"[gpu {prec}] exec min {t_gpu_min:.3f}s avg {t_gpu_avg:.3f}s "
          f"(reps={reps})  mx peak mem {peak:.2f} GiB", flush=True)

    # accuracy: subset direct-sum oracle (exact fp64)
    mx.clear_cache()
    e_gpu = rel_l2(f_gpu[idx], fd)
    print(f"[acc] rel_l2(gpu vs direct-subset n={idx.size}) = {e_gpu:.3e}",
          flush=True)

    out = dict(machine=machine(), lat=lat, N=N, M=int(M), P=P, eps=eps,
               prec=prec, w=plan.w, nf=plan.nf, n_up=plan.n_up,
               t_plan=t_plan, t_gpu_min=t_gpu_min, t_gpu_avg=t_gpu_avg,
               ts=ts, peak_gib=peak, rel_l2_subset=e_gpu, nsub=int(idx.size))

    if do_cpu:
        import finufft
        for cpu_eps in sorted({eps, 1e-4}):
            ts_c = []
            f_cpu = None
            for _ in range(max(2, reps // 2)):
                t0 = time.perf_counter()
                f_cpu = finufft.nufft3d3(*x, c, *s, isign=+1, eps=cpu_eps)
                ts_c.append(time.perf_counter() - t0)
            e_cpu = rel_l2(f_cpu[idx], fd)
            print(f"[cpu fp64 eps={cpu_eps:.0e}] min {min(ts_c):.3f}s "
                  f"avg {np.mean(ts_c):.3f}s  "
                  f"rel_l2(cpu vs direct-subset) = {e_cpu:.3e}")
            print(f"[speedup vs eps={cpu_eps:.0e}] "
                  f"cpu_min/gpu_min = {min(ts_c)/t_gpu_min:.2f}x  "
                  f"cpu_avg/gpu_avg = {np.mean(ts_c)/t_gpu_avg:.2f}x")
            out.setdefault("cpu", {})[f"{cpu_eps:.0e}"] = dict(
                t_min=min(ts_c), t_avg=float(np.mean(ts_c)),
                rel_l2_subset=e_cpu)

    tag = f"lat{lat}_N{N}_P{P}_eps{eps:.0e}_{prec}"
    (resdir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"saved results/{tag}.json")
