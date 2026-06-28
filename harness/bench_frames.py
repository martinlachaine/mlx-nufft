"""Bench: two-level plan for repeated fixed-geometry transforms.

Scenario: the target set is fixed while the source coordinates and strengths
change on every call (a common pattern for streaming / repeated solves).
Compares, per call:
  baseline:  a fresh Type3Plan + execute every time
  two-level: plan built once (source_extent = the known coordinate bounds),
             then set_sources(host|gpu) + execute per call
Accuracy: an updated source set vs the exact direct-sum subset oracle (gate
1e-4).
"""

import sys
import time
import pathlib

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
from harness.gen import gen_anisotropic, direct_sum_mp, rel_l2   # noqa: E402
from mlx_nufft import Type3Plan                          # noqa: E402

if __name__ == "__main__":
    N, P, lat = 1024, 100_000, 1.5
    sets = [gen_anisotropic(N=N, P=P, lat=lat, seed=k) for k in range(4)]
    s = sets[0]["s"]

    # coordinate bounds the source axes can reach across all source sets
    # (closed form from gen_anisotropic's fixed scale constants)
    k0, tmin, tmax = 2 * np.pi / 520e-9, 1.0 / 6.0, 1.0 / 1.2
    b = k0 * tmax * lat
    ext = [(-b, b), (-b, b), (0.5 * k0 * tmin, 0.5 * k0 * tmax)]

    t0 = time.perf_counter()
    plan = Type3Plan(sets[0]["x"], s, eps=1e-5, isign=+1, prec="crit64",
                     source_extent=ext)
    t_plan0 = time.perf_counter() - t0
    plan.execute(sets[0]["c"])      # warm (kernels, pools)
    print(f"plan build (set 0): {t_plan0:.2f}s")

    # baseline: fresh plan per call
    t0 = time.perf_counter()
    pb = Type3Plan(sets[1]["x"], s, eps=1e-5, isign=+1, prec="crit64")
    t_replan = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_base = pb.execute(sets[1]["c"])
    t_exec_b = time.perf_counter() - t0
    del pb
    mx.clear_cache()
    print(f"baseline per call: replan {t_replan:.2f}s + exec {t_exec_b:.2f}s"
          f" = {t_replan + t_exec_b:.2f}s")

    # two-level: per-call set_sources + execute
    res = {}
    for backend in ("host", "gpu"):
        ts_set, ts_exec = [], []
        f_upd = None
        for st in sets[1:]:
            mx.synchronize()
            t0 = time.perf_counter()
            plan.set_sources(st["x"], backend=backend)
            mx.synchronize()
            ts_set.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            f = plan.execute(st["c"])
            ts_exec.append(time.perf_counter() - t0)
            if f_upd is None:
                f_upd = f
        res[backend] = (min(ts_set), min(ts_exec), f_upd)
        print(f"two-level [{backend}]: set_sources {min(ts_set)*1000:.0f} ms"
              f" + exec {min(ts_exec):.2f}s")

    # accuracy on the updated source set (set 1) vs exact oracle
    rng = np.random.default_rng(1)
    idx = rng.choice(s[0].size, 20_000, replace=False)
    ofile = pathlib.Path(__file__).resolve().parents[1] / "results" / \
        f"oracle_sets_N{N}_P{P}_seed1.npz"
    if ofile.exists():
        fd = np.load(ofile)["fd"]
    else:
        t0 = time.perf_counter()
        fd = direct_sum_mp(sets[1]["x"], sets[1]["c"], s, isign=+1, idx=idx)
        print(f"oracle: {time.perf_counter()-t0:.0f}s")
        np.savez(ofile, fd=fd, idx=idx)
    print(f"accuracy set-1 vs oracle: baseline {rel_l2(f_base[idx], fd):.3e}"
          f"  host {rel_l2(res['host'][2][idx], fd):.3e}"
          f"  gpu {rel_l2(res['gpu'][2][idx], fd):.3e}")
    print(f"update vs fresh-plan (same set): "
          f"host {rel_l2(res['host'][2], f_base):.3e} "
          f"gpu {rel_l2(res['gpu'][2], f_base):.3e} "
          f"(run-to-run atomic noise floor ~6e-6)")
