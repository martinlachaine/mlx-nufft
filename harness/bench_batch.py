"""Multi-vector batching bench: N sequential executes vs one execute_batch.

Compares running multiple strength vectors over one fixed geometry as
separate transforms vs a single batched transform. On a 16 GB machine this
measured NEGATIVE (0.93x, see ACCEPTANCE.md): all stages are atomic- or
bandwidth-bound, so 3x the data costs ~3x the time, and full-size 3-vector
grids (~25 GB) do not fit 16 GB. Worth re-running on a large-memory machine,
where the headroom admits the full size and stage times shrink (so per-launch
overhead is a larger fraction) — the conditions under which batching could
flip positive.

Usage: bench_batch.py [lat] [N] [P]
  16 GB-safe comparison:        bench_batch.py 0.75 1024 100000
  full size (large-RAM only):   bench_batch.py 1.5 1024 100000
Refuses configurations whose predicted batched grids exceed ~60% of RAM.
"""

import sys
import time
import pathlib
import platform
import subprocess

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent))
import mlx_nufft.gpu_t3 as g                      # noqa: E402
from harness.gen import gen_anisotropic, rel_l2           # noqa: E402

NCH = 3


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, {platform.platform()}", mem


if __name__ == "__main__":
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
    mstr, ram_gb = machine()
    print(f"machine: {mstr}")

    prob = gen_anisotropic(N=N, P=P, lat=lat)
    x, c, s = prob["x"], prob["c"], prob["s"]
    rng = np.random.default_rng(3)
    cs = np.stack([c, c * (0.5 + 0.5j),
                   rng.standard_normal(c.size)
                   + 1j * rng.standard_normal(c.size)])

    # force the slab path for BOTH arms so the comparison is apples-to-apples
    g._SLAB_THRESHOLD = min(g._SLAB_THRESHOLD, 1.0e9)
    plan = g.GpuT3Plan(x, s, eps=1e-5, isign=+1, prec="crit64")
    grids_gib = NCH * (np.prod([float(v) for v in plan.nf])
                       + np.prod([float(v) for v in plan.n_up])) * 8 / 2**30
    print(f"nf={plan.nf} n_up={plan.n_up} slab={plan.slab_mode}  "
          f"batched grids ~{grids_gib:.1f} GiB (RAM {ram_gb} GB)")
    if grids_gib > 0.6 * ram_gb:
        print("REFUSING: batched grids exceed ~60% of RAM on this machine")
        sys.exit(2)

    fsep = [plan.execute(cs[ch]) for ch in range(NCH)]   # warm
    fb = plan.execute_batch(cs)
    mx.reset_peak_memory()
    ts_seq = []
    for _ in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        fsep = [plan.execute(cs[ch]) for ch in range(NCH)]
        ts_seq.append(time.perf_counter() - t0)
    pk_seq = mx.get_peak_memory() / 2**30
    mx.reset_peak_memory()
    ts_b = []
    for _ in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        fb = plan.execute_batch(cs)
        ts_b.append(time.perf_counter() - t0)
    pk_b = mx.get_peak_memory() / 2**30

    print(f"3 sequential: {min(ts_seq):.2f}s (peak {pk_seq:.1f} GiB)   "
          f"batched({NCH}): {min(ts_b):.2f}s (peak {pk_b:.1f} GiB)   "
          f"speedup {min(ts_seq) / min(ts_b):.2f}x")
    for ch in range(NCH):
        print(f"  ch{ch} batched-vs-separate rel_l2: "
              f"{rel_l2(fb[ch], fsep[ch]):.2e} "
              "(run-to-run atomic noise floor ~6e-6)")
