"""Is the machine quiet enough for the paper's timing sweeps?

Short transforms are launch-bound, so host and display load inflate them
while leaving long ones almost untouched. Reports the per-op launch latency
(a tiny op, eval and synchronize, median of 500) and the profiler's
calibration workload, and the GPU utilization sampled over two seconds.
On an idle M5 Max the launch latency is about 60 us and the stage floor
0.167 ms (September 2026 profile); a session of the desktop app rendering a
long transcript raised it to 160-190 us and the floor to 0.21 ms.

    .venv/bin/python harness/idle_check.py
"""

import pathlib
import re
import statistics
import subprocess
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def launch_latency_us(n=500):
    a = mx.ones((1024,))
    mx.eval(a)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        a = a + 1.0
        mx.eval(a)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e6 * statistics.median(ts)


def gpu_utilization(seconds=2.0, step=0.25):
    vals = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                             capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            vals.append(int(m.group(1)))
        time.sleep(step)
    return vals


def main():
    lat = launch_latency_us()
    util = gpu_utilization()
    import profile_stages as ps
    calib = 1e3 * statistics.median(ps.time_calib(7, 2, 100))
    print(f"launch latency {lat:.0f} us per op (eval + synchronize, median of 500)")
    print(f"calibration {calib:.2f} ms (median of 7)")
    print(f"GPU utilization while idle: {util} percent")
    quiet = lat <= 90 and (not util or max(util) <= 10)
    print("QUIET" if quiet else "BUSY: close other apps and let the desktop idle before timing")
    return 0 if quiet else 1


if __name__ == "__main__":
    sys.exit(main())
