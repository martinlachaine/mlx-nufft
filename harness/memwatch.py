"""Memory watchdog for the benchmark scripts, without forking.

The scripts used to poll `ps` every 0.5 s. Each poll forks the benchmark
process, and with gigabytes mapped the fork slowed the multithreaded CPU
FINUFFT reference by about 50 percent (1.05 s instead of 0.70 s for a 640^3
type-3 execute on an M5 Max) while leaving GPU times unchanged. The resident
size is now read in-process through libproc's proc_pid_rusage.
"""

import ctypes
import os
import threading
import time

import mlx.core as mx


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(_RusageInfoV0)]
_libproc.proc_pid_rusage.restype = ctypes.c_int


def resident_bytes(pid=None):
    """Resident size of a process in bytes (what `ps -o rss` reports, x1024)."""
    info = _RusageInfoV0()
    if _libproc.proc_pid_rusage(os.getpid() if pid is None else pid, 0,
                                ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return int(info.ri_resident_size)


def ram_gib():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def start_watchdog(limit_gib=None, period=0.5):
    """Abort the process (exit code 3) when its resident size or MLX's active
    memory exceeds limit_gib, by default 72 percent of RAM and at least
    11.5 GiB (the 16 GB mini's number)."""
    if limit_gib is None:
        limit_gib = max(11.5, 0.72 * ram_gib())

    def run():
        while True:
            gib = max(resident_bytes(), mx.get_active_memory()) / 2**30
            if gib > limit_gib:
                print(f"WATCHDOG: {gib:.1f} GiB > {limit_gib:.1f} GiB, aborting "
                      "to protect the machine", flush=True)
                os._exit(3)
            time.sleep(period)

    threading.Thread(target=run, daemon=True).start()
    return limit_gib
