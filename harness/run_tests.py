"""Run the full correctness test suite and report a pass/fail summary.

Usage (from the repo root, with the dev environment installed):

    .venv/bin/python harness/run_tests.py

Each test under harness/test_*.py is a standalone script that exits 0 on
success and non-zero on failure (the VkFFT backend test exits 0 with a SKIP
when the optional bridge is not built). This runner executes them all with the
current interpreter, prints each result, and exits non-zero if any failed —
so it is a single "does mlx-nufft work on my machine?" command, and is also
suitable for CI.

Requires the dev extras (finufft, scipy) used as references:

    uv pip install -p .venv/bin/python -e ".[dev]"
"""

import pathlib
import subprocess
import sys
import time

HARNESS = pathlib.Path(__file__).resolve().parent
TESTS = sorted(HARNESS.glob("test_*.py"))


def main() -> int:
    if not TESTS:
        print("no test_*.py found in", HARNESS)
        return 1

    print(f"Running {len(TESTS)} test scripts with {sys.executable}\n")
    results = []
    for path in TESTS:
        name = path.name
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
        )
        dt = time.perf_counter() - t0
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        skipped = "SKIP" in proc.stdout
        if proc.returncode == 0:
            status = "SKIP" if skipped else "PASS"
        else:
            status = "FAIL"
        results.append((name, status, dt))
        print(f"  {status:4}  {name:<34} {dt:6.1f}s   {tail[:70]}")
        if status == "FAIL":
            # surface why it failed
            err = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
            for line in err[-8:]:
                print(f"          | {line}")

    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
