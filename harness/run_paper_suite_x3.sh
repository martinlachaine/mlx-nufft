#!/usr/bin/env bash
# The paper's timing suite, three repetitions, with the idle check before
# and after: bench_multiplier (the speed-versus-accuracy sweep) and
# run_acceptance (the type-3 table) three times each, then the per-stage
# profile (with its calibration bracket) and the one-off diagnostics.
# Usage: bash harness/run_paper_suite_x3.sh <label>      e.g. m1_v0.3.3
# Writes paper/data/<label>_run{1,2,3}/ and paper/data/<label>_suite.log.
# Run it from a terminal on an idle, plugged-in machine with other apps
# closed; the idle check prints BUSY when the launch latency or GPU
# utilization says otherwise, and the sweep then understates the short
# transforms. Combine the runs with the paper's combine_runs.py.
set -u
LABEL="${1:?usage: run_paper_suite_x3.sh <label>}"
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
D="paper/data"
mkdir -p "$D"
LOG="$D/${LABEL}_suite.log"
{
  echo "commit $(git rev-parse --short HEAD) $(date)"
  $PY -c "import mlx_nufft, mlx.core as mx; print('mlx_nufft', mlx_nufft.__version__, 'mlx', mx.__version__)"
  echo "== idle check before"; $PY harness/idle_check.py
  for i in 1 2 3; do
    R="$D/${LABEL}_run$i"; mkdir -p "$R"
    echo "== run $i multiplier $(date +%H:%M)"
    $PY harness/bench_multiplier.py --rho > "$R/multiplier.log" 2>&1 && cp results/multiplier.json "$R/" || echo "multiplier run $i FAILED"
    echo "== run $i acceptance $(date +%H:%M)"
    $PY harness/run_acceptance.py > "$R/acceptance.log" 2>&1 && cp results/acceptance.json "$R/" || echo "acceptance run $i FAILED"
  done
  echo "== idle check after $(date +%H:%M)"; $PY harness/idle_check.py
  R="$D/${LABEL}_run1"; mkdir -p "$R/profile"
  echo "== profile $(date +%H:%M)"
  $PY harness/profile_stages.py --tag "$LABEL" 2>&1 | tail -2
  cp "results/$LABEL.json" "results/$LABEL.md" "$R/profile/" 2>/dev/null
  echo "== diagnostics"
  $PY harness/diagnose_ref.py > "$R/diagnose.log" 2>&1
  $PY harness/test_types12.py > "$R/types12.log" 2>&1
  $PY harness/test_gpu_small.py > "$R/gpu_small.log" 2>&1
  $PY harness/bench_batched.py > "$R/batched.log" 2>&1
  $PY harness/bench_frames.py > "$R/frames.log" 2>&1
  echo "SUITE DONE $(date +%H:%M)"
} > "$LOG" 2>&1
tail -3 "$LOG"
