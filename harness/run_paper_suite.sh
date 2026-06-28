#!/usr/bin/env bash
# Run the full paper benchmark suite and stage artifacts under paper/data/<label>.
# Usage: bash harness/run_paper_suite.sh m1     (on the Mac mini)
#        bash harness/run_paper_suite.sh m5     (on the MacBook Pro M5 Max)
# Requires the project venv (.venv) with mlx-nufft installed (pip install -e ".[dev]").
set -u
LABEL="${1:?usage: run_paper_suite.sh <m1|m5>}"
cd "$(dirname "$0")/.." || exit 1
OUT="paper/data/$LABEL"
mkdir -p "$OUT"
PY=.venv/bin/python

echo "[suite] $(date) start ($LABEL)"
echo "[suite] E1/E7 bench_multiplier"
$PY harness/bench_multiplier.py --rho 2>&1 | tee "$OUT/multiplier.log"
cp -f results/multiplier.json "$OUT/multiplier.json" 2>/dev/null || true
echo "[suite] E4 diagnose_ref"
$PY harness/diagnose_ref.py 2>&1 | tee "$OUT/diagnose.log"
echo "[suite] E5 test_types12"
$PY harness/test_types12.py 2>&1 | tee "$OUT/types12.log"
echo "[suite] E5 test_gpu_small"
$PY harness/test_gpu_small.py 2>&1 | tee "$OUT/gpu_small.log"
echo "[suite] E5/E3 run_acceptance"
$PY harness/run_acceptance.py 2>&1 | tee "$OUT/acceptance.log"
cp -f results/acceptance.json "$OUT/acceptance.json" 2>/dev/null || true
echo "[suite] E6 bench_batched (plan-reuse / multi-vector batching levers)"
$PY harness/bench_batched.py 2>&1 | tee "$OUT/batched.log"
echo "[suite] E6 bench_frames (two-level plan / set_sources re-point)"
$PY harness/bench_frames.py 2>&1 | tee "$OUT/frames.log"
echo "[suite] $(date) DONE -> $OUT"
