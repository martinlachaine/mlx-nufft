# Acceptance and benchmarks

The technical report's accuracy, speed, and memory results are v0.1
measurements on an M1 and an M5 Max; see [REPORT.md](REPORT.md). Current
per-stage timings against the committed baselines come from
`harness/profile_stages.py` (tables in `harness/PROFILE_*.md`).

To generate the acceptance matrix for your checkout and hardware:

```bash
.venv/bin/python harness/run_acceptance.py          # full matrix
.venv/bin/python harness/run_acceptance.py --quick  # small subset
```

Results are written to `results/acceptance.json`.
