# Acceptance / benchmarks

Accuracy, speed, and memory results for v0.1 — measured on two Apple-silicon
machines (an M1 and an M5 Max) — are reported in the
[technical report](mlx-nufft.pdf). These measurements predate the v0.2 changes
to plan setup and execution. See [report status](REPORT.md) and the
[v0.2.0 changelog](CHANGELOG.md#v020---2026-07-02) for context and reported
performance improvements.

To generate the acceptance matrix for your checkout and hardware:

```bash
.venv/bin/python harness/run_acceptance.py          # full matrix
.venv/bin/python harness/run_acceptance.py --quick  # small subset
```

Results are written to `results/acceptance.json`.
