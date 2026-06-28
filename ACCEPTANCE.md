# Acceptance / benchmarks

Accuracy, speed, and memory results — measured on two Apple-silicon machines
(an M1 and an M5 Max) — are reported in the technical report included in this
repository (`mlx-nufft.pdf`).

To regenerate the acceptance matrix locally:

```bash
.venv/bin/python harness/run_acceptance.py          # full matrix
.venv/bin/python harness/run_acceptance.py --quick  # small subset
```

Results are written to `results/acceptance.json`.
