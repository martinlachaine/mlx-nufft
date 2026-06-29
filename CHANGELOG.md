# Changelog

## v0.1.3 — 2026-06-28

Infrastructure only — no library or numerical behavior changes. First release
archived on Zenodo, giving the project a citable DOI.

## v0.1.2 — 2026-06-28

Documentation and packaging only — no library or numerical behavior changes.

- Published to PyPI: `pip install mlx-nufft`.
- README install section updated to lead with PyPI, plus a PyPI version badge.
- First release published through the Trusted Publishing workflow on release.

## v0.1.1 — 2026-06-28

Documentation and infrastructure only — no library or numerical behavior
changes (the `mlx_nufft` package is identical to v0.1.0).

- Continuous integration: the correctness suite runs on Apple-silicon GitHub
  runners (`.github/workflows/ci.yml`); large type-3 cases that need a big-GPU
  Metal buffer skip cleanly on small/CI GPUs.
- `harness/run_tests.py`: one-command runner for the full correctness suite.
- `examples/quickstart.py`: runnable, dependency-light demo that self-checks
  accuracy against an exact direct DFT.
- README: copy-pasteable quickstart, "verify the install" section, status
  badges, and a "Development and validation" note.
- `CONTRIBUTING.md` and GitHub issue templates.

## v0.1.0 — 2026-06-28

Initial public release. Non-uniform FFTs (types 1/2/3, dimensions 1/2/3) for
Apple GPUs via Metal/MLX, with a drop-in FINUFFT-compatible Python API and the
`crit64` mixed-precision coordinate setup (fp32 execute, double-precision
plan-time coordinate handling). See `mlx-nufft.pdf` for the method, accuracy
study, and M1 / M5 Max benchmarks.
