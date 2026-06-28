# Changelog

## v0.1.0 — 2026-06-28

Initial public release. Non-uniform FFTs (types 1/2/3, dimensions 1/2/3) for
Apple GPUs via Metal/MLX, with a drop-in FINUFFT-compatible Python API and the
`crit64` mixed-precision coordinate setup (fp32 execute, double-precision
plan-time coordinate handling). See `mlx-nufft.pdf` for the method, accuracy
study, and M1 / M5 Max benchmarks.
