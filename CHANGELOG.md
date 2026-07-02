# Changelog

## Unreleased

Performance release: 1.4–2.8x faster whole-transform execution across types
and dimensions, and 30–100x faster plan construction (t1 3D M=1e7 plan
5.5 s -> 0.05 s; t3 N=1024 M=P=1e6 plan 2.0 s -> 0.06 s; one-shot `nufft3d3`
2.2 s -> 0.26 s), on Apple silicon (measured on M5 Max, 128 GB; default MLX
backend, identical accuracy gates — the full acceptance matrix and adjoint
checks pass unchanged). No breaking API changes (additive kwargs only);
numerical results agree with the previous release within the documented fp32
atomic-ordering noise.

- ES spreading kernel evaluated via `metal::fast::exp2`/`fast::sqrt`
  (end-to-end rel-L2 1.70e-6 vs 1.55e-6 at eps=1e-6 — below the fp32
  pipeline floor).
- Type-1 output-driven spread reworked cuFINUFFT-style: points processed in
  batches with cooperatively staged per-point 1D kernel weights and a flat
  tap loop (two threadgroup barriers per batch instead of one per point;
  all 256 lanes busy). 2D M=1e7 spread 14.3 -> 5.0 ms.
- 1D type-1 spread switched to an exclusive-ownership (gather-formulation)
  kernel: each threadgroup owns an unpadded region of the fine grid
  outright, scans its three neighbour bins simdgroup-autonomously, and
  flushes with plain stores — no global atomics and no output zero-init
  (1D M=1e7 spread 3.8 -> 2.0 ms, whole 5.6 -> 3.6 ms). The same kernel
  measures at par in 2D and slower in 3D (neighbour-scan and staging
  duplication exceed the padded kernel's flush budget), so dims 2-3 keep
  the padded-tile spread.
- Type-3 non-slab point kernels ported to the same output-driven machinery
  (w=9 tiles), with a zero-skip tile flush; targets are cell-sorted at plan
  time and gathered perm-indexed with the postphase folded in
  (spread 40 -> 14 ms, gather 38 -> 5 ms at P=M=1e6).
- Progressive FFT: type 1 crops each axis to its mode band immediately after
  that axis's FFT (type 2 pads the mirror way), with axis cycling keeping
  every FFT contiguous-last-axis — FFT work drops to
  (1 + 1/sigma + 1/sigma^2)/3 in 3D (t1 fft+crop 27.5 -> 10.9 ms at 512^3).
- Type-3 slab pipeline batched: fused twiddle+transpose Metal kernel in the
  lateral four-step, and one full-z gather launch instead of nu3 accumulating
  launches (anisotropic whole-execute 320 -> 264 ms; with VkFFT 178 -> 160).
- Sort + prephase fused into the type-3 spread kernels; execute-path input
  conversions no longer copy complex64 arrays. Peak execute memory is
  unchanged (type 3) or lower (types 1/2: progressive crop trims the
  intermediates, 6.3 -> 4.0 GiB on the 3D M=1e7 case).
- Type-2 gather: the threadgroup-tiled OD interp is replaced by a
  CELL-SORTED NAIVE gather (per-target thread over cell-sorted targets,
  perm-indexed output write so caller order stays free) — the round-1
  type-3 finding holds for type 2 at every measured density: Apple's L1/L2
  dedups the overlapping w^d neighbourhood reads without explicit staging
  (3D 256^3 gather 21 -> 4 ms at M=1e6, 28 -> 16 ms at M=1e7; 2D/1D tie).
  Outputs are bit-identical between the two gathers;
  `spread_method="od"` retains the tiled path. The internal gather sort
  applies regardless of `sort_points`, which keeps governing only the
  type-1 source-side semantics.
- ND plan construction fast path: `points_backend="auto"|"host"|"gpu"` on
  `Type1PlanND`/`Type2PlanND` (default auto = gpu) routes `__init__` point
  setup through the df64 Metal + `mx.argsort` path already validated by
  `set_sources(backend="gpu")`; ES cell indices are bit-identical to the
  host fp64 path. 3D M=1e7 plan build ~5.6 s -> 0.05 s (type 1),
  ~2.9 s -> 0.04 s (type 2); `"host"` preserves the numpy setup for exact
  reproducibility.
- Optional VkFFT backend extended with a whole-array 1D/2D/3D in-place FFT:
  `fft_backend="vkfft"` now also covers the non-slab type-3 inner-grid FFT
  (640^3: 54 -> 36 ms; whole t3 generic P=1e6 94 -> 78 ms). Still opt-in;
  MLX remains the validated default.
- Type-3 plan construction ~20x faster (generic N=1024, M=P=1e6: 2.3 s ->
  ~0.1 s), which is most of one-shot `nufft3d3` latency: the M-point
  ES-kernel Fourier transform behind the target deconvolution is evaluated
  through a cached fp64 Chebyshev fit validated to <= 1e-12 relative against
  the 128-node quadrature (falls back to the quadrature whenever the fit
  cannot certify that bound); crit64 plans route their initial source setup
  through the validated df64 GPU path; the target cell sort runs on the GPU.
  tdec deviation at plan targets <= 1.2e-13 relative; end-to-end accuracy
  unchanged (anisotropic subset-oracle rel-L2 1.51e-5 -> 1.49e-5).

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
