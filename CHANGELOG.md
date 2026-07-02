# Changelog

## v0.2.0 - 2026-07-02

Note: the benchmarks in the technical report (`mlx-nufft.pdf`) describe v0.1 and predate the speedups in this release.

This release makes the transforms substantially faster on Apple silicon while leaving the public API and the numerical results unchanged. Whole-transform execution is 1.4 to 2.8 times faster across all types and dimensions, plan construction is 30 to 100 times faster, and a one-shot `nufft3d3` call drops from about 2.2 seconds to 0.26 seconds. Every change is additive, so existing code continues to work, and the full acceptance matrix and adjoint checks pass at the same accuracy as before. Peak execute memory is unchanged for type 3 and lower for types 1 and 2.

**Faster execution**
- The spreading and interpolation kernels were reworked in the style of cuFINUFFT, processing points in batches with cooperatively staged weights rather than one point at a time.
- Type-1 transforms now crop each axis to its mode band as soon as that axis has been transformed, and type 2 pads in the mirror way, which cuts three-dimensional FFT work to roughly 58 percent and avoids strided transposes.
- The type-3 slab pipeline was batched, fusing the twiddle and transpose into one kernel and replacing many small gather launches with a single pass.
- Type-2 interpolation now uses a cell-sorted gather that beats the previous tiled version at every density tested, with bit-identical results.

**Faster plan construction**
- Plan setup for the ND types runs on the GPU by default, through the same double-single path already used by `set_sources`, which cuts a large 3D plan build from several seconds to about 0.05 seconds.
- Type-3 plan setup evaluates the kernel Fourier transform through a cached Chebyshev fit that is certified to 1e-12 against the reference quadrature and falls back to the quadrature whenever it cannot certify, so accuracy is unaffected.

**New options, all optional with existing defaults preserved**
- `spread_method` selects the spreading kernel, with the faster method chosen automatically.
- `points_backend` selects host or GPU plan setup, and defaults to GPU.
- The optional `fft_backend="vkfft"` now covers 1D, 2D, and 3D transforms, including the non-slab type-3 grid. MLX remains the default and the validated path.

**Trade-offs**
- Re-pointing a type-3 plan with `set_sources` is somewhat slower because of an added target sort, though it stays far faster than rebuilding the plan and each execute more than repays it.
- The full detail of every change, including the measurements and the approaches that were tried and rejected, is recorded in the git history.

## v0.1.3 - 2026-06-28

Infrastructure only. No library or numerical behavior changes. First release
archived on Zenodo, giving the project a citable DOI.

## v0.1.2 - 2026-06-28

Documentation and packaging only. No library or numerical behavior changes.

- Published to PyPI: `pip install mlx-nufft`.
- README install section updated to lead with PyPI, plus a PyPI version badge.
- First release published through the Trusted Publishing workflow on release.

## v0.1.1 - 2026-06-28

Documentation and infrastructure only. No library or numerical behavior
changes. The `mlx_nufft` package is identical to v0.1.0.

- Continuous integration: the correctness suite runs on Apple-silicon GitHub
  runners (`.github/workflows/ci.yml`); large type-3 cases that need a big-GPU
  Metal buffer skip cleanly on small/CI GPUs.
- `harness/run_tests.py`: one-command runner for the full correctness suite.
- `examples/quickstart.py`: runnable, dependency-light demo that self-checks
  accuracy against an exact direct DFT.
- README: copy-pasteable quickstart, "verify the install" section, status
  badges, and a "Development and validation" note.
- `CONTRIBUTING.md` and GitHub issue templates.

## v0.1.0 - 2026-06-28

Initial public release. Non-uniform FFTs (types 1/2/3, dimensions 1/2/3) for
Apple GPUs via Metal/MLX, with a drop-in FINUFFT-compatible Python API and the
`crit64` mixed-precision coordinate setup (fp32 execute, double-precision
plan-time coordinate handling). See `mlx-nufft.pdf` for the method, accuracy
study, and M1 / M5 Max benchmarks.
