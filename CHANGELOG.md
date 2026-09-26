# Changelog

## v0.3.1 - 2026-09-25

- Type-1 spreading uses the tile kernels from kernel width 4 instead of 5. Measured against the direct atomic spread at width 4: 1.1x to 1.9x on 2D and 3D 128^3 with random and clustered points on both an M5 Max and an M1, so it is a global default, not a device rule.
- `harness/bench_multiplier.py` no longer un-permutes type-2 outputs, which every type-2 path already returns in caller order; its type-2 accuracy column read about 1.4 before. Found by the M1 run of the paper suite.
- `harness/run_acceptance.py` runs the fp64 CPU oracle when its predicted grids fit in half of the machine's RAM instead of a fixed 9 GiB.

## v0.3.0 - 2026-09-25

Note: the benchmarks in the technical report (`mlx-nufft.pdf`) describe v0.1; a revised report is in preparation.

This release is about correctness at tight tolerances, faster type-2 and type-3 execution, and much faster plan construction. The numbers below come from the per-stage profiler in `harness/profile_stages.py` (median of 7 runs on an M5 Max) against the committed v0.2.0 baseline. The public API is unchanged, and every new default keeps the previous path selectable.

**Accuracy**
- Type 3 at eps at or below 1e-5 now delivers what single-precision FINUFFT delivers. The type-3 path runs the kernel at upsampling factor 1.25, where fp32 grid error is amplified by the deconvolution. v0.2.0 used kernel widths of 9 and 10 there and could land at 1e-4 to 3e-4 on isotropic 3D problems. The width is now capped at 8 when three axes carry a full band, FINUFFT's own fp32 rule, and at 10 otherwise, so thin slabs keep their better widths. Isotropic 3D type 3 at eps=1e-6 improves from 3.3e-4 to 5.3e-5; the anisotropic slab case stays in its 1.4e-5 class.
- Documented floor: with upsampling factor 1.25 in fp32 the achievable error is about 5e-5 for eps at or below 1e-5, and at eps=1e-4 these paths sit 1.5 to 2.6 times above FINUFFT's fp32 result. Upsampling factor 2 is unaffected (1.2e-5 at eps=1e-5, 1.5e-6 at 1e-6).

**Faster execution**
- Types 1 and 2 in 3D default to upsampling factor 1.25 on grids of at least 32768 modes when eps is 1e-4 or looser (type 2) or 1e-3 or looser (type 1, and 1e-4 on grids of 2^24 modes or more). At eps=1e-3, 128^3 and 256^3 run 2.0 to 2.2x faster for type 1 (20.5 to 10.3 ms at 256^3) and 1.8 to 2.1x for type 2 (17.5 to 8.4 ms); the achieved error at those eps is 1.5 to 1.7 times the sigma-2 result, in the same eps bracket. Pass `upsampfac=2.0` to keep the previous behavior; `MLX_NUFFT_UPSAMPFAC` overrides the default for comparisons.
- Type 2: the zero-pad before each axis FFT is one tiled kernel instead of four, with no zero temporary (1.5 grid passes instead of 2.5). 3D 256^3: 21.0 to 17.6 ms (1.19x); 128^3: 1.14 to 1.16x.
- Type 3: the FFT chain keeps every FFT on a contiguous axis, with tiled transposes on full 3D grids (MLX's FFT on a non-last axis was copying the grid twice), a dense DFT kernel for axes of 32 cells or fewer, and the four-step twiddle fused into its transpose. 3D generic: 65.0 to 60.4 ms (1.08x) at eps=1e-3 and 70.0 to 63.2 ms (1.11x) at 1e-5; 1D: 23.3 to 20.2 ms (1.15x) at 1e-5.
- The public `Plan` batches `n_trans` vectors through one spread and whole-batch FFTs for type-1 plans on the global-memory spread path (2.7x on small plans). Plans on the tile path keep the per-vector loop, which measured faster there.

**Faster plan construction**
- Type-1 and type-2 plans evaluate the deconvolution through the certified Chebyshev proxy already used by type 3, with the quadrature as fallback. A 1D plan over 2^20 modes builds in 0.023 s instead of 0.39 s.

**Tried and not adopted**
- Piecewise-polynomial (Horner) kernel evaluation, the main cuFINUFFT spreading lever, is neutral on Apple GPUs (0.94 to 1.01x): exp2 is cheap and the spread is bound by atomics and memory traffic. Kept on a branch.
- Staging the global-memory spread's weights once per point gave identical times. FINUFFT 2.5's re-tuned beta for upsampling factor 1.25 was worse on 23 of 28 measured cases for this kernel.

**New**
- `harness/profile_stages.py`, a per-stage profiler with committed baseline tables, plus accuracy studies for the width cap and the upsampling policy. Module-level switches with environment overrides (`MLX_NUFFT_FFT_STRATEGY`, `MLX_NUFFT_PAD_PATH`, `MLX_NUFFT_UPSAMPFAC`) select the previous paths for comparison.
- The optional VkFFT backend measures 1.2x on the 3D type-3 case on top of this release. Making it the default with a prebuilt bridge is planned for v0.4, together with device-resident inputs and outputs.

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
