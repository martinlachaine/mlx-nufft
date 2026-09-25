# mlx-nufft — design and validation

The [technical report](mlx-nufft.pdf) describes the original v0.1
implementation, its mixed-precision coordinate setup (crit64), and the accuracy,
runtime, and memory results measured on M1 and M5 Max systems. The report was
included in [v0.1.0](https://github.com/martinlachaine/mlx-nufft/tree/v0.1.0)
and has not been revised for v0.2.

The core numerical approach remains relevant, but the implementation and
performance discussion should be read in that version context. In v0.2,
type-1/2 plans use GPU double-single coordinate setup by default, type-3 setup
uses a cached kernel Fourier-transform approximation with a quadrature fallback,
and the spreading, interpolation, and FFT execution paths have been optimized.
See the [v0.2.0 changelog](CHANGELOG.md#v020---2026-07-02) for details and
reported performance improvements. The report's figures and tables are v0.1
measurements, not benchmarks of the current release.

The project link on the report's first page uses the old `mcnufft` name. The
current repository is [martinlachaine/mlx-nufft](https://github.com/martinlachaine/mlx-nufft),
as listed in the report's Availability section.
