# mlx-nufft: technical report status

The [technical report](mlx-nufft.pdf) describes the original v0.1
implementation: the fp32 GPU pipeline with double-precision coordinate setup at
plan time (crit64), the accuracy study, and the runtime and memory results on an
M1 Mac mini and an M5 Max MacBook Pro. The method and the accuracy analysis
still apply. The performance figures do not: v0.2 and v0.3 changed plan setup,
spreading, interpolation, and the FFT chain, and v0.3 also corrected type-3
accuracy at tight tolerances. Current numbers are in [CHANGELOG.md](CHANGELOG.md)
and in the committed profiler tables under `harness/`. A revised report covering
v0.3 is in preparation.

The project link on the report's first page uses the old `mcnufft` name. The
repository is [martinlachaine/mlx-nufft](https://github.com/martinlachaine/mlx-nufft).
