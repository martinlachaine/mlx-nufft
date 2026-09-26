# Contributing to mlx-nufft

Thanks for your interest. This is a small, focused library; contributions,
bug reports, and questions are all welcome.

## Reporting a bug

Open a [GitHub issue](https://github.com/martinlachaine/mlx-nufft/issues) and
include:

- your hardware and OS (e.g. *Apple M2 Pro, macOS 15.4*),
- Python, `mlx-nufft`, and `mlx` versions (`pip show mlx-nufft mlx`),
- a minimal snippet that reproduces the problem, and
- what you expected versus what happened (error text, or the wrong numbers and
  the reference you compared against).

Accuracy reports are most useful as a relative-L2 error against CPU `finufft`
or a direct-summation oracle at a stated `eps`; see `harness/` for the
patterns used in the test suite.

## Development setup

Requires an Apple-silicon Mac (Metal/MLX).

```bash
git clone https://github.com/martinlachaine/mlx-nufft && cd mlx-nufft
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
```

## Running the tests

```bash
.venv/bin/python harness/run_tests.py
```

This runs every `harness/test_*.py` and exits non-zero on any failure. Please
make sure it passes before opening a pull request, and add or extend a test
under `harness/` for any behavior change.

The `mlx` dependency is pinned (`mlx==0.31.2`) because the library works around
version-specific Metal FFT behavior. If you need to bump it, re-validate
`harness/test_gpu_small.py` and the full suite first, and say so in the PR.

## Performance changes

Measure before and after with the per-stage profiler, on an otherwise idle GPU:

```bash
.venv/bin/python harness/profile_stages.py            # full matrix, ~10 s
.venv/bin/python harness/profile_stages.py --cases t2_3d_256,t3_3d_generic512
```

It reports the median of 7 warm runs per stage and writes `results/<tag>.md`;
the tables committed as `harness/PROFILE_*.md` are the reference points for
each release. A/B runs must be taken within a few minutes of each other,
interleaved when possible, because some Apple GPUs (the M1 Mac mini measured)
drift between performance states by up to 25 percent over tens of minutes.
The profiler's calibration timing (`--calib-only` runs it alone) is the
cross-session reference, and a run whose start and end calibration differ by
more than 5 percent should be repeated.

A performance change should come with a way to compare both paths in one
build (a module switch or one of the environment overrides
`MLX_NUFFT_FFT_STRATEGY`, `MLX_NUFFT_PAD_PATH`, `MLX_NUFFT_UPSAMPFAC`) and with
a test that pins the outputs: bit-identical for pure data-movement changes,
within the existing CPU-reference tolerances otherwise. Changes that measure
neutral are not merged, even when they look like they should help; the spread
kernels on Apple GPUs are bound by atomics and memory traffic, not arithmetic.

## Pull requests

- Keep changes focused and match the surrounding code style.
- Note any change to the public API or to `finufft` parity in the PR
  description, and update `README.md` / `CHANGELOG.md` accordingly.
- By contributing, you agree your contributions are licensed under the
  project's [Apache-2.0 license](LICENSE).
