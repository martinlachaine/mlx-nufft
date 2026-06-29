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
or a direct-summation oracle at a stated `eps` — see `harness/` for the
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

## Pull requests

- Keep changes focused and match the surrounding code style.
- Note any change to the public API or to `finufft` parity in the PR
  description, and update `README.md` / `CHANGELOG.md` accordingly.
- By contributing, you agree your contributions are licensed under the
  project's [Apache-2.0 license](LICENSE).
