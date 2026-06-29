# mlx-nufft — non-uniform FFTs on Apple GPUs (Metal/MLX)

[![tests](https://github.com/martinlachaine/mlx-nufft/actions/workflows/ci.yml/badge.svg)](https://github.com/martinlachaine/mlx-nufft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-lightgrey.svg)](#install)
[![Backend: MLX](https://img.shields.io/badge/backend-MLX%200.31.2-orange.svg)](https://github.com/ml-explore/mlx)

mlx-nufft computes non-uniform fast Fourier transforms — types 1, 2 and 3, in
dimensions 1, 2 and 3 — on Apple-silicon GPUs via Metal/MLX, with a drop-in
mirror of the `finufft` Python API. It runs an fp32 GPU pipeline with the
precision-critical coordinate setup performed in double precision at plan time
("crit64"), so it reaches fp64-grade accuracy on hardware that has no native
double precision.

> **Paper:** [`mlx-nufft.pdf`](mlx-nufft.pdf) — a technical report
> describing the method, accuracy, and performance (types 1/2/3 in 1/2/3D, the
> crit64 precision mechanism, and M1 / M5 Max benchmarks).
> Pin a tagged release rather than tracking `main`.

## Install

Requires an Apple-silicon Mac (Metal/MLX). Install the release directly from
GitHub (not yet on PyPI):

```bash
pip install "git+https://github.com/martinlachaine/mlx-nufft.git@v0.1.0"
```

then `import mlx_nufft`. Dependencies are pinned (notably `mlx==0.31.2`).

For development — running the test/benchmark harness, which uses CPU `finufft`
and `scipy` as references:

```bash
git clone https://github.com/martinlachaine/mlx-nufft && cd mlx-nufft
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
```

Verify the install — run the full correctness suite (each test compares against
CPU `finufft` and/or an exact direct-summation oracle):

```bash
.venv/bin/python harness/run_tests.py
```

It prints a per-test pass/fail summary and exits non-zero on any failure. The
optional VkFFT backend test reports `SKIP` unless the bridge in
`vkfft_bridge/` is built.

## Quickstart

A complete, copy-paste-runnable 2-D type-1 transform (M nonuniform points →
`N1 × N2` uniform Fourier modes):

```python
import numpy as np
import mlx_nufft as finufft

rng = np.random.default_rng(0)
M, N1, N2 = 100_000, 256, 256
x = rng.uniform(-np.pi, np.pi, M)                          # coords in [-pi, pi)
y = rng.uniform(-np.pi, np.pi, M)
c = rng.standard_normal(M) + 1j * rng.standard_normal(M)   # source strengths

fk = finufft.nufft2d1(x, y, c, (N1, N2), eps=1e-6)         # -> (256, 256) complex
```

A fuller runnable script (basic call, plan reuse, and a self-check against an
exact direct DFT — no `finufft` install needed) is in
[`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py
```

## Usage

Drop-in `finufft` API (same call surface):

```python
import mlx_nufft as finufft

fk = finufft.nufft2d1(x, y, c, (N1, N2), eps=1e-6)   # all nufft{1,2,3}d{1,2,3}
plan = finufft.Plan(1, (N1, N2), n_trans=8, eps=1e-6)
plan.setpts(x, y)
fk = plan.execute(c)
```

Native plan classes — `Type3Plan` (type-3 engine) and `Type1PlanND` /
`Type2PlanND` (dims 1–3):

```python
from mlx_nufft import Type3Plan

# plan once (geometry-dependent setup cached), execute per call
plan = Type3Plan((x1, x2, x3), (s1, s2, s3), eps=1e-5, isign=+1)
f = plan.execute(c)            # f[k] = sum_j c[j] exp(i*isign * s_k . x_j)
```

`Type1PlanND` also offers a batched multi-strength execute over one shared
point spread, and a cheap re-point of a fixed-mode-box plan to new coordinates:

```python
from mlx_nufft import Type1PlanND

plan = Type1PlanND((x, y), (N1, N2), eps=1e-5, isign=+1)
fk = plan.execute_batch(cs)               # cs: (B, P) over the same points -> (B, N1, N2)
plan.set_sources((x2, y2), backend="gpu") # re-point without recompiling kernels
fk2 = plan.execute(c2)
```

The double-single phase primitive is also exposed standalone, for callers with
a large-magnitude fp64 phase that cannot be reduced mod 2π in fp32 but want the
`cos`/`sin` on the GPU:

```python
from mlx_nufft import expi, EXPI_MAX_PHASE

z = expi(phi)                    # device complex64 e^{i*phi}, phi an fp64 array
z = expi([a, b, c], isign=-1)    # phase summed in double-single: e^{-i*(a+b+c)}
```

An optional VkFFT-Metal FFT backend is selectable for the type-3 slab path
(`Type3Plan(..., fft_backend="vkfft")`, requires building the bridge in
`vkfft_bridge/`); MLX's FFT is the default. `vkfft_available()` reports whether
the bridge is built.

## Documented differences from `finufft`

- Computation is fp32-grade (crit64): `eps` below 1e-6 clamps with a warning;
  complex128 inputs are accepted and returned but transformed at fp32 grade.
- `modeord=1` (FFT ordering) is not implemented.
- 1D/2D type 3 run as degenerate slices of the 3D type-3 kernel.
- Plans hold points as plan state (`setpts`); `out=` and multi-vector
  `(n_trans, ...)` shapes mirror finufft, and `Plan.execute_adjoint`
  (finufft 2.5) is supported for all three types.

## Layout

- `mlx_nufft/` — the library: `gpu_t3.py` (type-3 engine), `nd.py`
  (`Type1PlanND`/`Type2PlanND`), `types12.py`, `dfmath.py` (the `expi` /
  double-single primitive), `sizing.py` (kernel/grid sizing), `api.py` (the
  `finufft`-compatible surface), `vkfft_backend.py`.
- `examples/` — runnable, dependency-light usage examples
  (`quickstart.py`).
- `harness/` — correctness tests (`test_*.py`), the suite runner
  (`run_tests.py`), the acceptance/benchmark runner, and the CPU-reference
  oracle.
- `vkfft_bridge/` — optional VkFFT-Metal backend build.

## Development and validation

mlx-nufft is AI-assisted, human-directed research software. Its scope, numerical
requirements, validation strategy, acceptance criteria, and release decisions
are the author's; generative AI tools accelerated implementation, refactoring,
test scaffolding, and documentation.

Generated code was not trusted by default. Every component was validated against
independent references — CPU FINUFFT, exact direct-summation oracles on small
problems, transform-convention and adjoint checks, full dimension/type coverage,
dtype/device behavior, and API-parity tests (see `harness/`).

The author is responsible for the released software — its design, limitations,
and maintenance. Known limitations and hardware assumptions are documented;
corrections are welcome via the issue tracker.

## License & citation

Apache-2.0 (see `LICENSE`). mlx-nufft is an independent implementation that
follows the FINUFFT/cuFINUFFT algorithms and was validated against FINUFFT;
see `NOTICE` for attribution and the methods papers to cite, and `CITATION.cff`
to cite mlx-nufft itself.
