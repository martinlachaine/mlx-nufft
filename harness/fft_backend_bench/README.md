# FFT-backend comparison (harness)

Screens whether a different Metal FFT backend beats MLX's four-step for the
type-3 slab pipeline's dominant FFT stage. It benchmarks the exact per-slab
work the pipeline does: a batched 2D complex64 FFT over `nu3` z-slabs of
`(nu1, nu2)`, at the non-power-of-two grids that matter (e.g. nu1=nu2=7200).

Backends screened: MLX's four-step (the pipeline default), Apple's MPSGraph
`fastFourierTransform`, and VkFFT-Metal. Each is checked for correctness
against `numpy.fft.fft2`.

## Reproduce
```bash
# MLX baseline (uses the project venv)
.venv/bin/python harness/fft_backend_bench/bench_mlx_fft.py            # run per-size

# MPSGraph (Apple framework; needs swiftc)
swiftc -O harness/fft_backend_bench/mpsgraph_fft.swift -o /tmp/mps && /tmp/mps 7200 24

# VkFFT-Metal: clone VkFFT (header-only) next to the drivers, then clang++
git clone --depth 1 https://github.com/DTolm/VkFFT harness/fft_backend_bench/VkFFT
clang++ -std=c++17 -O3 -DVKFFT_BACKEND=5 \
  -I harness/fft_backend_bench/VkFFT/vkFFT -I harness/fft_backend_bench/VkFFT/metal-cpp \
  -framework Metal -framework Foundation -framework QuartzCore \
  harness/fft_backend_bench/vkfft_bench.cpp -o /tmp/vkfft && /tmp/vkfft 7200 24 8
# correctness: vkfft_verify.cpp (n=8) vs numpy np.fft.fft2
```

**Caution:** large grids (n_up ≥ 9000 → ≥15 GB Metal buffers) repeatedly
allocated across MLX + an external backend can destabilize the GPU driver.
Keep isolated FFT benchmarks at n_up ≤ 7200 unless memory is capped.
