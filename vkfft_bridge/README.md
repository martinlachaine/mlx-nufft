# VkFFT-Metal bridge (optional `fft_backend="vkfft"`)

Builds `libvkfft_bridge.dylib`, a C-ABI shim that runs a batched 2D complex64
FFT in place on an MLX array's unified-memory buffer (wrapped as an
`MTL::Buffer` via `bytesNoCopy` — zero copy, no MLX/pybind linkage). Loaded by
`mlx_nufft/vkfft_backend.py` via ctypes.

## Build
```bash
vkfft_bridge/build.sh          # clones VkFFT (header-only) + one clang++ call
```
Requires Apple clang + Metal/Foundation/QuartzCore frameworks (Command Line
Tools). No cmake, no pybind/nanobind. Output: `vkfft_bridge/libvkfft_bridge.dylib`
(found automatically; or set `MLX_NUFFT_VKFFT_LIB`).

## Why opt-in
MLX is the **validated reference** FFT. VkFFT is ~2.58× on the dominant slab
FFT (→ ~2× whole-execute) but pulls in an external dependency + a native build,
so it is off by default and only used when `fft_backend="vkfft"` is requested
and the dylib is present. The clone (`VkFFT/`) and binary are git-ignored.
