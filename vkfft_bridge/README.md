# VkFFT-Metal bridge (optional `fft_backend="vkfft"`)

Builds `libvkfft_bridge.dylib`, a C-ABI shim that runs complex64 FFTs in place
on an MLX array's unified-memory buffer (wrapped as an `MTL::Buffer` via
`bytesNoCopy` — zero copy, no MLX/pybind linkage). Loaded by
`mlx_nufft/vkfft_backend.py` via ctypes.

## Coverage
- `vkfft_fft2_inplace` — batched 2D FFT over the leading axis of a
  (nb, n_outer, n_contig) array: the type-3 SLAB lateral FFT (natural order,
  identity-scramble gather).
- `vkfft_fftn_inplace` — whole-array 1D/2D/3D FFT: the type-3 non-slab inner
  grid (opt-in via `fft_backend="vkfft"` on `Type3Plan`). The type-1/2 ND
  plans keep the MLX path: their progressive per-axis crop/pad FFT measures
  faster than a whole-grid VkFFT call there.

Axis lengths must factor into radix 2,3,5,7,11,13 (always true for the
`next235even` grid sizes); unsupported lengths fall back to the MLX path.
VkFFT plans are cached per (dim, shape, batch, normalize); direction is a
launch-time argument. `coalescedMemory=32` is baked in (measured best on
Apple silicon: 640^3 c2c 50.7 -> 37.1 ms vs the Metal default of 64).

## Build
```bash
vkfft_bridge/build.sh          # clones VkFFT (header-only) + one clang++ call
```
Requires Apple clang + Metal/Foundation/QuartzCore frameworks (Command Line
Tools). No cmake, no pybind/nanobind. Output: `vkfft_bridge/libvkfft_bridge.dylib`
(found automatically; or set `MLX_NUFFT_VKFFT_LIB`).

## Why opt-in
MLX is the **validated reference** FFT. VkFFT is ~2.58× on the dominant slab
FFT (→ ~2× whole-execute) and 1.4× on the non-slab type-3 3D FFT (640^3:
55 -> 38 ms), but pulls in an external dependency + a native build, so it is
off by default and only used when `fft_backend="vkfft"` is requested and the
dylib is present. The clone (`VkFFT/`) and binary are git-ignored.
