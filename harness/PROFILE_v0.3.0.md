# Per-stage execute profile: mlx-nufft 0.2.0

- machine: Apple M5 Max, 128 GB, macOS-27.0-arm64-arm-64bit-Mach-O
- mlx 0.31.2, python 3.13.12
- git fd7b517a2c96 (perf/v0.3)
- run: 2026-09-25T20:56:46+00:00
- type-2 pad path: fused (nd.PAD_PATH)
- protocol: per measurement, the GPU is kept busy for 100 ms (matmul loop) so the clocks are at steady state, then 2 warm-up executes, then 7 timed executes; median and min reported; mx.synchronize() before and after every timed region. Stage times come from a second set of runs that re-issue execute()'s kernel sequence with an eval + synchronize after each stage; sum/whole is the sum of stage medians over the whole-execute median (the per-stage syncs cost pipelining, so sum/whole above 1 is expected; rows outside 1 +/- 0.15 are flagged). Stage percentages are of the whole-execute median. A rep more than 2x its median triggered one re-run of that measurement (noted per row). Measured stage-boundary floor: 0.257 ms per eval + synchronize; for flagged rows the Reading section puts the excess next to the cost of the stage boundaries at that floor.
- inputs: n_trans=1, complex64, fixed seed, default plan options (upsampfac auto for types 1/2: nd.py's policy takes sigma=1.25 on 3D grids of at least 2^15 modes at loose eps and 2.0 otherwise, recorded as sigma in each case's info, and MLX_NUFFT_UPSAMPFAC=2.0|1.25 forces one for A/B runs; 1.25 for type 3; spread_method auto, points_backend auto, MLX FFT, crit64). h2d/d2h are the input upload and result download that execute() performs.

## Type 1, 1D

| case | eps | whole med (ms) | h2d | spread | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|
| t1 1D N=2^20 M=1e6 | 1e-03 | 1.08 | 0.16 (15%) | 0.31 (29%) | 0.55 (51%) | 0.19 (17%) | 0.15 (14%) | 1.06 | 1.25 FLAG |
| t1 1D N=2^20 M=1e6 | 1e-05 | 1.04 | 0.15 (15%) | 0.32 (31%) | 0.56 (54%) | 0.19 (18%) | 0.15 (14%) | 0.97 | 1.32 FLAG |

## Type 1, 2D

| case | eps | whole med (ms) | h2d | spread | fft_y | crop_y | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 2D 512^2 M=1e6 | 1e-03 | 0.81 | 0.15 (18%) | 0.55 (68%) | 0.19 (23%) | 0.16 (20%) | 0.17 (21%) | 0.16 (20%) | 0.05 (7%) | 0.80 | 1.77 FLAG |
| t1 2D 512^2 M=1e6 | 1e-05 | 0.85 | 0.15 (18%) | 0.58 (68%) | 0.19 (22%) | 0.16 (19%) | 0.16 (19%) | 0.18 (22%) | 0.05 (6%) | 0.83 | 1.75 FLAG |

## Type 1, 3D

| case | eps | whole med (ms) | h2d | spread | fft_z | crop_z | fft_y | crop_y | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 3D 128^3 M=1e6 | 1e-03 | 2.68 | 0.16 (6%) | 1.66 (62%) | 0.29 (11%) | 0.25 (9%) | 0.24 (9%) | 0.21 (8%) | 0.23 (8%) | 0.25 (9%) | 0.28 (10%) | 2.66 | 1.33 FLAG |
| t1 3D 128^3 M=1e6 | 1e-05 | 4.76 | 0.16 (3%) | 2.95 (62%) | 0.72 (15%) | 0.42 (9%) | 0.42 (9%) | 0.28 (6%) | 0.27 (6%) | 0.25 (5%) | 0.28 (6%) | 4.70 | 1.21 FLAG |
| t1 3D 256^3 M=1e6 | 1e-03 | 10.28 | 0.18 (2%) | 3.42 (33%) | 1.23 (12%) | 1.00 (10%) | 0.98 (10%) | 0.81 (8%) | 0.81 (8%) | 0.97 (9%) | 2.12 (21%) | 10.04 | 1.12 |
| t1 3D 256^3 M=1e6 | 1e-05 | 24.51 | 0.18 (1%) | 11.00 (45%) | 4.26 (17%) | 2.26 (9%) | 2.21 (9%) | 1.20 (5%) | 1.17 (5%) | 1.02 (4%) | 2.12 (9%) | 24.24 | 1.04 |

## Type 2, 1D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|
| t2 1D N=2^20 M=1e6 | 1e-03 | 1.09 | 0.15 (14%) | 0.22 (20%) | 0.55 (50%) | 0.24 (22%) | 0.14 (13%) | 1.06 | 1.20 FLAG |
| t2 1D N=2^20 M=1e6 | 1e-05 | 1.14 | 0.16 (14%) | 0.23 (20%) | 0.56 (50%) | 0.25 (22%) | 0.15 (13%) | 1.09 | 1.19 FLAG |

## Type 2, 2D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | pad_y | fft_y | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t2 2D 512^2 M=1e6 | 1e-03 | 0.66 | 0.06 (8%) | 0.18 (28%) | 0.17 (25%) | 0.17 (26%) | 0.18 (27%) | 0.23 (34%) | 0.14 (22%) | 0.64 | 1.71 FLAG |
| t2 2D 512^2 M=1e6 | 1e-05 | 0.62 | 0.06 (9%) | 0.19 (31%) | 0.17 (27%) | 0.17 (27%) | 0.17 (27%) | 0.25 (40%) | 0.14 (23%) | 0.60 | 1.84 FLAG |

## Type 2, 3D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | pad_y | fft_y | pad_z | fft_z | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2 3D 128^3 M=1e6 | 1e-03 | 1.58 | 0.29 (18%) | 0.24 (15%) | 0.20 (13%) | 0.23 (15%) | 0.24 (15%) | 0.26 (16%) | 0.29 (18%) | 0.42 (26%) | 0.15 (9%) | 1.55 | 1.47 FLAG |
| t2 3D 128^3 M=1e6 | 1e-05 | 3.18 | 0.29 (9%) | 0.26 (8%) | 0.28 (9%) | 0.36 (11%) | 0.42 (13%) | 0.56 (18%) | 0.67 (21%) | 0.97 (30%) | 0.15 (5%) | 3.14 | 1.24 FLAG |
| t2 3D 256^3 M=1e6 | 1e-03 | 8.35 | 2.13 (25%) | 1.27 (15%) | 0.83 (10%) | 0.90 (11%) | 0.97 (12%) | 1.08 (13%) | 1.16 (14%) | 0.86 (10%) | 0.16 (2%) | 7.92 | 1.12 |
| t2 3D 256^3 M=1e6 | 1e-05 | 18.19 | 2.13 (12%) | 1.51 (8%) | 1.20 (7%) | 1.81 (10%) | 2.19 (12%) | 3.39 (19%) | 4.14 (23%) | 2.54 (14%) | 0.16 (1%) | 18.05 | 1.05 |

## Type 3, 1D/3D

| case | eps | whole med (ms) | h2d | spread | pad+deconv | fft_z | fft_y | fft_x | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-03 | 60.41 | 0.05 (0%) | 3.35 (6%) | 6.60 (11%) | 8.00 (13%) | 16.60 (27%) | 24.57 (41%) | 1.40 (2%) | 0.08 (0%) | 59.90 | 1.00 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-05 | 63.60 | 0.05 (0%) | 4.72 (7%) | 6.62 (10%) | 8.02 (13%) | 16.35 (26%) | 24.63 (39%) | 3.13 (5%) | 0.07 (0%) | 63.51 | 1.00 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-03 | 14.69 | 0.15 (1%) | 7.53 (51%) | 0.67 (5%) | 0.89 (6%) | 0.91 (6%) | 4.05 (28%) | 0.43 (3%) | 0.16 (1%) | 14.48 | 1.01 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-05 | 20.20 | 0.16 (1%) | 4.87 (24%) | 1.37 (7%) | 1.75 (9%) | 1.79 (9%) | 8.93 (44%) | 1.05 (5%) | 0.16 (1%) | 19.81 | 0.99 |

## Plan variants, build time and memory

staged vs whole rel-L2 compares the staged reproduction's output with execute()'s; nonzero values come from the run-to-run summation order of the atomic spreads, not from the staging.

| case | eps | w | n_up | pts/cell | path | plan build (s) | peak GiB | staged vs whole rel-L2 |
|---|---|---|---|---|---|---|---|---|
| t1 1D N=2^20 M=1e6 | 1e-03 | 4 | 2097152 | 0.477 | gm | 0.024 | 0.15 | 1.3e-07 |
| t1 1D N=2^20 M=1e6 | 1e-05 | 6 | 2097152 | 0.477 | od_ex | 0.018 | 0.14 | 1.3e-07 |
| t1 2D 512^2 M=1e6 | 1e-03 | 4 | 1024x1024 | 0.954 | gm | 0.006 | 0.07 | 1.6e-07 |
| t1 2D 512^2 M=1e6 | 1e-05 | 6 | 1024x1024 | 0.954 | od | 0.007 | 0.07 | 1.3e-07 |
| t1 3D 128^3 M=1e6 | 1e-03 | 5 | 160x160x160 | 0.244 | od | 0.010 | 0.20 | 5.5e-07 |
| t1 3D 128^3 M=1e6 | 1e-05 | 6 | 256x256x256 | 0.0596 | od | 0.008 | 0.49 | 1.3e-07 |
| t1 3D 256^3 M=1e6 | 1e-03 | 5 | 320x320x320 | 0.0305 | od | 0.009 | 1.35 | 4.5e-07 |
| t1 3D 256^3 M=1e6 | 1e-05 | 6 | 512x512x512 | 0.00745 | od | 0.014 | 3.66 | 1.2e-07 |
| t2 1D N=2^20 M=1e6 | 1e-03 | 4 | 2097152 | 0.477 | sorted_gather | 0.018 | 0.14 | 0.0e+00 |
| t2 1D N=2^20 M=1e6 | 1e-05 | 6 | 2097152 | 0.477 | sorted_gather | 0.015 | 0.14 | 0.0e+00 |
| t2 2D 512^2 M=1e6 | 1e-03 | 4 | 1024x1024 | 0.954 | sorted_gather | 0.006 | 0.07 | 0.0e+00 |
| t2 2D 512^2 M=1e6 | 1e-05 | 6 | 1024x1024 | 0.954 | sorted_gather | 0.006 | 0.07 | 0.0e+00 |
| t2 3D 128^3 M=1e6 | 1e-03 | 5 | 160x160x160 | 0.244 | sorted_gather | 0.008 | 0.20 | 0.0e+00 |
| t2 3D 128^3 M=1e6 | 1e-05 | 6 | 256x256x256 | 0.0596 | sorted_gather | 0.008 | 0.49 | 0.0e+00 |
| t2 3D 256^3 M=1e6 | 1e-03 | 5 | 320x320x320 | 0.0305 | sorted_gather | 0.008 | 1.34 | 0.0e+00 |
| t2 3D 256^3 M=1e6 | 1e-05 | 6 | 512x512x512 | 0.00745 | sorted_gather | 0.009 | 3.65 | 0.0e+00 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-03 | 5 | 640x640x640 | 0.000381 | od_spread/sorted_gather | 0.035 | 5.88 | 1.7e-07 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-05 | 8 | 640x640x640 | 0.000381 | od_spread/sorted_gather | 0.037 | 5.88 | 1.7e-05 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-03 | 5 | 90000x16x16 | 0.0434 | gm_spread/sorted_gather | 0.107 | 1.14 | 3.0e-07 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-05 | 9 | 90000x24x24 | 0.0193 | od_spread/sorted_gather | 0.112 | 2.42 | 1.2e-06 |

## Reading

- t1 1D N=2^20 M=1e6 eps=1e-03: dominant stage fft_x at 51% (0.55 of 1.08 ms)
- t1 1D N=2^20 M=1e6 eps=1e-05: dominant stage fft_x at 54% (0.56 of 1.04 ms)
- t1 2D 512^2 M=1e6 eps=1e-03: dominant stage spread at 68% (0.55 of 0.81 ms)
- t1 2D 512^2 M=1e6 eps=1e-05: dominant stage spread at 68% (0.58 of 0.85 ms)
- t1 3D 128^3 M=1e6 eps=1e-03: dominant stage spread at 62% (1.66 of 2.68 ms)
- t1 3D 128^3 M=1e6 eps=1e-05: dominant stage spread at 62% (2.95 of 4.76 ms)
- t1 3D 256^3 M=1e6 eps=1e-03: dominant stage spread at 33% (3.42 of 10.28 ms)
- t1 3D 256^3 M=1e6 eps=1e-05: dominant stage spread at 45% (11.00 of 24.51 ms)
- t2 1D N=2^20 M=1e6 eps=1e-03: dominant stage fft_x at 50% (0.55 of 1.09 ms)
- t2 1D N=2^20 M=1e6 eps=1e-05: dominant stage fft_x at 50% (0.56 of 1.14 ms)
- t2 2D 512^2 M=1e6 eps=1e-03: dominant stage gather at 34% (0.23 of 0.66 ms)
- t2 2D 512^2 M=1e6 eps=1e-05: dominant stage gather at 40% (0.25 of 0.62 ms)
- t2 3D 128^3 M=1e6 eps=1e-03: dominant stage gather at 26% (0.42 of 1.58 ms)
- t2 3D 128^3 M=1e6 eps=1e-05: dominant stage gather at 30% (0.97 of 3.18 ms)
- t2 3D 256^3 M=1e6 eps=1e-03: dominant stage h2d at 25% (2.13 of 8.35 ms)
- t2 3D 256^3 M=1e6 eps=1e-05: dominant stage fft_z at 23% (4.14 of 18.19 ms)
- t3 3D generic N=512 P=1e5 M=2.6e5 eps=1e-03: dominant stage fft_x at 41% (24.57 of 60.41 ms)
- t3 3D generic N=512 P=1e5 M=2.6e5 eps=1e-05: dominant stage fft_x at 39% (24.63 of 63.60 ms)
- t3 1D X=S=300 P=1e6 M=1e6 eps=1e-03: dominant stage spread at 51% (7.53 of 14.69 ms)
- t3 1D X=S=300 P=1e6 M=1e6 eps=1e-05: dominant stage fft_x at 44% (8.93 of 20.20 ms)

Flagged (sum of stages vs whole execute differs by more than 15%): t1 1D N=2^20 M=1e6 eps=1e-03 (sum/whole 1.25: excess 0.27 ms vs 5 boundaries x floor = 1.29 ms); t1 1D N=2^20 M=1e6 eps=1e-05 (sum/whole 1.32: excess 0.33 ms vs 5 boundaries x floor = 1.29 ms); t1 2D 512^2 M=1e6 eps=1e-03 (sum/whole 1.77: excess 0.62 ms vs 7 boundaries x floor = 1.80 ms); t1 2D 512^2 M=1e6 eps=1e-05 (sum/whole 1.75: excess 0.64 ms vs 7 boundaries x floor = 1.80 ms); t1 3D 128^3 M=1e6 eps=1e-03 (sum/whole 1.33: excess 0.88 ms vs 9 boundaries x floor = 2.32 ms); t1 3D 128^3 M=1e6 eps=1e-05 (sum/whole 1.21: excess 0.99 ms vs 9 boundaries x floor = 2.32 ms); t2 1D N=2^20 M=1e6 eps=1e-03 (sum/whole 1.20: excess 0.22 ms vs 5 boundaries x floor = 1.29 ms); t2 1D N=2^20 M=1e6 eps=1e-05 (sum/whole 1.19: excess 0.21 ms vs 5 boundaries x floor = 1.29 ms); t2 2D 512^2 M=1e6 eps=1e-03 (sum/whole 1.71: excess 0.47 ms vs 7 boundaries x floor = 1.80 ms); t2 2D 512^2 M=1e6 eps=1e-05 (sum/whole 1.84: excess 0.52 ms vs 7 boundaries x floor = 1.80 ms); t2 3D 128^3 M=1e6 eps=1e-03 (sum/whole 1.47: excess 0.74 ms vs 9 boundaries x floor = 2.32 ms); t2 3D 128^3 M=1e6 eps=1e-05 (sum/whole 1.24: excess 0.78 ms vs 9 boundaries x floor = 2.32 ms)
