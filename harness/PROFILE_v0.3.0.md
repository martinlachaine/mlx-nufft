# Per-stage execute profile: mlx-nufft 0.3.0

- machine: Apple M5 Max, 128 GB, macOS-27.0-arm64-arm-64bit-Mach-O
- mlx 0.31.2, python 3.13.12
- git 68f31b765f0a (perf/v0.3, dirty)
- run: 2026-09-25T20:58:32+00:00
- type-2 pad path: fused (nd.PAD_PATH)
- protocol: per measurement, the GPU is kept busy for 100 ms (matmul loop) so the clocks are at steady state, then 2 warm-up executes, then 7 timed executes; median and min reported; mx.synchronize() before and after every timed region. Stage times come from a second set of runs that re-issue execute()'s kernel sequence with an eval + synchronize after each stage; sum/whole is the sum of stage medians over the whole-execute median (the per-stage syncs cost pipelining, so sum/whole above 1 is expected; rows outside 1 +/- 0.15 are flagged). Stage percentages are of the whole-execute median. A rep more than 2x its median triggered one re-run of that measurement (noted per row). Measured stage-boundary floor: 0.198 ms per eval + synchronize; for flagged rows the Reading section puts the excess next to the cost of the stage boundaries at that floor.
- inputs: n_trans=1, complex64, fixed seed, default plan options (upsampfac auto for types 1/2: nd.py's policy takes sigma=1.25 on 3D grids of at least 2^15 modes at loose eps and 2.0 otherwise, recorded as sigma in each case's info, and MLX_NUFFT_UPSAMPFAC=2.0|1.25 forces one for A/B runs; 1.25 for type 3; spread_method auto, points_backend auto, MLX FFT, crit64). h2d/d2h are the input upload and result download that execute() performs.

## Type 1, 1D

| case | eps | whole med (ms) | h2d | spread | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|
| t1 1D N=2^20 M=1e6 | 1e-03 | 1.05 | 0.14 (14%) | 0.32 (30%) | 0.53 (51%) | 0.20 (19%) | 0.14 (13%) | 1.03 | 1.27 FLAG |
| t1 1D N=2^20 M=1e6 | 1e-05 | 0.98 | 0.14 (14%) | 0.32 (32%) | 0.53 (54%) | 0.19 (19%) | 0.14 (14%) | 0.96 | 1.34 FLAG |

## Type 1, 2D

| case | eps | whole med (ms) | h2d | spread | fft_y | crop_y | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 2D 512^2 M=1e6 | 1e-03 | 0.77 | 0.14 (18%) | 0.54 (70%) | 0.19 (24%) | 0.17 (21%) | 0.18 (23%) | 0.18 (23%) | 0.05 (7%) | 0.77 | 1.87 FLAG |
| t1 2D 512^2 M=1e6 | 1e-05 | 0.81 | 0.14 (17%) | 0.57 (71%) | 0.19 (23%) | 0.16 (19%) | 0.16 (20%) | 0.16 (20%) | 0.05 (6%) | 0.79 | 1.77 FLAG |

## Type 1, 3D

| case | eps | whole med (ms) | h2d | spread | fft_z | crop_z | fft_y | crop_y | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 3D 128^3 M=1e6 | 1e-03 | 2.59 | 0.14 (5%) | 1.65 (64%) | 0.29 (11%) | 0.24 (9%) | 0.24 (9%) | 0.22 (9%) | 0.24 (9%) | 0.24 (9%) | 0.26 (10%) | 2.57 | 1.36 FLAG |
| t1 3D 128^3 M=1e6 | 1e-05 | 4.66 | 0.16 (3%) | 2.95 (63%) | 0.71 (15%) | 0.42 (9%) | 0.41 (9%) | 0.28 (6%) | 0.27 (6%) | 0.24 (5%) | 0.27 (6%) | 4.63 | 1.22 FLAG |
| t1 3D 256^3 M=1e6 | 1e-03 | 10.20 | 0.17 (2%) | 3.37 (33%) | 1.21 (12%) | 1.01 (10%) | 0.96 (9%) | 0.80 (8%) | 0.79 (8%) | 0.98 (10%) | 2.06 (20%) | 9.67 | 1.11 |
| t1 3D 256^3 M=1e6 | 1e-05 | 24.19 | 0.15 (1%) | 10.94 (45%) | 4.19 (17%) | 2.21 (9%) | 2.16 (9%) | 1.20 (5%) | 1.18 (5%) | 0.97 (4%) | 2.03 (8%) | 23.96 | 1.03 |

## Type 2, 1D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|
| t2 1D N=2^20 M=1e6 | 1e-03 | 1.08 | 0.15 (14%) | 0.22 (21%) | 0.52 (48%) | 0.24 (22%) | 0.14 (13%) | 1.02 | 1.17 FLAG |
| t2 1D N=2^20 M=1e6 | 1e-05 | 1.06 | 0.15 (14%) | 0.22 (20%) | 0.53 (50%) | 0.24 (22%) | 0.14 (13%) | 1.04 | 1.20 FLAG |

## Type 2, 2D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | pad_y | fft_y | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t2 2D 512^2 M=1e6 | 1e-03 | 0.63 | 0.05 (8%) | 0.18 (29%) | 0.17 (27%) | 0.16 (26%) | 0.17 (27%) | 0.24 (38%) | 0.14 (22%) | 0.59 | 1.77 FLAG |
| t2 2D 512^2 M=1e6 | 1e-05 | 0.63 | 0.05 (8%) | 0.18 (29%) | 0.17 (26%) | 0.17 (27%) | 0.18 (28%) | 0.24 (37%) | 0.13 (21%) | 0.62 | 1.76 FLAG |

## Type 2, 3D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | pad_y | fft_y | pad_z | fft_z | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2 3D 128^3 M=1e6 | 1e-03 | 1.53 | 0.27 (18%) | 0.23 (15%) | 0.20 (13%) | 0.23 (15%) | 0.27 (18%) | 0.26 (17%) | 0.30 (19%) | 0.42 (28%) | 0.14 (9%) | 1.51 | 1.51 FLAG |
| t2 3D 128^3 M=1e6 | 1e-05 | 3.12 | 0.27 (9%) | 0.25 (8%) | 0.28 (9%) | 0.35 (11%) | 0.43 (14%) | 0.56 (18%) | 0.68 (22%) | 0.97 (31%) | 0.14 (5%) | 3.09 | 1.26 FLAG |
| t2 3D 256^3 M=1e6 | 1e-03 | 8.32 | 2.05 (25%) | 1.53 (18%) | 0.84 (10%) | 0.93 (11%) | 0.98 (12%) | 1.07 (13%) | 1.17 (14%) | 0.89 (11%) | 0.15 (2%) | 8.21 | 1.15 FLAG |
| t2 3D 256^3 M=1e6 | 1e-05 | 18.10 | 2.04 (11%) | 1.73 (10%) | 1.20 (7%) | 1.79 (10%) | 2.23 (12%) | 3.35 (18%) | 4.16 (23%) | 2.57 (14%) | 0.15 (1%) | 17.49 | 1.06 |

## Type 3, 1D/3D

| case | eps | whole med (ms) | h2d | spread | pad+deconv | fft_z | fft_y | fft_x | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-03 | 59.94 | 0.04 (0%) | 3.34 (6%) | 6.55 (11%) | 7.94 (13%) | 16.46 (27%) | 24.08 (40%) | 1.40 (2%) | 0.06 (0%) | 59.57 | 1.00 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-05 | 63.53 | 0.04 (0%) | 4.75 (7%) | 6.59 (10%) | 7.94 (12%) | 16.45 (26%) | 24.16 (38%) | 3.15 (5%) | 0.06 (0%) | 62.85 | 0.99 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-03 | 14.67 | 0.16 (1%) | 7.51 (51%) | 0.70 (5%) | 0.93 (6%) | 0.91 (6%) | 4.06 (28%) | 0.41 (3%) | 0.16 (1%) | 14.54 | 1.01 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-05 | 19.97 | 0.14 (1%) | 4.87 (24%) | 1.39 (7%) | 1.76 (9%) | 1.77 (9%) | 8.92 (45%) | 1.06 (5%) | 0.15 (1%) | 19.75 | 1.00 |

## Plan variants, build time and memory

staged vs whole rel-L2 compares the staged reproduction's output with execute()'s; nonzero values come from the run-to-run summation order of the atomic spreads, not from the staging.

| case | eps | w | n_up | pts/cell | path | plan build (s) | peak GiB | staged vs whole rel-L2 |
|---|---|---|---|---|---|---|---|---|
| t1 1D N=2^20 M=1e6 | 1e-03 | 4 | 2097152 | 0.477 | gm | 0.026 | 0.15 | 1.3e-07 |
| t1 1D N=2^20 M=1e6 | 1e-05 | 6 | 2097152 | 0.477 | od_ex | 0.017 | 0.14 | 1.3e-07 |
| t1 2D 512^2 M=1e6 | 1e-03 | 4 | 1024x1024 | 0.954 | gm | 0.005 | 0.07 | 1.6e-07 |
| t1 2D 512^2 M=1e6 | 1e-05 | 6 | 1024x1024 | 0.954 | od | 0.006 | 0.07 | 1.3e-07 |
| t1 3D 128^3 M=1e6 | 1e-03 | 5 | 160x160x160 | 0.244 | od | 0.009 | 0.20 | 5.6e-07 |
| t1 3D 128^3 M=1e6 | 1e-05 | 6 | 256x256x256 | 0.0596 | od | 0.008 | 0.49 | 1.3e-07 |
| t1 3D 256^3 M=1e6 | 1e-03 | 5 | 320x320x320 | 0.0305 | od | 0.009 | 1.35 | 4.5e-07 |
| t1 3D 256^3 M=1e6 | 1e-05 | 6 | 512x512x512 | 0.00745 | od | 0.012 | 3.66 | 1.2e-07 |
| t2 1D N=2^20 M=1e6 | 1e-03 | 4 | 2097152 | 0.477 | sorted_gather | 0.018 | 0.14 | 0.0e+00 |
| t2 1D N=2^20 M=1e6 | 1e-05 | 6 | 2097152 | 0.477 | sorted_gather | 0.014 | 0.14 | 0.0e+00 |
| t2 2D 512^2 M=1e6 | 1e-03 | 4 | 1024x1024 | 0.954 | sorted_gather | 0.005 | 0.07 | 0.0e+00 |
| t2 2D 512^2 M=1e6 | 1e-05 | 6 | 1024x1024 | 0.954 | sorted_gather | 0.005 | 0.07 | 0.0e+00 |
| t2 3D 128^3 M=1e6 | 1e-03 | 5 | 160x160x160 | 0.244 | sorted_gather | 0.007 | 0.20 | 0.0e+00 |
| t2 3D 128^3 M=1e6 | 1e-05 | 6 | 256x256x256 | 0.0596 | sorted_gather | 0.007 | 0.48 | 0.0e+00 |
| t2 3D 256^3 M=1e6 | 1e-03 | 5 | 320x320x320 | 0.0305 | sorted_gather | 0.007 | 1.34 | 0.0e+00 |
| t2 3D 256^3 M=1e6 | 1e-05 | 6 | 512x512x512 | 0.00745 | sorted_gather | 0.010 | 3.65 | 0.0e+00 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-03 | 5 | 640x640x640 | 0.000381 | od_spread/sorted_gather | 0.032 | 5.88 | 1.3e-07 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-05 | 8 | 640x640x640 | 0.000381 | od_spread/sorted_gather | 0.035 | 5.88 | 1.7e-05 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-03 | 5 | 90000x16x16 | 0.0434 | gm_spread/sorted_gather | 0.097 | 1.14 | 3.0e-07 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-05 | 9 | 90000x24x24 | 0.0193 | od_spread/sorted_gather | 0.089 | 2.42 | 1.2e-06 |

## Reading

- t1 1D N=2^20 M=1e6 eps=1e-03: dominant stage fft_x at 51% (0.53 of 1.05 ms)
- t1 1D N=2^20 M=1e6 eps=1e-05: dominant stage fft_x at 54% (0.53 of 0.98 ms)
- t1 2D 512^2 M=1e6 eps=1e-03: dominant stage spread at 70% (0.54 of 0.77 ms)
- t1 2D 512^2 M=1e6 eps=1e-05: dominant stage spread at 71% (0.57 of 0.81 ms)
- t1 3D 128^3 M=1e6 eps=1e-03: dominant stage spread at 64% (1.65 of 2.59 ms)
- t1 3D 128^3 M=1e6 eps=1e-05: dominant stage spread at 63% (2.95 of 4.66 ms)
- t1 3D 256^3 M=1e6 eps=1e-03: dominant stage spread at 33% (3.37 of 10.20 ms)
- t1 3D 256^3 M=1e6 eps=1e-05: dominant stage spread at 45% (10.94 of 24.19 ms)
- t2 1D N=2^20 M=1e6 eps=1e-03: dominant stage fft_x at 48% (0.52 of 1.08 ms)
- t2 1D N=2^20 M=1e6 eps=1e-05: dominant stage fft_x at 50% (0.53 of 1.06 ms)
- t2 2D 512^2 M=1e6 eps=1e-03: dominant stage gather at 38% (0.24 of 0.63 ms)
- t2 2D 512^2 M=1e6 eps=1e-05: dominant stage gather at 37% (0.24 of 0.63 ms)
- t2 3D 128^3 M=1e6 eps=1e-03: dominant stage gather at 28% (0.42 of 1.53 ms)
- t2 3D 128^3 M=1e6 eps=1e-05: dominant stage gather at 31% (0.97 of 3.12 ms)
- t2 3D 256^3 M=1e6 eps=1e-03: dominant stage h2d at 25% (2.05 of 8.32 ms)
- t2 3D 256^3 M=1e6 eps=1e-05: dominant stage fft_z at 23% (4.16 of 18.10 ms)
- t3 3D generic N=512 P=1e5 M=2.6e5 eps=1e-03: dominant stage fft_x at 40% (24.08 of 59.94 ms)
- t3 3D generic N=512 P=1e5 M=2.6e5 eps=1e-05: dominant stage fft_x at 38% (24.16 of 63.53 ms)
- t3 1D X=S=300 P=1e6 M=1e6 eps=1e-03: dominant stage spread at 51% (7.51 of 14.67 ms)
- t3 1D X=S=300 P=1e6 M=1e6 eps=1e-05: dominant stage fft_x at 45% (8.92 of 19.97 ms)

Flagged (sum of stages vs whole execute differs by more than 15%): t1 1D N=2^20 M=1e6 eps=1e-03 (sum/whole 1.27: excess 0.28 ms vs 5 boundaries x floor = 0.99 ms); t1 1D N=2^20 M=1e6 eps=1e-05 (sum/whole 1.34: excess 0.33 ms vs 5 boundaries x floor = 0.99 ms); t1 2D 512^2 M=1e6 eps=1e-03 (sum/whole 1.87: excess 0.67 ms vs 7 boundaries x floor = 1.39 ms); t1 2D 512^2 M=1e6 eps=1e-05 (sum/whole 1.77: excess 0.63 ms vs 7 boundaries x floor = 1.39 ms); t1 3D 128^3 M=1e6 eps=1e-03 (sum/whole 1.36: excess 0.94 ms vs 9 boundaries x floor = 1.78 ms); t1 3D 128^3 M=1e6 eps=1e-05 (sum/whole 1.22: excess 1.04 ms vs 9 boundaries x floor = 1.78 ms); t2 1D N=2^20 M=1e6 eps=1e-03 (sum/whole 1.17: excess 0.18 ms vs 5 boundaries x floor = 0.99 ms); t2 1D N=2^20 M=1e6 eps=1e-05 (sum/whole 1.20: excess 0.21 ms vs 5 boundaries x floor = 0.99 ms); t2 2D 512^2 M=1e6 eps=1e-03 (sum/whole 1.77: excess 0.48 ms vs 7 boundaries x floor = 1.39 ms); t2 2D 512^2 M=1e6 eps=1e-05 (sum/whole 1.76: excess 0.48 ms vs 7 boundaries x floor = 1.39 ms); t2 3D 128^3 M=1e6 eps=1e-03 (sum/whole 1.51: excess 0.78 ms vs 9 boundaries x floor = 1.78 ms); t2 3D 128^3 M=1e6 eps=1e-05 (sum/whole 1.26: excess 0.82 ms vs 9 boundaries x floor = 1.78 ms); t2 3D 256^3 M=1e6 eps=1e-03 (sum/whole 1.15: excess 1.28 ms vs 9 boundaries x floor = 1.78 ms)
