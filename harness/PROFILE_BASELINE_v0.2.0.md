# Per-stage execute profile: mlx-nufft 0.2.0

- machine: Apple M5 Max, 128 GB, macOS-27.0-arm64-arm-64bit-Mach-O
- mlx 0.31.2, python 3.13.12
- git b0621d984429 (worktree-agent-a596c875d9385a28c)
- run: 2026-09-25T18:18:35+00:00
- protocol: per measurement, the GPU is kept busy for 100 ms (matmul loop) so the clocks are at steady state, then 2 warm-up executes, then 7 timed executes; median and min reported; mx.synchronize() before and after every timed region. Stage times come from a second set of runs that re-issue execute()'s kernel sequence with an eval + synchronize after each stage; sum/whole is the sum of stage medians over the whole-execute median (the per-stage syncs cost pipelining, so sum/whole above 1 is expected; rows outside 1 +/- 0.15 are flagged). Stage percentages are of the whole-execute median. A rep more than 2x its median triggered one re-run of that measurement (noted per row). Measured stage-boundary floor: 0.198 ms per eval + synchronize; for flagged rows the Reading section puts the excess next to the cost of the stage boundaries at that floor.
- inputs: n_trans=1, complex64, fixed seed, default plan options (upsampfac 2.0 for types 1/2 and 1.25 for type 3, spread_method auto, points_backend auto, MLX FFT, crit64). h2d/d2h are the input upload and result download that execute() performs.

## Type 1, 1D

| case | eps | whole med (ms) | h2d | spread | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|
| t1 1D N=2^20 M=1e6 | 1e-03 | 1.05 | 0.14 (13%) | 0.32 (31%) | 0.54 (52%) | 0.19 (18%) | 0.13 (12%) | 1.04 | 1.27 FLAG |
| t1 1D N=2^20 M=1e6 | 1e-05 | 0.99 | 0.14 (14%) | 0.31 (31%) | 0.53 (54%) | 0.20 (20%) | 0.13 (13%) | 0.97 | 1.33 FLAG |

## Type 1, 2D

| case | eps | whole med (ms) | h2d | spread | fft_y | crop_y | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 2D 512^2 M=1e6 | 1e-03 | 0.83 | 0.14 (17%) | 0.55 (66%) | 0.19 (23%) | 0.16 (19%) | 0.18 (22%) | 0.18 (22%) | 0.05 (6%) | 0.82 | 1.75 FLAG |
| t1 2D 512^2 M=1e6 | 1e-05 | 0.83 | 0.14 (17%) | 0.59 (70%) | 0.20 (24%) | 0.16 (19%) | 0.17 (20%) | 0.17 (20%) | 0.05 (6%) | 0.81 | 1.77 FLAG |

## Type 1, 3D

| case | eps | whole med (ms) | h2d | spread | fft_z | crop_z | fft_y | crop_y | fft_x | crop_x+deconv | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 3D 128^3 M=1e6 | 1e-03 | 5.56 | 0.15 (3%) | 3.75 (67%) | 0.74 (13%) | 0.41 (7%) | 0.44 (8%) | 0.28 (5%) | 0.28 (5%) | 0.28 (5%) | 0.26 (5%) | 5.49 | 1.18 FLAG |
| t1 3D 128^3 M=1e6 | 1e-05 | 4.69 | 0.15 (3%) | 2.95 (63%) | 0.72 (15%) | 0.41 (9%) | 0.41 (9%) | 0.28 (6%) | 0.29 (6%) | 0.22 (5%) | 0.25 (5%) | 4.68 | 1.21 FLAG |
| t1 3D 256^3 M=1e6 | 1e-03 | 20.42 | 0.16 (1%) | 7.23 (35%) | 4.20 (21%) | 2.24 (11%) | 2.18 (11%) | 1.21 (6%) | 1.18 (6%) | 0.99 (5%) | 1.98 (10%) | 19.95 | 1.05 |
| t1 3D 256^3 M=1e6 | 1e-05 | 24.28 | 0.16 (1%) | 10.69 (44%) | 4.25 (17%) | 2.25 (9%) | 2.20 (9%) | 1.19 (5%) | 1.17 (5%) | 1.03 (4%) | 2.01 (8%) | 23.60 | 1.03 |

## Type 2, 1D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|
| t2 1D N=2^20 M=1e6 | 1e-03 | 1.11 | 0.14 (13%) | 0.22 (20%) | 0.52 (47%) | 0.25 (22%) | 0.13 (12%) | 1.08 | 1.14 |
| t2 1D N=2^20 M=1e6 | 1e-05 | 1.08 | 0.14 (13%) | 0.22 (21%) | 0.52 (48%) | 0.25 (23%) | 0.13 (12%) | 1.06 | 1.16 FLAG |

## Type 2, 2D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | pad_y | fft_y | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t2 2D 512^2 M=1e6 | 1e-03 | 0.64 | 0.05 (8%) | 0.19 (30%) | 0.17 (27%) | 0.18 (29%) | 0.18 (28%) | 0.23 (37%) | 0.13 (20%) | 0.62 | 1.79 FLAG |
| t2 2D 512^2 M=1e6 | 1e-05 | 0.64 | 0.05 (8%) | 0.19 (30%) | 0.16 (25%) | 0.19 (29%) | 0.17 (27%) | 0.23 (36%) | 0.13 (20%) | 0.62 | 1.74 FLAG |

## Type 2, 3D

| case | eps | whole med (ms) | h2d | pad_x+deconv | fft_x | pad_y | fft_y | pad_z | fft_z | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2 3D 128^3 M=1e6 | 1e-03 | 3.06 | 0.26 (9%) | 0.27 (9%) | 0.28 (9%) | 0.48 (16%) | 0.41 (14%) | 0.83 (27%) | 0.67 (22%) | 0.54 (18%) | 0.14 (5%) | 3.03 | 1.27 FLAG |
| t2 3D 128^3 M=1e6 | 1e-05 | 3.51 | 0.27 (8%) | 0.28 (8%) | 0.28 (8%) | 0.49 (14%) | 0.42 (12%) | 0.80 (23%) | 0.66 (19%) | 0.96 (27%) | 0.14 (4%) | 3.46 | 1.23 FLAG |
| t2 3D 256^3 M=1e6 | 1e-03 | 20.95 | 1.98 (9%) | 1.55 (7%) | 1.17 (6%) | 2.77 (13%) | 2.17 (10%) | 5.33 (25%) | 4.17 (20%) | 2.17 (10%) | 0.16 (1%) | 20.56 | 1.03 |
| t2 3D 256^3 M=1e6 | 1e-05 | 21.63 | 1.99 (9%) | 1.51 (7%) | 1.17 (5%) | 2.75 (13%) | 2.14 (10%) | 5.28 (24%) | 4.14 (19%) | 2.68 (12%) | 0.16 (1%) | 21.12 | 1.01 |

## Type 3, 1D/3D

| case | eps | whole med (ms) | h2d | spread | pad+deconv | fft_z | fft_y | fft_x | gather | d2h | whole min (ms) | sum/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-03 | 65.03 | 0.05 (0%) | 3.32 (5%) | 6.64 (10%) | 7.93 (12%) | 16.24 (25%) | 19.23 (30%) | 11.64 (18%) | 0.07 (0%) | 64.66 | 1.00 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-05 | 69.96 | 0.05 (0%) | 5.47 (8%) | 6.79 (10%) | 7.96 (11%) | 16.32 (23%) | 19.18 (27%) | 14.29 (20%) | 0.07 (0%) | 69.43 | 1.00 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-03 | 15.85 | 0.15 (1%) | 7.46 (47%) | 0.66 (4%) | 0.87 (6%) | 1.54 (10%) | 4.81 (30%) | 0.42 (3%) | 0.15 (1%) | 15.80 | 1.01 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-05 | 23.31 | 0.15 (1%) | 4.84 (21%) | 1.37 (6%) | 1.74 (7%) | 3.39 (15%) | 10.71 (46%) | 1.05 (5%) | 0.16 (1%) | 23.17 | 1.00 |

## Plan variants, build time and memory

staged vs whole rel-L2 compares the staged reproduction's output with execute()'s; nonzero values come from the run-to-run summation order of the atomic spreads, not from the staging.

| case | eps | w | n_up | pts/cell | path | plan build (s) | peak GiB | staged vs whole rel-L2 |
|---|---|---|---|---|---|---|---|---|
| t1 1D N=2^20 M=1e6 | 1e-03 | 4 | 2097152 | 0.477 | gm | 0.392 | 0.15 | 1.3e-07 |
| t1 1D N=2^20 M=1e6 | 1e-05 | 6 | 2097152 | 0.477 | od_ex | 0.274 | 0.14 | 1.3e-07 |
| t1 2D 512^2 M=1e6 | 1e-03 | 4 | 1024x1024 | 0.954 | gm | 0.007 | 0.07 | 1.6e-07 |
| t1 2D 512^2 M=1e6 | 1e-05 | 6 | 1024x1024 | 0.954 | od | 0.007 | 0.07 | 1.3e-07 |
| t1 3D 128^3 M=1e6 | 1e-03 | 4 | 256x256x256 | 0.0596 | gm | 0.008 | 0.49 | 1.3e-07 |
| t1 3D 128^3 M=1e6 | 1e-05 | 6 | 256x256x256 | 0.0596 | od | 0.009 | 0.49 | 1.3e-07 |
| t1 3D 256^3 M=1e6 | 1e-03 | 4 | 512x512x512 | 0.00745 | gm | 0.007 | 3.67 | 1.1e-07 |
| t1 3D 256^3 M=1e6 | 1e-05 | 6 | 512x512x512 | 0.00745 | od | 0.017 | 3.66 | 1.2e-07 |
| t2 1D N=2^20 M=1e6 | 1e-03 | 4 | 2097152 | 0.477 | sorted_gather | 0.261 | 0.14 | 0.0e+00 |
| t2 1D N=2^20 M=1e6 | 1e-05 | 6 | 2097152 | 0.477 | sorted_gather | 0.270 | 0.14 | 0.0e+00 |
| t2 2D 512^2 M=1e6 | 1e-03 | 4 | 1024x1024 | 0.954 | sorted_gather | 0.006 | 0.07 | 0.0e+00 |
| t2 2D 512^2 M=1e6 | 1e-05 | 6 | 1024x1024 | 0.954 | sorted_gather | 0.005 | 0.07 | 0.0e+00 |
| t2 3D 128^3 M=1e6 | 1e-03 | 4 | 256x256x256 | 0.0596 | sorted_gather | 0.007 | 0.57 | 0.0e+00 |
| t2 3D 128^3 M=1e6 | 1e-05 | 6 | 256x256x256 | 0.0596 | sorted_gather | 0.008 | 0.57 | 0.0e+00 |
| t2 3D 256^3 M=1e6 | 1e-03 | 4 | 512x512x512 | 0.00745 | sorted_gather | 0.008 | 4.40 | 0.0e+00 |
| t2 3D 256^3 M=1e6 | 1e-05 | 6 | 512x512x512 | 0.00745 | sorted_gather | 0.007 | 4.40 | 0.0e+00 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-03 | 5 | 640x640x640 | 0.000381 | od_spread/sorted_gather | 0.040 | 5.88 | 1.2e-07 |
| t3 3D generic N=512 P=1e5 M=2.6e5 | 1e-05 | 9 | 640x640x640 | 0.000381 | od_spread/sorted_gather | 0.036 | 5.88 | 8.0e-05 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-03 | 5 | 90000x16x16 | 0.0434 | gm_spread/sorted_gather | 0.096 | 1.14 | 3.0e-07 |
| t3 1D X=S=300 P=1e6 M=1e6 | 1e-05 | 9 | 90000x24x24 | 0.0193 | od_spread/sorted_gather | 0.083 | 2.42 | 1.2e-06 |

## Reading

- t1 1D N=2^20 M=1e6 eps=1e-03: dominant stage fft_x at 52% (0.54 of 1.05 ms)
- t1 1D N=2^20 M=1e6 eps=1e-05: dominant stage fft_x at 54% (0.53 of 0.99 ms)
- t1 2D 512^2 M=1e6 eps=1e-03: dominant stage spread at 66% (0.55 of 0.83 ms)
- t1 2D 512^2 M=1e6 eps=1e-05: dominant stage spread at 70% (0.59 of 0.83 ms)
- t1 3D 128^3 M=1e6 eps=1e-03: dominant stage spread at 67% (3.75 of 5.56 ms)
- t1 3D 128^3 M=1e6 eps=1e-05: dominant stage spread at 63% (2.95 of 4.69 ms)
- t1 3D 256^3 M=1e6 eps=1e-03: dominant stage spread at 35% (7.23 of 20.42 ms)
- t1 3D 256^3 M=1e6 eps=1e-05: dominant stage spread at 44% (10.69 of 24.28 ms)
- t2 1D N=2^20 M=1e6 eps=1e-03: dominant stage fft_x at 47% (0.52 of 1.11 ms)
- t2 1D N=2^20 M=1e6 eps=1e-05: dominant stage fft_x at 48% (0.52 of 1.08 ms)
- t2 2D 512^2 M=1e6 eps=1e-03: dominant stage gather at 37% (0.23 of 0.64 ms)
- t2 2D 512^2 M=1e6 eps=1e-05: dominant stage gather at 36% (0.23 of 0.64 ms)
- t2 3D 128^3 M=1e6 eps=1e-03: dominant stage pad_z at 27% (0.83 of 3.06 ms)
- t2 3D 128^3 M=1e6 eps=1e-05: dominant stage gather at 27% (0.96 of 3.51 ms)
- t2 3D 256^3 M=1e6 eps=1e-03: dominant stage pad_z at 25% (5.33 of 20.95 ms)
- t2 3D 256^3 M=1e6 eps=1e-05: dominant stage pad_z at 24% (5.28 of 21.63 ms)
- t3 3D generic N=512 P=1e5 M=2.6e5 eps=1e-03: dominant stage fft_x at 30% (19.23 of 65.03 ms)
- t3 3D generic N=512 P=1e5 M=2.6e5 eps=1e-05: dominant stage fft_x at 27% (19.18 of 69.96 ms)
- t3 1D X=S=300 P=1e6 M=1e6 eps=1e-03: dominant stage spread at 47% (7.46 of 15.85 ms)
- t3 1D X=S=300 P=1e6 M=1e6 eps=1e-05: dominant stage fft_x at 46% (10.71 of 23.31 ms)

Flagged (sum of stages vs whole execute differs by more than 15%): t1 1D N=2^20 M=1e6 eps=1e-03 (sum/whole 1.27: excess 0.28 ms vs 5 boundaries x floor = 0.99 ms); t1 1D N=2^20 M=1e6 eps=1e-05 (sum/whole 1.33: excess 0.32 ms vs 5 boundaries x floor = 0.99 ms); t1 2D 512^2 M=1e6 eps=1e-03 (sum/whole 1.75: excess 0.62 ms vs 7 boundaries x floor = 1.39 ms); t1 2D 512^2 M=1e6 eps=1e-05 (sum/whole 1.77: excess 0.64 ms vs 7 boundaries x floor = 1.39 ms); t1 3D 128^3 M=1e6 eps=1e-03 (sum/whole 1.18: excess 1.03 ms vs 9 boundaries x floor = 1.78 ms); t1 3D 128^3 M=1e6 eps=1e-05 (sum/whole 1.21: excess 0.98 ms vs 9 boundaries x floor = 1.78 ms); t2 1D N=2^20 M=1e6 eps=1e-05 (sum/whole 1.16: excess 0.17 ms vs 5 boundaries x floor = 0.99 ms); t2 2D 512^2 M=1e6 eps=1e-03 (sum/whole 1.79: excess 0.50 ms vs 7 boundaries x floor = 1.39 ms); t2 2D 512^2 M=1e6 eps=1e-05 (sum/whole 1.74: excess 0.47 ms vs 7 boundaries x floor = 1.39 ms); t2 3D 128^3 M=1e6 eps=1e-03 (sum/whole 1.27: excess 0.82 ms vs 9 boundaries x floor = 1.78 ms); t2 3D 128^3 M=1e6 eps=1e-05 (sum/whole 1.23: excess 0.80 ms vs 9 boundaries x floor = 1.78 ms)
