"""MLX baseline for the FFT-backend screen (WS1). Times the EXACT slab-FFT the
type-3 pipeline runs: per z-slab, fft_axis_scrambled(axis1)+fft_axis(axis0)
four-step, over nu3 slabs of a (nu3,nu1,nu2) complex64 grid. This is the number
a candidate backend (MPSGraph/VkFFT) must beat by K>=2 to justify wiring.

Also tries mx.fft.fft2 to show why the four-step exists (Bluestein blowup).
"""
import sys, time
sys.path.insert(0, "..")
import numpy as np
import mlx.core as mx
from mlx_nufft.gpu_t3 import fft_axis, fft_axis_scrambled

def tmin(fn, n=5):
    fn(); mx.synchronize()
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); fn(); mx.synchronize()
        ts.append(time.perf_counter()-t0)
    return min(ts)*1000

for (nu1, nu2, nu3) in [(7200,7200,24), (9000,9000,24), (9600,9600,24)]:
    Zc = mx.random.normal((nu3, nu1, nu2)).astype(mx.complex64); mx.eval(Zc)
    tw = {}
    def slabfft():
        # process+eval each slab then discard (as the real pipeline gathers and
        # frees per slab) — never hold all nu3 transformed planes at once.
        last = None
        for kz in range(nu3):
            vk = Zc[kz]
            vk, _ = fft_axis_scrambled(vk, 1, inverse=True, twiddle_cache=tw)
            vk = fft_axis(vk, 0, inverse=True, twiddle_cache=tw)
            mx.eval(vk)
            last = vk
            del vk
            mx.clear_cache()
        return last
    try:
        t = tmin(slabfft)
        gb = nu3*nu1*nu2*8/2**30
        print(f"MLX four-step slab FFT  ({nu3}x{nu1}x{nu2}): {t:.1f} ms  "
              f"({t/nu3:.2f} ms/slab, grid {gb:.1f} GiB c64)")
    except Exception as e:
        print(f"MLX four-step ({nu1}): FAIL {type(e).__name__}: {str(e)[:80]}")
    # show why four-step exists: native fft2 Bluestein blowup at non-pow2 >4096
    if nu1 == 7200:
        try:
            r = mx.fft.fftn(Zc, axes=[1,2]); mx.eval(r)
            print("  mx.fft.fftn(axes=1,2): ran (unexpected)")
        except Exception as e:
            print(f"  mx.fft.fftn(axes=1,2): {type(e).__name__}: {str(e)[:70]} (why four-step exists)")
    del Zc; mx.clear_cache()
