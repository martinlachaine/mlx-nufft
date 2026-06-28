"""Optional VkFFT-Metal FFT backend for the type-3 slab pipeline.

Zero-copy bridge: an MLX array's data lives in 16 KB-aligned unified memory; we
hand its pointer (via dlpack) to a small C-ABI dylib that wraps it as an
`MTL::Buffer` (bytesNoCopy — no copy, no MLX C++ linkage) and runs VkFFT in
place. Measured 2.58× over MLX's four-step on the dominant slab FFT, → ~1.68×
whole-execute (see ACCEPTANCE / harness/fft_backend_bench).

Opt-in: requires building `vkfft_bridge/libvkfft_bridge.dylib`
(`vkfft_bridge/build.sh`, clones VkFFT + one clang++ call). MLX remains the
validated default backend; `fft_backend="vkfft"` raises a clear error if the
bridge is not built.

Conventions (validated vs mx.fft): the bridge does a batched 2D complex64 FFT
over the leading (batch) axis of a C-contiguous (nb, n_outer, n_contig) array.
`size[0]` is the contiguous axis, so we pass (n_contig, n_outer, nb). For the
MLX inverse-FFT convention (isign>0) use inverse=1, normalize=1; for the
forward convention (isign<0) use inverse=-1, normalize=0.
"""

import ctypes
import os
import pathlib

_lib = None
_load_error = None


def _candidate_paths():
    env = os.environ.get("MLX_NUFFT_VKFFT_LIB")
    if env:
        yield pathlib.Path(env)
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent
    yield root / "vkfft_bridge" / "libvkfft_bridge.dylib"
    yield here / "libvkfft_bridge.dylib"


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return
    for p in _candidate_paths():
        if p and p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.vkfft_fft2_inplace.restype = ctypes.c_int
                lib.vkfft_fft2_inplace.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
                _lib = lib
                return
            except Exception as e:                       # pragma: no cover
                _load_error = f"failed to load {p}: {e}"
                return
    _load_error = ("vkfft_bridge dylib not found; build it with "
                   "vkfft_bridge/build.sh or set MLX_NUFFT_VKFFT_LIB")


def available():
    _load()
    return _lib is not None


def require():
    _load()
    if _lib is None:
        raise RuntimeError(f"VkFFT backend unavailable: {_load_error}")


_GET = ctypes.pythonapi.PyCapsule_GetPointer
_GET.restype = ctypes.c_void_p
_GET.argtypes = [ctypes.py_object, ctypes.c_char_p]


def fft2_inplace(arr, inverse, normalize):
    """In-place batched 2D FFT of an MLX complex64 array of shape
    (nb, n_outer, n_contig). `arr` must be evaluated (mx.eval) first; it is
    mutated in place and also returned. inverse/normalize: see module docstring.
    """
    require()
    import mlx.core as mx
    assert isinstance(arr, mx.array) and arr.dtype == mx.complex64
    assert arr.ndim == 3, "expect (nb, n_outer, n_contig)"
    nb, n_outer, n_contig = (int(s) for s in arr.shape)
    cap = arr.__dlpack__()                 # MUST stay alive across the call
    ptr = ctypes.cast(_GET(cap, b"dltensor"),
                      ctypes.POINTER(ctypes.c_void_p))[0]
    rc = _lib.vkfft_fft2_inplace(ctypes.c_void_p(ptr), n_contig, n_outer, nb,
                                 int(inverse), int(normalize))
    del cap
    if rc != 0:
        raise RuntimeError(f"vkfft_fft2_inplace failed (rc={rc})")
    return arr
