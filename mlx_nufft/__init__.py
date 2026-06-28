"""mlx-nufft: non-uniform FFTs on Apple GPUs (Metal, via MLX).

Drop-in mirror of the `finufft` Python package's interface:

    import mlx_nufft as finufft
    fk = finufft.nufft2d1(x, y, c, (N1, N2), eps=1e-6)   # types 1/2/3, dims 1/2/3
    plan = finufft.Plan(1, (N1, N2), eps=1e-6)
    plan.setpts(x, y)
    fk = plan.execute(c)

plus the native plan classes:

    from mlx_nufft import Type3Plan
    plan = Type3Plan((x1, x2, x3), (s1, s2, s3), eps=1e-5, isign=+1)
    f = plan.execute(c)          # f[k] = sum_j c[j] exp(i*isign*s_k.x_j)

Precision model (see REPORT.md): fp32 GPU pipeline with the precision-
critical setup (coordinate rescale, pre/post phases) in fp64 at plan time
('crit64', the default). Plans cache all geometry-dependent state, so
fixed-geometry workloads pay setup once and execute() per call.
"""

from .gpu_t3 import GpuT3Plan as Type3Plan
from .types12 import Type1Plan, Type2Plan
from .nd import Type1PlanND, Type2PlanND
from .dfmath import expi, EXPI_MAX_PHASE
from .sizing import kernel_params, next235even
from .api import (Plan,
                  nufft1d1, nufft1d2, nufft1d3,
                  nufft2d1, nufft2d2, nufft2d3,
                  nufft3d1, nufft3d2, nufft3d3)


def vkfft_available():
    """True if the optional VkFFT-Metal FFT backend (fft_backend='vkfft') is
    built and loadable — see vkfft_bridge/build.sh."""
    from . import vkfft_backend
    return vkfft_backend.available()


__version__ = "0.1.0"
__all__ = ["Type3Plan", "Type1Plan", "Type2Plan",
           "Type1PlanND", "Type2PlanND", "Plan",
           "nufft1d1", "nufft1d2", "nufft1d3",
           "nufft2d1", "nufft2d2", "nufft2d3",
           "nufft3d1", "nufft3d2", "nufft3d3",
           "kernel_params", "next235even", "vkfft_available",
           "expi", "EXPI_MAX_PHASE"]
