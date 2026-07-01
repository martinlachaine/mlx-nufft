// C-ABI VkFFT-Metal bridge: batched 1D/2D/3D complex64 FFTs in-place on a raw
// unified-memory pointer (an MLX array's 16 KB-aligned data). Wraps it as an
// MTL::Buffer via bytesNoCopy (no copy, no MLX linkage); driven via ctypes.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#include "vkFFT.h"
#include <map>
#include <tuple>

namespace {
MTL::Device* g_dev = nullptr;
MTL::CommandQueue* g_queue = nullptr;
void ensure_device() {
    if (!g_dev) { g_dev = MTL::CreateSystemDefaultDevice();
                  g_queue = g_dev->newCommandQueue(); }
}
// Cache one VkFFTApplication per (dim,n1,n2,n3,nb,norm). VkFFT plans (kernels)
// are data-buffer-independent; the actual buffer is supplied per call via
// VkFFTLaunchParams.buffer. Direction is a launch-time argument (VkFFT builds
// forward+inverse kernels in one app), so it is not part of the key; the
// normalization flag is baked at plan time, so it is.
struct Key { int dim; uint64_t n1,n2,n3,nb; int norm;
    bool operator<(const Key&o) const {
        return std::tie(dim,n1,n2,n3,nb,norm)
             < std::tie(o.dim,o.n1,o.n2,o.n3,o.nb,o.norm);} };
struct Entry { VkFFTApplication* app; uint64_t bytes; };
std::map<Key,Entry> g_cache;

Entry* get_entry(int dim, uint64_t n1, uint64_t n2, uint64_t n3,
                 uint64_t nb, int norm) {
    Key k{dim,n1,n2,n3,nb,norm};
    auto it=g_cache.find(k);
    if (it!=g_cache.end()) return &it->second;
    // Insert first, then fill via a reference into the map's PERMANENT storage,
    // so the pointer handed to VkFFT (&e.bytes) never dangles. The data buffer
    // itself is launch-only (lp.buffer): planning needs bufferSize, not the
    // buffer, so no persistent plan buffer is allocated.
    Entry& e = g_cache[k];
    e.bytes=(uint64_t)8*n1*n2*n3*nb;
    VkFFTConfiguration cfg={};
    cfg.FFTdim=dim; cfg.size[0]=n1;
    if (dim>=2) cfg.size[1]=n2;
    if (dim>=3) cfg.size[2]=n3;
    cfg.numberBatches=nb;
    cfg.device=g_dev; cfg.queue=g_queue; cfg.normalize=(norm?1:0);
    cfg.bufferSize=&e.bytes;                          // point into map storage (stable)
    // Apple-measured (M5 Max): a 32-byte coalescing granule beats VkFFT's
    // Metal default of 64 on strided (multi-axis) passes — 640^3 c2c
    // 50.7 -> 37.1 ms, 512^3 24.0 -> 19.7 ms — and is neutral on the
    // batched-2D slab shapes (55.7 -> 54.8 ms). 8/16 regress the slab path.
    cfg.coalescedMemory = 32;
    e.app=new VkFFTApplication(); *e.app=VkFFTApplication{};
    if(initializeVkFFT(e.app,cfg)!=VKFFT_SUCCESS){ delete e.app; g_cache.erase(k); return nullptr; }
    return &e;
}

int run_fft(void* dataptr, int dim, uint64_t n1, uint64_t n2, uint64_t n3,
            uint64_t nb, int inverse, int normalize) {
    ensure_device();
    uint64_t bytes=(uint64_t)8*n1*n2*n3*nb;
    MTL::ResourceOptions opts = MTL::ResourceStorageModeShared
                              | MTL::ResourceHazardTrackingModeUntracked;
    MTL::Buffer* buf=g_dev->newBuffer(dataptr, bytes, opts, nullptr);
    if(!buf) return -1;
    Entry* e=get_entry(dim,n1,n2,n3,nb,normalize);
    if(!e){ buf->release(); return -2; }
    VkFFTLaunchParams lp={};
    lp.buffer=&buf;                                   // MLX's buffer, launch-bound
    MTL::CommandBuffer* cb=g_queue->commandBuffer();
    lp.commandBuffer=cb;
    MTL::ComputeCommandEncoder* enc=cb->computeCommandEncoder();
    lp.commandEncoder=enc;
    VkFFTResult r=VkFFTAppend(e->app, inverse, &lp);
    enc->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    buf->release();                                   // wrapper only (nil deallocator => MLX memory kept)
    return (r==VKFFT_SUCCESS)?0:-3;
}
} // namespace

extern "C" int vkfft_fft2_inplace(void* dataptr, uint64_t n1, uint64_t n2,
                                  uint64_t nb, int inverse, int normalize) {
    return run_fft(dataptr, 2, n1, n2, 1, nb, inverse, normalize);
}

// n1 is the contiguous axis, n2/n3 outer (pass 1 for unused dims), nb batches
// over the outermost axis. fftdim in {1,2,3}.
extern "C" int vkfft_fftn_inplace(void* dataptr, uint64_t n1, uint64_t n2,
                                  uint64_t n3, uint64_t nb, int fftdim,
                                  int inverse, int normalize) {
    if (fftdim < 1 || fftdim > 3) return -4;
    return run_fft(dataptr, fftdim, n1, n2, n3, nb, inverse, normalize);
}
