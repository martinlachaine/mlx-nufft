// C-ABI VkFFT-Metal bridge: batched 2D complex64 FFT in-place on a raw
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
// Cache one VkFFTApplication per (n1,n2,nb,norm). VkFFT plans (kernels) are
// data-buffer-independent; the actual buffer is supplied per call via
// VkFFTLaunchParams.buffer. The app is initialized with a small persistent
// "plan" buffer of the right size so configuration.buffer never dangles.
struct Key { uint64_t n1,n2,nb; int norm;
    bool operator<(const Key&o) const {
        return std::tie(n1,n2,nb,norm)<std::tie(o.n1,o.n2,o.nb,o.norm);} };
struct Entry { VkFFTApplication* app; MTL::Buffer* planBuf; uint64_t bytes; };
std::map<Key,Entry> g_cache;

Entry* get_entry(uint64_t n1,uint64_t n2,uint64_t nb,int norm) {
    Key k{n1,n2,nb,norm};
    auto it=g_cache.find(k);
    if (it!=g_cache.end()) return &it->second;
    // Insert first, then fill via a reference into the map's PERMANENT storage,
    // so the pointers handed to VkFFT (&e.planBuf, &e.bytes) never dangle.
    Entry& e = g_cache[k];
    e.bytes=(uint64_t)8*n1*n2*nb;
    e.planBuf=g_dev->newBuffer(e.bytes, MTL::ResourceStorageModePrivate);
    if(!e.planBuf){ g_cache.erase(k); return nullptr; }
    VkFFTConfiguration cfg={};
    cfg.FFTdim=2; cfg.size[0]=n1; cfg.size[1]=n2; cfg.numberBatches=nb;
    cfg.device=g_dev; cfg.queue=g_queue; cfg.normalize=(norm?1:0);
    cfg.buffer=&e.planBuf; cfg.bufferSize=&e.bytes;   // point into map storage (stable)
    e.app=new VkFFTApplication(); *e.app=VkFFTApplication{};
    if(initializeVkFFT(e.app,cfg)!=VKFFT_SUCCESS){ delete e.app; e.planBuf->release(); g_cache.erase(k); return nullptr; }
    return &e;
}
} // namespace

extern "C" int vkfft_fft2_inplace(void* dataptr, uint64_t n1, uint64_t n2,
                                  uint64_t nb, int inverse, int normalize) {
    ensure_device();
    uint64_t bytes=(uint64_t)8*n1*n2*nb;
    MTL::ResourceOptions opts = MTL::ResourceStorageModeShared
                              | MTL::ResourceHazardTrackingModeUntracked;
    MTL::Buffer* buf=g_dev->newBuffer(dataptr, bytes, opts, nullptr);
    if(!buf) return -1;
    Entry* e=get_entry(n1,n2,nb,normalize);
    if(!e){ buf->release(); return -2; }
    VkFFTLaunchParams lp={};
    lp.buffer=&buf;                                   // override the plan buffer with MLX's
    MTL::CommandBuffer* cb=g_queue->commandBuffer();
    lp.commandBuffer=cb;
    MTL::ComputeCommandEncoder* enc=cb->computeCommandEncoder();
    lp.commandEncoder=enc;
    VkFFTResult r=VkFFTAppend(e->app, inverse, &lp);
    enc->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    buf->release();                                   // wrapper only (nil deallocator => MLX memory kept)
    return (r==VKFFT_SUCCESS)?0:-3;
}
