// VkFFT-Metal correctness: small known input, forward 2D FFT, read back,
// print first values to compare against numpy np.fft.fft2.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#include "vkFFT.h"
#include <cstdio>

int main() {
    const uint64_t n = 8, nb = 2, nE = n * n * nb;
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* queue = device->newCommandQueue();
    MTL::Buffer* buffer = device->newBuffer(nE * 8, MTL::ResourceStorageModeShared);
    float* fp = (float*)buffer->contents();
    for (uint64_t i = 0; i < nE; i++) {              // same input as numpy check
        fp[2*i]   = (float)(i % 5) - 2.0f;
        fp[2*i+1] = (float)((i*3) % 7) - 3.0f;
    }
    VkFFTConfiguration cfg = {};
    cfg.FFTdim = 2; cfg.size[0] = n; cfg.size[1] = n; cfg.numberBatches = nb;
    cfg.device = device; cfg.queue = queue;
    uint64_t bs = nE * 8; cfg.buffer = &buffer; cfg.bufferSize = &bs;
    VkFFTApplication app = {};
    if (initializeVkFFT(&app, cfg) != VKFFT_SUCCESS) { printf("init fail\n"); return 1; }
    VkFFTLaunchParams lp = {}; lp.buffer = &buffer;
    MTL::CommandBuffer* cb = queue->commandBuffer();
    lp.commandBuffer = cb;
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    lp.commandEncoder = enc;
    VkFFTAppend(&app, -1, &lp);                       // -1 = forward
    enc->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    printf("VkFFT fwd first 4 (slab0): ");
    for (int i = 0; i < 4; i++) printf("%.2f%+.2fi ", fp[2*i], fp[2*i+1]);
    printf("\n");
    deleteVkFFT(&app);
    return 0;
}
