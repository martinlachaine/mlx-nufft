// VkFFT (Metal backend) bench — candidate vs MLX four-step / MPSGraph (WS1).
// One batched 2D complex64 FFT over numberBatches z-slabs of (n,n) — the same
// work the slab pipeline's FFT stage does. Args: n numberBatches reps.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#include "vkFFT.h"
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    uint64_t n = (argc > 1) ? strtoull(argv[1], 0, 10) : 7200;
    uint64_t nb = (argc > 2) ? strtoull(argv[2], 0, 10) : 24;
    int reps = (argc > 3) ? atoi(argv[3]) : 8;

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { printf("no Metal device\n"); return 1; }
    MTL::CommandQueue* queue = device->newCommandQueue();

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 2;
    configuration.size[0] = n;
    configuration.size[1] = n;
    configuration.numberBatches = nb;
    configuration.device = device;
    configuration.queue = queue;

    uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * n * n * nb;
    MTL::Buffer* buffer = device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
    if (!buffer) { printf("buffer alloc failed (%.2f GB)\n", bufferSize/1e9); return 1; }
    configuration.buffer = &buffer;
    configuration.bufferSize = &bufferSize;

    VkFFTApplication app = {};
    VkFFTResult res = initializeVkFFT(&app, configuration);
    if (res != VKFFT_SUCCESS) { printf("initializeVkFFT failed: %d\n", res); return 1; }

    VkFFTLaunchParams lp = {};
    lp.buffer = &buffer;

    auto one = [&](){
        MTL::CommandBuffer* cb = queue->commandBuffer();
        lp.commandBuffer = cb;
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        lp.commandEncoder = enc;
        VkFFTAppend(&app, -1, &lp);      // -1 = forward; timing is direction-agnostic
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    };

    for (int i = 0; i < 3; i++) one();      // warm
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        one();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
    }
    printf("VkFFT-Metal fft2 (%llux%llux%llu): %.1f ms (%.2f ms/slab, grid %.1f GiB c64)\n",
           (unsigned long long)nb, (unsigned long long)n, (unsigned long long)n,
           best, best / (double)nb, bufferSize / 1073741824.0);
    deleteVkFFT(&app);
    return 0;
}
