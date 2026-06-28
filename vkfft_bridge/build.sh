#!/bin/bash
# Build the VkFFT-Metal zero-copy bridge dylib for mlx-nufft's optional
# fft_backend="vkfft". Clones VkFFT (header-only) if absent, then one clang++.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f VkFFT/vkFFT/vkFFT.h ]; then
  echo "cloning VkFFT (header-only)..."
  git clone --depth 1 https://github.com/DTolm/VkFFT
fi
clang++ -dynamiclib -std=c++17 -O3 -DVKFFT_BACKEND=5 \
  -I VkFFT/vkFFT -I VkFFT/metal-cpp \
  -framework Metal -framework Foundation -framework QuartzCore \
  vkfft_bridge.cpp -o libvkfft_bridge.dylib
echo "built $(pwd)/libvkfft_bridge.dylib"
