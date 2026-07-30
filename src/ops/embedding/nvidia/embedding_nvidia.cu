#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void embeddingKernel(T *out,
                                const int64_t *indices,
                                const T *weight,
                                size_t rows,
                                size_t columns) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = rows * columns;
    if (index < count) {
        const size_t row = index / columns;
        const size_t column = index % columns;
        out[index] = weight[static_cast<size_t>(indices[row]) * columns + column];
    }
}

template <typename T>
void launch(std::byte *out,
            const std::byte *indices,
            const std::byte *weight,
            size_t rows,
            size_t columns) {
    constexpr int threads = 256;
    const size_t count = rows * columns;
    embeddingKernel<<<(count + threads - 1) / threads, threads>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(indices),
        reinterpret_cast<const T *>(weight),
        rows,
        columns);
    device::nvidia::checkCudaLaunch("embeddingKernel");
}
} // namespace

void embedding(std::byte *out,
               const std::byte *indices,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t rows,
               size_t columns) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, indices, weight, rows, columns);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, indices, weight, rows, columns);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, indices, weight, rows, columns);
    default:
        throw std::invalid_argument("CUDA embedding: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
