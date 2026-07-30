#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cfloat>
#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void argmaxKernel(int64_t *max_index,
                             T *max_value,
                             const T *values,
                             size_t count) {
    __shared__ float shared_values[256];
    __shared__ int64_t shared_indices[256];
    float local_value = -FLT_MAX;
    int64_t local_index = 0;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = device::nvidia::toFloat(values[i]);
        if (value > local_value || (value == local_value && static_cast<int64_t>(i) < local_index)) {
            local_value = value;
            local_index = static_cast<int64_t>(i);
        }
    }
    shared_values[threadIdx.x] = local_value;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = shared_values[threadIdx.x + stride];
            const int64_t other_index = shared_indices[threadIdx.x + stride];
            if (other > shared_values[threadIdx.x] || (other == shared_values[threadIdx.x] && other_index < shared_indices[threadIdx.x])) {
                shared_values[threadIdx.x] = other;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *max_index = shared_indices[0];
        *max_value = values[shared_indices[0]];
    }
}

template <typename T>
void launch(std::byte *max_index,
            std::byte *max_value,
            const std::byte *values,
            size_t count) {
    argmaxKernel<<<1, 256>>>(
        reinterpret_cast<int64_t *>(max_index),
        reinterpret_cast<T *>(max_value),
        reinterpret_cast<const T *>(values),
        count);
    device::nvidia::checkCudaLaunch("argmaxKernel");
}
} // namespace

void argmax(std::byte *max_index,
            std::byte *max_value,
            const std::byte *values,
            llaisysDataType_t dtype,
            size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(max_index, max_value, values, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(max_index, max_value, values, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(max_index, max_value, values, count);
    default:
        throw std::invalid_argument("CUDA argmax: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
