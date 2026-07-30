#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void rmsNormKernel(T *out,
                              const T *in,
                              const T *weight,
                              float eps,
                              size_t columns) {
    __shared__ float reduction[256];
    const size_t row = blockIdx.x;
    float local = 0.0F;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = device::nvidia::toFloat(in[row * columns + column]);
        local += value * value;
    }
    reduction[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / static_cast<float>(columns) + eps);
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = device::nvidia::toFloat(in[row * columns + column]);
        const float weight_value = device::nvidia::toFloat(weight[column]);
        out[row * columns + column] = device::nvidia::fromFloat<T>(value * scale * weight_value);
    }
}

template <typename T>
void launch(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            float eps,
            size_t rows,
            size_t columns) {
    rmsNormKernel<<<rows, 256>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        eps,
        columns);
    device::nvidia::checkCudaLaunch("rmsNormKernel");
}
} // namespace

void rmsNorm(std::byte *out,
             const std::byte *in,
             const std::byte *weight,
             float eps,
             llaisysDataType_t dtype,
             size_t rows,
             size_t columns) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, weight, eps, rows, columns);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, in, weight, eps, rows, columns);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, in, weight, eps, rows, columns);
    default:
        throw std::invalid_argument("CUDA rms_norm: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
