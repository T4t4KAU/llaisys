#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float gate_value = device::nvidia::toFloat(gate[index]);
        const float up_value = device::nvidia::toFloat(up[index]);
        out[index] = device::nvidia::fromFloat<T>(
            up_value * gate_value / (1.0F + expf(-gate_value)));
    }
}

template <typename T>
void launch(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            size_t count) {
    constexpr int threads = 256;
    swigluKernel<<<(count + threads - 1) / threads, threads>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        count);
    device::nvidia::checkCudaLaunch("swigluKernel");
}
} // namespace

void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, gate, up, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, gate, up, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, gate, up, count);
    default:
        throw std::invalid_argument("CUDA swiglu: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
