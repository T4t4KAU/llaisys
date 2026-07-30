#include "add_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = device::nvidia::fromFloat<T>(
            device::nvidia::toFloat(a[index]) + device::nvidia::toFloat(b[index]));
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *a, const std::byte *b, size_t count) {
    constexpr int threads = 256;
    addKernel<<<(count + threads - 1) / threads, threads>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        count);
    device::nvidia::checkCudaLaunch("addKernel");
}
} // namespace

void add(std::byte *out,
         const std::byte *a,
         const std::byte *b,
         llaisysDataType_t dtype,
         size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, a, b, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, a, b, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, a, b, count);
    default:
        throw std::invalid_argument("CUDA add: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
