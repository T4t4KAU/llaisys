#include "rope_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void ropeKernel(T *out,
                           const T *in,
                           const int64_t *positions,
                           float theta,
                           size_t heads,
                           size_t head_size,
                           size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const size_t half = head_size / 2;
    const size_t pair = index % half;
    const size_t head = (index / half) % heads;
    const size_t sequence = index / (half * heads);
    const size_t base = (sequence * heads + head) * head_size;
    const float angle = static_cast<float>(positions[sequence]) / powf(theta, 2.0F * static_cast<float>(pair) / static_cast<float>(head_size));
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = device::nvidia::toFloat(in[base + pair]);
    const float second = device::nvidia::toFloat(in[base + pair + half]);
    out[base + pair] = device::nvidia::fromFloat<T>(first * cosine - second * sine);
    out[base + pair + half] = device::nvidia::fromFloat<T>(second * cosine + first * sine);
}

template <typename T>
void launch(std::byte *out,
            const std::byte *in,
            const std::byte *positions,
            float theta,
            size_t sequence_length,
            size_t heads,
            size_t head_size) {
    constexpr int threads = 256;
    const size_t count = sequence_length * heads * head_size / 2;
    ropeKernel<<<(count + threads - 1) / threads, threads>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(positions),
        theta,
        heads,
        head_size,
        count);
    device::nvidia::checkCudaLaunch("ropeKernel");
}
} // namespace

void rope(std::byte *out,
          const std::byte *in,
          const std::byte *positions,
          float theta,
          llaisysDataType_t dtype,
          size_t sequence_length,
          size_t heads,
          size_t head_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, positions, theta, sequence_length, heads, head_size);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, in, positions, theta, sequence_length, heads, head_size);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, in, positions, theta, sequence_length, heads, head_size);
    default:
        throw std::invalid_argument("CUDA rope: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
