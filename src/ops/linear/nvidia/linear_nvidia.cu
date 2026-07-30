#include "linear_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {
constexpr int tile = 16;

template <typename T>
__global__ void linearKernel(T *out,
                             const T *in,
                             const T *weight,
                             const T *bias,
                             size_t rows,
                             size_t inner,
                             size_t columns) {
    __shared__ float input_tile[tile][tile];
    __shared__ float weight_tile[tile][tile];
    const size_t row = blockIdx.y * tile + threadIdx.y;
    const size_t column = blockIdx.x * tile + threadIdx.x;
    float accumulator = 0.0F;

    for (size_t offset = 0; offset < inner; offset += tile) {
        input_tile[threadIdx.y][threadIdx.x] = row < rows && offset + threadIdx.x < inner
                                                 ? device::nvidia::toFloat(in[row * inner + offset + threadIdx.x])
                                                 : 0.0F;
        weight_tile[threadIdx.y][threadIdx.x] = column < columns && offset + threadIdx.y < inner
                                                  ? device::nvidia::toFloat(weight[column * inner + offset + threadIdx.y])
                                                  : 0.0F;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < tile; ++i) {
            accumulator += input_tile[threadIdx.y][i] * weight_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && column < columns) {
        if (bias != nullptr) {
            accumulator += device::nvidia::toFloat(bias[column]);
        }
        out[row * columns + column] = device::nvidia::fromFloat<T>(accumulator);
    }
}

template <typename T>
void launch(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            size_t rows,
            size_t inner,
            size_t columns) {
    const dim3 threads(tile, tile);
    const dim3 blocks(
        static_cast<unsigned int>((columns + tile - 1) / tile),
        static_cast<unsigned int>((rows + tile - 1) / tile));
    linearKernel<<<blocks, threads>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        reinterpret_cast<const T *>(bias),
        rows,
        inner,
        columns);
    device::nvidia::checkCudaLaunch("linearKernel");
}
} // namespace

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t rows,
            size_t inner,
            size_t columns) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, weight, bias, rows, inner, columns);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, in, weight, bias, rows, inner, columns);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, in, weight, bias, rows, inner, columns);
    default:
        throw std::invalid_argument("CUDA linear: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
