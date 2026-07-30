#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <cfloat>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void selfAttentionKernel(T *out,
                                    const T *query,
                                    const T *key,
                                    const T *value,
                                    size_t query_length,
                                    size_t key_length,
                                    size_t heads,
                                    size_t kv_heads,
                                    size_t head_size,
                                    size_t value_size,
                                    float scale) {
    extern __shared__ float scores[];
    const size_t query_index = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t group_size = heads / kv_heads;
    const size_t kv_head = head / group_size;
    const size_t available = key_length - query_length + query_index + 1;

    if (threadIdx.x == 0) {
        float maximum = -FLT_MAX;
        const T *query_vector = query + (query_index * heads + head) * head_size;
        for (size_t sequence = 0; sequence < available; ++sequence) {
            const T *key_vector = key + (sequence * kv_heads + kv_head) * head_size;
            float score = 0.0F;
            for (size_t dimension = 0; dimension < head_size; ++dimension) {
                score += device::nvidia::toFloat(query_vector[dimension]) * device::nvidia::toFloat(key_vector[dimension]);
            }
            score *= scale;
            scores[sequence] = score;
            maximum = fmaxf(maximum, score);
        }
        float sum = 0.0F;
        for (size_t sequence = 0; sequence < available; ++sequence) {
            scores[sequence] = expf(scores[sequence] - maximum);
            sum += scores[sequence];
        }
        for (size_t sequence = 0; sequence < available; ++sequence) {
            scores[sequence] /= sum;
        }
    }
    __syncthreads();

    for (size_t dimension = threadIdx.x; dimension < value_size; dimension += blockDim.x) {
        float result = 0.0F;
        for (size_t sequence = 0; sequence < available; ++sequence) {
            const T cached = value[(sequence * kv_heads + kv_head) * value_size + dimension];
            result += scores[sequence] * device::nvidia::toFloat(cached);
        }
        out[(query_index * heads + head) * value_size + dimension] = device::nvidia::fromFloat<T>(result);
    }
}

template <typename T>
void launch(std::byte *out,
            const std::byte *query,
            const std::byte *key,
            const std::byte *value,
            size_t query_length,
            size_t key_length,
            size_t heads,
            size_t kv_heads,
            size_t head_size,
            size_t value_size,
            float scale) {
    const dim3 blocks(
        static_cast<unsigned int>(query_length),
        static_cast<unsigned int>(heads));
    selfAttentionKernel<<<blocks, 256, key_length * sizeof(float)>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(query),
        reinterpret_cast<const T *>(key),
        reinterpret_cast<const T *>(value),
        query_length,
        key_length,
        heads,
        kv_heads,
        head_size,
        value_size,
        scale);
    device::nvidia::checkCudaLaunch("selfAttentionKernel");
}
} // namespace

void selfAttention(std::byte *out,
                   const std::byte *query,
                   const std::byte *key,
                   const std::byte *value,
                   llaisysDataType_t dtype,
                   size_t query_length,
                   size_t key_length,
                   size_t heads,
                   size_t kv_heads,
                   size_t head_size,
                   size_t value_size,
                   float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(
            out, query, key, value, query_length, key_length, heads, kv_heads,
            head_size, value_size, scale);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(
            out, query, key, value, query_length, key_length, heads, kv_heads,
            head_size, value_size, scale);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(
            out, query, key, value, query_length, key_length, heads, kv_heads,
            head_size, value_size, scale);
    default:
        throw std::invalid_argument("CUDA self_attention: unsupported dtype");
    }
}
} // namespace llaisys::ops::nvidia
