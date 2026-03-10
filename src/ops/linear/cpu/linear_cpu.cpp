#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out,
             const T *in,        // [M, K]
             const T *weight,    // [N, K]  (NOTE: not transposed in memory)
             const T *bias_data, // [N] or nullptr
             size_t M,
             size_t K,
             size_t N) {
    // out: [M, N], row-major contiguous
    // Computes: out = in * weight^T + bias

    const bool has_bias = (bias_data != nullptr);

    for (size_t m = 0; m < M; ++m) {
        const T *x_row = in + m * K; // X[m, :]
        T *y_row = out + m * N;      // Y[m, :]

        for (size_t n = 0; n < N; ++n) {
            const T *w_row = weight + n * K; // W[n, :]

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // Accumulate in float for fp16/bf16
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_row[k]);
                }
                if (has_bias) {
                    acc += llaisys::utils::cast<float>(bias_data[n]);
                }
                y_row[n] = llaisys::utils::cast<T>(acc);
            } else {
                // Accumulate in T for normal types (float/double/etc.)
                T acc = T{};
                for (size_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                if (has_bias) {
                    acc += bias_data[n];
                }
                y_row[n] = acc;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t M, size_t K, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            M, K, N);

    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu