#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps,
               size_t M, size_t D) {
    for (size_t m = 0; m < M; ++m) {
        const T *x = in + m * D;
        T *y = out + m * D;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // accumulate in float for stability
            float sumsq = 0.0f;
            for (size_t j = 0; j < D; ++j) {
                float v = llaisys::utils::cast<float>(x[j]);
                sumsq += v * v;
            }
            float mean_sq = sumsq / static_cast<float>(D);
            float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

            for (size_t j = 0; j < D; ++j) {
                float xv = llaisys::utils::cast<float>(x[j]);
                float wv = llaisys::utils::cast<float>(weight[j]);
                float outv = (xv * inv_rms) * wv;
                y[j] = llaisys::utils::cast<T>(outv);
            }
        } else {
            double sumsq = 0.0;
            for (size_t j = 0; j < D; ++j) {
                double v = static_cast<double>(x[j]);
                sumsq += v * v;
            }
            double mean_sq = sumsq / static_cast<double>(D);
            double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

            for (size_t j = 0; j < D; ++j) {
                // y = (x * inv_rms) * w
                y[j] = static_cast<T>((static_cast<double>(x[j]) * inv_rms) * static_cast<double>(weight[j]));
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              float eps,
              llaisysDataType_t type,
              size_t M,
              size_t D) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_(reinterpret_cast<float *>(out),
                  reinterpret_cast<const float *>(in),
                  reinterpret_cast<const float *>(weight),
                  eps, M, D);
        return;

    case LLAISYS_DTYPE_BF16:
        rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                  reinterpret_cast<const llaisys::bf16_t *>(in),
                  reinterpret_cast<const llaisys::bf16_t *>(weight),
                  eps, M, D);
        return;

    case LLAISYS_DTYPE_F16:
        rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                  reinterpret_cast<const llaisys::fp16_t *>(in),
                  reinterpret_cast<const llaisys::fp16_t *>(weight),
                  eps, M, D);
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu