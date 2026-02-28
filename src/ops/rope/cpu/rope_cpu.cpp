#include "rope_cpu.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../../../utils.hpp"

template <typename T>
void rope_(T *out,
           const T *in,
           const int64_t *pos_ids,
           float theta,
           size_t seq_len,
           size_t nhead,
           size_t d) {
    const size_t half = d / 2;

    // inv_freq[j] = theta^(-2j/d)
    std::vector<double> inv_freq(half);
    const double log_theta = std::log(static_cast<double>(theta));
    const double inv_d = 1.0 / static_cast<double>(d);
    for (size_t j = 0; j < half; ++j) {
        inv_freq[j] = std::exp(-log_theta * (2.0 * static_cast<double>(j) * inv_d));
    }

    for (size_t t = 0; t < seq_len; ++t) {
        const double pos = static_cast<double>(pos_ids[t]);

        for (size_t h = 0; h < nhead; ++h) {
            const size_t base = (t * nhead + h) * d;
            const T *x = in + base;
            T *y = out + base;

            for (size_t j = 0; j < half; ++j) {
                const double angle = pos * inv_freq[j];
                const double c = std::cos(angle);
                const double s = std::sin(angle);

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    const double a = static_cast<double>(llaisys::utils::cast<float>(x[j]));
                    const double b = static_cast<double>(llaisys::utils::cast<float>(x[j + half]));

                    const double ap = a * c - b * s;
                    const double bp = b * c + a * s;

                    y[j] = llaisys::utils::cast<T>(static_cast<float>(ap));
                    y[j + half] = llaisys::utils::cast<T>(static_cast<float>(bp));
                } else {
                    const double a = static_cast<double>(x[j]);
                    const double b = static_cast<double>(x[j + half]);

                    const double ap = a * c - b * s;
                    const double bp = b * c + a * s;

                    y[j] = static_cast<T>(ap);
                    y[j + half] = static_cast<T>(bp);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          float theta,
          llaisysDataType_t type,
          size_t seq_len,
          size_t nhead,
          size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     theta, seq_len, nhead, d);

    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     theta, seq_len, nhead, d);

    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     theta, seq_len, nhead, d);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu