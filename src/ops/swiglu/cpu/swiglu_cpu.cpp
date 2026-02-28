#include "swiglu_cpu.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "../../../utils.hpp"

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            const float g = llaisys::utils::cast<float>(gate[i]);
            const float u = llaisys::utils::cast<float>(up[i]);

            // sigmoid(g) = 1 / (1 + exp(-g))
            const float sig = 1.0f / (1.0f + std::exp(-g));
            const float y = u * (g * sig);

            out[i] = llaisys::utils::cast<T>(y);
        } else {
            const T g = gate[i];
            const T u = up[i];

            const T sig = T(1) / (T(1) + static_cast<T>(std::exp(-static_cast<double>(g))));
            out[i] = u * (g * sig);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up),
                       numel);
        return;
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up),
                       numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu