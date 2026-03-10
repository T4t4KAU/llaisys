#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void argmax_(int64_t &max_idx, T &max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        max_idx = -1;
        return;
    }

    max_idx = 0;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float best_val = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; i++) {
            float current = llaisys::utils::cast<float>(vals[i]);
            if (current > best_val) {
                best_val = current;
                max_idx = static_cast<int>(i);
            }
        }
        max_val = llaisys::utils::cast<T>(best_val);
    } else {
        T best_val = vals[0];
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > best_val) {
                best_val = vals[i];
                max_idx = static_cast<int>(i);
            }
        }
        max_val = best_val;
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        argmax_(reinterpret_cast<int64_t *>(max_idx)[0],
                reinterpret_cast<float *>(max_val)[0],
                reinterpret_cast<const float *>(vals),
                numel);
        return;
    }
    case LLAISYS_DTYPE_BF16: {
        argmax_(reinterpret_cast<int64_t *>(max_idx)[0],
                reinterpret_cast<llaisys::bf16_t *>(max_val)[0],
                reinterpret_cast<const llaisys::bf16_t *>(vals),
                numel);
        return;
    }
    case LLAISYS_DTYPE_F16: {
        argmax_(reinterpret_cast<int64_t *>(max_idx)[0],
                reinterpret_cast<llaisys::fp16_t *>(max_val)[0],
                reinterpret_cast<const llaisys::fp16_t *>(vals),
                numel);
        return;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu