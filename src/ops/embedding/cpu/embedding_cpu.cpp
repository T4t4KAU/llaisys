#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

void embedding_(std::byte *out, const int64_t *index, const std::byte *weight, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        std::memcpy(out + i * d, weight + index[i] * d, d);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t n, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_BF16:
    case LLAISYS_DTYPE_F16:
        return embedding_(out, reinterpret_cast<const int64_t *>(index), weight, n, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu