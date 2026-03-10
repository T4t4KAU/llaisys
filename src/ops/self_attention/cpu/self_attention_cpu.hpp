#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(
    std::byte *attn_val,
    const std::byte *Q, const std::byte *K, const std::byte *V, llaisysDataType_t type,
    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale);
}