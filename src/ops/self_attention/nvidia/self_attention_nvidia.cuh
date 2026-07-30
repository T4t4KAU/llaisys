#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
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
                   float scale);
}
