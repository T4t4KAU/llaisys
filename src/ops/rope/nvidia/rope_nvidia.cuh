#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(std::byte *out,
          const std::byte *in,
          const std::byte *positions,
          float theta,
          llaisysDataType_t dtype,
          size_t sequence_length,
          size_t heads,
          size_t head_size);
}
