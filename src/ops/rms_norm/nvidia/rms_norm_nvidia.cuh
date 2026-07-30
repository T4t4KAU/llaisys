#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rmsNorm(std::byte *out,
             const std::byte *in,
             const std::byte *weight,
             float eps,
             llaisysDataType_t dtype,
             size_t rows,
             size_t columns);
}
