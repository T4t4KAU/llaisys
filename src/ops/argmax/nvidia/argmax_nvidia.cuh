#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_index,
            std::byte *max_value,
            const std::byte *values,
            llaisysDataType_t dtype,
            size_t count);
}
