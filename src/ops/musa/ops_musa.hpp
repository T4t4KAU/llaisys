#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {
void add(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t);
void argmax(std::byte *, std::byte *, const std::byte *, llaisysDataType_t, size_t);
void embedding(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t);
void linear(std::byte *, const std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, size_t);
void rmsNorm(std::byte *, const std::byte *, const std::byte *, float, llaisysDataType_t, size_t, size_t);
void rope(std::byte *, const std::byte *, const std::byte *, float, llaisysDataType_t, size_t, size_t, size_t);
void selfAttention(std::byte *, const std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, size_t, size_t, size_t, size_t, float);
void swiglu(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t);
} // namespace llaisys::ops::musa
