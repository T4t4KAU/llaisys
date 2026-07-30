#pragma once

#include "llaisys/models/qwen2.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace llaisys::models {

class Qwen2NvidiaModel {
public:
    explicit Qwen2NvidiaModel(const LlaisysQwen2Config &config);
    ~Qwen2NvidiaModel();

    bool loadWeight(const std::string &name,
                    const void *data,
                    const size_t *shape,
                    size_t ndim,
                    llaisysDataType_t dtype);
    bool finalize();
    size_t generate(const int64_t *input_ids,
                    size_t input_count,
                    size_t max_new_tokens,
                    int64_t *output_ids,
                    size_t output_capacity);

private:
    struct Impl;
    Impl *_impl;
};

} // namespace llaisys::models
