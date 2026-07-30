#pragma once

#include "llaisys/models/qwen2.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaisys::models {

struct Weight {
    std::vector<uint16_t> data;
    std::vector<size_t> shape;

    bool loaded() const {
        return !data.empty();
    }
};

struct Qwen2LayerWeights {
    Weight input_norm;
    Weight post_attention_norm;
    Weight q_weight;
    Weight q_bias;
    Weight k_weight;
    Weight k_bias;
    Weight v_weight;
    Weight v_bias;
    Weight o_weight;
    Weight gate_weight;
    Weight up_weight;
    Weight down_weight;
};

class Qwen2Model {
public:
    explicit Qwen2Model(const LlaisysQwen2Config &config);

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
    struct LayerCache {
        std::vector<uint16_t> key;
        std::vector<uint16_t> value;
    };

    LlaisysQwen2Config _config;
    Weight _embedding;
    Weight _final_norm;
    Weight _lm_head;
    std::vector<Qwen2LayerWeights> _layers;
    std::vector<LayerCache> _cache;
    bool _ready = false;

    std::vector<uint16_t> _hidden;
    std::vector<uint16_t> _norm;
    std::vector<uint16_t> _q;
    std::vector<uint16_t> _k;
    std::vector<uint16_t> _v;
    std::vector<uint16_t> _attention;
    std::vector<uint16_t> _projection;
    std::vector<uint16_t> _gate;
    std::vector<uint16_t> _up;
    std::vector<uint16_t> _mlp;
    std::vector<float> _scores;

    int64_t forwardToken(int64_t token, size_t position);
    Weight *findWeight(const std::string &name);
};

} // namespace llaisys::models
