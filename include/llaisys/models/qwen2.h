#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../../llaisys.h"

__C {
    typedef struct LlaisysQwen2Model *llaisysQwen2Model_t;

    typedef struct {
        size_t vocab_size;
        size_t hidden_size;
        size_t intermediate_size;
        size_t num_hidden_layers;
        size_t num_attention_heads;
        size_t num_key_value_heads;
        float rms_norm_eps;
        float rope_theta;
        int64_t eos_token_id;
        llaisysDeviceType_t device_type;
        int device_id;
    } LlaisysQwen2Config;

    __export llaisysQwen2Model_t llaisysQwen2Create(const LlaisysQwen2Config *config);
    __export void llaisysQwen2Destroy(llaisysQwen2Model_t model);

    __export int llaisysQwen2LoadWeight(
        llaisysQwen2Model_t model,
        const char *name,
        const void *data,
        const size_t *shape,
        size_t ndim,
        llaisysDataType_t dtype);

    __export int llaisysQwen2Finalize(llaisysQwen2Model_t model);

    // Returns the total number of tokens written, including the input tokens.
    __export size_t llaisysQwen2Generate(
        llaisysQwen2Model_t model,
        const int64_t *input_ids,
        size_t input_count,
        size_t max_new_tokens,
        int64_t *output_ids,
        size_t output_capacity);
}

#endif
