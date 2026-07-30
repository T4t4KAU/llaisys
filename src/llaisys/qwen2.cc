#include "llaisys/models/qwen2.h"

#include "../models/qwen2/model.hpp"

#include <exception>
#include <iostream>
#include <new>

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *model;
};

__C {
    llaisysQwen2Model_t llaisysQwen2Create(const LlaisysQwen2Config *config) {
        if (config == nullptr) {
            return nullptr;
        }
        try {
            auto *handle = new LlaisysQwen2Model;
            handle->model = new llaisys::models::Qwen2Model(*config);
            return handle;
        } catch (const std::exception &error) {
            std::cerr << "Failed to create Qwen2 model: " << error.what() << std::endl;
            return nullptr;
        }
    }

    void llaisysQwen2Destroy(llaisysQwen2Model_t model) {
        if (model != nullptr) {
            delete model->model;
            delete model;
        }
    }

    int llaisysQwen2LoadWeight(llaisysQwen2Model_t model,
                               const char *name,
                               const void *data,
                               const size_t *shape,
                               size_t ndim,
                               llaisysDataType_t dtype) {
        if (model == nullptr || name == nullptr) {
            return 0;
        }
        try {
            return model->model->loadWeight(name, data, shape, ndim, dtype) ? 1 : 0;
        } catch (const std::exception &error) {
            std::cerr << "Failed to load Qwen2 weight " << name << ": " << error.what() << std::endl;
            return 0;
        }
    }

    int llaisysQwen2Finalize(llaisysQwen2Model_t model) {
        return model != nullptr && model->model->finalize() ? 1 : 0;
    }

    size_t llaisysQwen2Generate(llaisysQwen2Model_t model,
                                const int64_t *input_ids,
                                size_t input_count,
                                size_t max_new_tokens,
                                int64_t *output_ids,
                                size_t output_capacity) {
        if (model == nullptr) {
            return 0;
        }
        try {
            return model->model->generate(
                input_ids, input_count, max_new_tokens, output_ids, output_capacity);
        } catch (const std::exception &error) {
            std::cerr << "Qwen2 generation failed: " << error.what() << std::endl;
            return 0;
        }
    }
}
