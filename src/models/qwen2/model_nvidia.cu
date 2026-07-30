#include "model_nvidia.cuh"

#include "../../device/nvidia/cuda_utils.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaisys::models {
namespace {

using device::nvidia::checkCuda;
using device::nvidia::checkCudaLaunch;

struct CudaWeight {
    __nv_bfloat16 *data = nullptr;
    std::vector<size_t> shape;
};

struct CudaLayer {
    const CudaWeight *input_norm;
    const CudaWeight *post_attention_norm;
    const CudaWeight *q_weight;
    const CudaWeight *q_bias;
    const CudaWeight *k_weight;
    const CudaWeight *k_bias;
    const CudaWeight *v_weight;
    const CudaWeight *v_bias;
    const CudaWeight *o_weight;
    const CudaWeight *gate_weight;
    const CudaWeight *up_weight;
    const CudaWeight *down_weight;
};

__device__ inline __nv_bfloat16 rounded(float value) {
    return __float2bfloat16_rn(value);
}

__global__ void rmsNormKernel(__nv_bfloat16 *out,
                              const __nv_bfloat16 *in,
                              const __nv_bfloat16 *weight,
                              float eps,
                              size_t count) {
    __shared__ float reduction[256];
    float local = 0.0F;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = __bfloat162float(in[i]);
        local += value * value;
    }
    reduction[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / static_cast<float>(count) + eps);
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const __nv_bfloat16 normalized = rounded(__bfloat162float(in[i]) * scale);
        out[i] = rounded(__bfloat162float(normalized) * __bfloat162float(weight[i]));
    }
}

__global__ void linearKernel(__nv_bfloat16 *out,
                             const __nv_bfloat16 *in,
                             const __nv_bfloat16 *weight,
                             const __nv_bfloat16 *bias,
                             size_t rows,
                             size_t columns) {
    __shared__ float reduction[256];
    const size_t row = blockIdx.x;
    float local = 0.0F;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        local += __bfloat162float(in[column]) * __bfloat162float(weight[row * columns + column]);
    }
    reduction[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float value = reduction[0];
        if (bias != nullptr) {
            value += __bfloat162float(bias[row]);
        }
        out[row] = rounded(value);
    }
}

__global__ void addKernel(__nv_bfloat16 *left,
                          const __nv_bfloat16 *right,
                          size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        left[index] = rounded(__bfloat162float(left[index]) + __bfloat162float(right[index]));
    }
}

__global__ void swigluKernel(__nv_bfloat16 *out,
                             const __nv_bfloat16 *gate,
                             const __nv_bfloat16 *up,
                             size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float gate_value = __bfloat162float(gate[index]);
        const __nv_bfloat16 activated = rounded(gate_value / (1.0F + expf(-gate_value)));
        out[index] = rounded(__bfloat162float(activated) * __bfloat162float(up[index]));
    }
}

__global__ void ropeKernel(__nv_bfloat16 *query,
                           __nv_bfloat16 *key,
                           size_t query_heads,
                           size_t key_heads,
                           size_t head_size,
                           size_t position,
                           float theta) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t half = head_size / 2;
    const size_t total_pairs = (query_heads + key_heads) * half;
    if (index >= total_pairs) {
        return;
    }
    const size_t combined_head = index / half;
    const size_t pair = index % half;
    __nv_bfloat16 *values = combined_head < query_heads
                              ? query + combined_head * head_size
                              : key + (combined_head - query_heads) * head_size;
    const float angle = static_cast<float>(position) / powf(theta, 2.0F * static_cast<float>(pair) / static_cast<float>(head_size));
    const __nv_bfloat16 cosine = rounded(cosf(angle));
    const __nv_bfloat16 sine = rounded(sinf(angle));
    const float first = __bfloat162float(values[pair]);
    const float second = __bfloat162float(values[pair + half]);
    const __nv_bfloat16 first_product = rounded(first * __bfloat162float(cosine));
    const __nv_bfloat16 second_product = rounded(second * __bfloat162float(sine));
    const __nv_bfloat16 third_product = rounded(second * __bfloat162float(cosine));
    const __nv_bfloat16 fourth_product = rounded(first * __bfloat162float(sine));
    values[pair] = rounded(
        __bfloat162float(first_product) - __bfloat162float(second_product));
    values[pair + half] = rounded(
        __bfloat162float(third_product) + __bfloat162float(fourth_product));
}

__global__ void attentionKernel(__nv_bfloat16 *out,
                                const __nv_bfloat16 *query,
                                const __nv_bfloat16 *key_cache,
                                const __nv_bfloat16 *value_cache,
                                size_t context,
                                size_t heads,
                                size_t kv_heads,
                                size_t head_size) {
    extern __shared__ float scores[];
    const size_t head = blockIdx.x;
    const size_t kv_head = head / (heads / kv_heads);
    const size_t kv_size = kv_heads * head_size;
    const __nv_bfloat16 *query_vector = query + head * head_size;

    if (threadIdx.x == 0) {
        float maximum = -FLT_MAX;
        const float scale = rsqrtf(static_cast<float>(head_size));
        for (size_t sequence = 0; sequence < context; ++sequence) {
            const __nv_bfloat16 *key = key_cache + sequence * kv_size + kv_head * head_size;
            float dot = 0.0F;
            for (size_t dimension = 0; dimension < head_size; ++dimension) {
                dot += __bfloat162float(query_vector[dimension]) * __bfloat162float(key[dimension]);
            }
            const __nv_bfloat16 dot_bf16 = rounded(dot);
            const __nv_bfloat16 scaled = rounded(__bfloat162float(dot_bf16) * scale);
            scores[sequence] = __bfloat162float(scaled);
            maximum = fmaxf(maximum, scores[sequence]);
        }
        float sum = 0.0F;
        for (size_t sequence = 0; sequence < context; ++sequence) {
            scores[sequence] = expf(scores[sequence] - maximum);
            sum += scores[sequence];
        }
        for (size_t sequence = 0; sequence < context; ++sequence) {
            scores[sequence] = __bfloat162float(rounded(scores[sequence] / sum));
        }
    }
    __syncthreads();

    for (size_t dimension = threadIdx.x; dimension < head_size; dimension += blockDim.x) {
        float value = 0.0F;
        for (size_t sequence = 0; sequence < context; ++sequence) {
            const __nv_bfloat16 cached = value_cache[sequence * kv_size + kv_head * head_size + dimension];
            value += scores[sequence] * __bfloat162float(cached);
        }
        out[head * head_size + dimension] = rounded(value);
    }
}

__global__ void argmaxKernel(int64_t *result,
                             const __nv_bfloat16 *values,
                             size_t count) {
    __shared__ float shared_values[256];
    __shared__ int64_t shared_indices[256];
    float local_value = -FLT_MAX;
    int64_t local_index = 0;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = __bfloat162float(values[i]);
        if (value > local_value || (value == local_value && static_cast<int64_t>(i) < local_index)) {
            local_value = value;
            local_index = static_cast<int64_t>(i);
        }
    }
    shared_values[threadIdx.x] = local_value;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = shared_values[threadIdx.x + stride];
            const int64_t other_index = shared_indices[threadIdx.x + stride];
            if (other > shared_values[threadIdx.x] || (other == shared_values[threadIdx.x] && other_index < shared_indices[threadIdx.x])) {
                shared_values[threadIdx.x] = other;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *result = shared_indices[0];
    }
}

void allocate(__nv_bfloat16 **pointer, size_t count) {
    checkCuda(cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(__nv_bfloat16)),
              "cudaMalloc Qwen2 buffer");
}

void launchLinear(__nv_bfloat16 *out,
                  const __nv_bfloat16 *in,
                  const CudaWeight &weight,
                  const CudaWeight *bias = nullptr) {
    linearKernel<<<weight.shape[0], 256>>>(
        out,
        in,
        weight.data,
        bias == nullptr ? nullptr : bias->data,
        weight.shape[0],
        weight.shape[1]);
    checkCudaLaunch("Qwen2 linearKernel");
}

} // namespace

struct Qwen2NvidiaModel::Impl {
    explicit Impl(const LlaisysQwen2Config &model_config)
        : config(model_config) {
        checkCuda(cudaSetDevice(config.device_id), "cudaSetDevice Qwen2");
        const size_t hidden = config.hidden_size;
        const size_t kv = config.num_key_value_heads * hidden / config.num_attention_heads;
        allocate(&hidden_state, hidden);
        allocate(&norm, hidden);
        allocate(&query, hidden);
        allocate(&key, kv);
        allocate(&value, kv);
        allocate(&attention, hidden);
        allocate(&projection, hidden);
        allocate(&gate, config.intermediate_size);
        allocate(&up, config.intermediate_size);
        allocate(&mlp, config.intermediate_size);
        allocate(&logits, config.vocab_size);
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&next_token), sizeof(int64_t)),
                  "cudaMalloc Qwen2 token");
    }

    ~Impl() {
        cudaSetDevice(config.device_id);
        for (auto &entry : weights) {
            cudaFree(entry.second.data);
        }
        for (void *pointer : {
                 static_cast<void *>(hidden_state),
                 static_cast<void *>(norm),
                 static_cast<void *>(query),
                 static_cast<void *>(key),
                 static_cast<void *>(value),
                 static_cast<void *>(attention),
                 static_cast<void *>(projection),
                 static_cast<void *>(gate),
                 static_cast<void *>(up),
                 static_cast<void *>(mlp),
                 static_cast<void *>(logits),
                 static_cast<void *>(next_token),
                 static_cast<void *>(key_cache),
                 static_cast<void *>(value_cache)}) {
            if (pointer != nullptr) {
                cudaFree(pointer);
            }
        }
    }

    const CudaWeight *get(const std::string &name) const {
        auto iterator = weights.find(name);
        return iterator == weights.end() ? nullptr : &iterator->second;
    }

    int64_t forward(int64_t token, size_t position) {
        const size_t hidden = config.hidden_size;
        const size_t heads = config.num_attention_heads;
        const size_t kv_heads = config.num_key_value_heads;
        const size_t head_size = hidden / heads;
        const size_t kv_size = kv_heads * head_size;
        checkCuda(cudaMemcpy(
                      hidden_state,
                      embedding->data + static_cast<size_t>(token) * hidden,
                      hidden * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToDevice),
                  "cudaMemcpy Qwen2 embedding");

        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const CudaLayer &layer = layers[layer_index];
            rmsNormKernel<<<1, 256>>>(
                norm, hidden_state, layer.input_norm->data,
                config.rms_norm_eps, hidden);
            launchLinear(query, norm, *layer.q_weight, layer.q_bias);
            launchLinear(key, norm, *layer.k_weight, layer.k_bias);
            launchLinear(value, norm, *layer.v_weight, layer.v_bias);

            constexpr int threads = 256;
            const size_t pairs = (heads + kv_heads) * head_size / 2;
            ropeKernel<<<(pairs + threads - 1) / threads, threads>>>(
                query, key, heads, kv_heads, head_size, position, config.rope_theta);
            checkCuda(cudaMemcpy(
                          key_cache + (layer_index * cache_capacity + position) * kv_size,
                          key,
                          kv_size * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice),
                      "cudaMemcpy Qwen2 key cache");
            checkCuda(cudaMemcpy(
                          value_cache + (layer_index * cache_capacity + position) * kv_size,
                          value,
                          kv_size * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice),
                      "cudaMemcpy Qwen2 value cache");
            attentionKernel<<<heads, 256, (position + 1) * sizeof(float)>>>(
                attention,
                query,
                key_cache + layer_index * cache_capacity * kv_size,
                value_cache + layer_index * cache_capacity * kv_size,
                position + 1,
                heads,
                kv_heads,
                head_size);
            launchLinear(projection, attention, *layer.o_weight);
            addKernel<<<(hidden + threads - 1) / threads, threads>>>(
                hidden_state, projection, hidden);
            rmsNormKernel<<<1, 256>>>(
                norm, hidden_state, layer.post_attention_norm->data,
                config.rms_norm_eps, hidden);
            launchLinear(gate, norm, *layer.gate_weight);
            launchLinear(up, norm, *layer.up_weight);
            swigluKernel<<<(config.intermediate_size + threads - 1) / threads, threads>>>(
                mlp, gate, up, config.intermediate_size);
            launchLinear(projection, mlp, *layer.down_weight);
            addKernel<<<(hidden + threads - 1) / threads, threads>>>(
                hidden_state, projection, hidden);
        }

        rmsNormKernel<<<1, 256>>>(
            norm, hidden_state, final_norm->data, config.rms_norm_eps, hidden);
        launchLinear(logits, norm, *lm_head);
        argmaxKernel<<<1, 256>>>(next_token, logits, config.vocab_size);
        checkCudaLaunch("Qwen2 forward kernels");
        int64_t host_token = 0;
        checkCuda(cudaMemcpy(
                      &host_token, next_token, sizeof(host_token), cudaMemcpyDeviceToHost),
                  "cudaMemcpy Qwen2 result");
        return host_token;
    }

    LlaisysQwen2Config config;
    std::unordered_map<std::string, CudaWeight> weights;
    const CudaWeight *embedding = nullptr;
    const CudaWeight *final_norm = nullptr;
    const CudaWeight *lm_head = nullptr;
    std::vector<CudaLayer> layers;
    bool ready = false;

    __nv_bfloat16 *hidden_state = nullptr;
    __nv_bfloat16 *norm = nullptr;
    __nv_bfloat16 *query = nullptr;
    __nv_bfloat16 *key = nullptr;
    __nv_bfloat16 *value = nullptr;
    __nv_bfloat16 *attention = nullptr;
    __nv_bfloat16 *projection = nullptr;
    __nv_bfloat16 *gate = nullptr;
    __nv_bfloat16 *up = nullptr;
    __nv_bfloat16 *mlp = nullptr;
    __nv_bfloat16 *logits = nullptr;
    int64_t *next_token = nullptr;
    __nv_bfloat16 *key_cache = nullptr;
    __nv_bfloat16 *value_cache = nullptr;
    size_t cache_capacity = 0;
};

Qwen2NvidiaModel::Qwen2NvidiaModel(const LlaisysQwen2Config &config)
    : _impl(new Impl(config)) {}

Qwen2NvidiaModel::~Qwen2NvidiaModel() {
    delete _impl;
}

bool Qwen2NvidiaModel::loadWeight(const std::string &name,
                                  const void *data,
                                  const size_t *shape,
                                  size_t ndim,
                                  llaisysDataType_t dtype) {
    if (_impl->ready || data == nullptr || dtype != LLAISYS_DTYPE_BF16 || _impl->weights.count(name) != 0) {
        return false;
    }
    CudaWeight weight;
    weight.shape.assign(shape, shape + ndim);
    size_t elements = 1;
    for (size_t dimension : weight.shape) {
        elements *= dimension;
    }
    checkCuda(cudaMalloc(
                  reinterpret_cast<void **>(&weight.data),
                  elements * sizeof(__nv_bfloat16)),
              "cudaMalloc Qwen2 weight");
    checkCuda(cudaMemcpy(
                  weight.data,
                  data,
                  elements * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice),
              "cudaMemcpy Qwen2 weight");
    _impl->weights.emplace(name, std::move(weight));
    return true;
}

bool Qwen2NvidiaModel::finalize() {
    _impl->embedding = _impl->get("model.embed_tokens.weight");
    _impl->final_norm = _impl->get("model.norm.weight");
    _impl->lm_head = _impl->get("lm_head.weight");
    if (_impl->embedding == nullptr || _impl->final_norm == nullptr || _impl->lm_head == nullptr) {
        return false;
    }

    _impl->layers.clear();
    for (size_t index = 0; index < _impl->config.num_hidden_layers; ++index) {
        const std::string prefix = "model.layers." + std::to_string(index) + ".";
        CudaLayer layer{
            _impl->get(prefix + "input_layernorm.weight"),
            _impl->get(prefix + "post_attention_layernorm.weight"),
            _impl->get(prefix + "self_attn.q_proj.weight"),
            _impl->get(prefix + "self_attn.q_proj.bias"),
            _impl->get(prefix + "self_attn.k_proj.weight"),
            _impl->get(prefix + "self_attn.k_proj.bias"),
            _impl->get(prefix + "self_attn.v_proj.weight"),
            _impl->get(prefix + "self_attn.v_proj.bias"),
            _impl->get(prefix + "self_attn.o_proj.weight"),
            _impl->get(prefix + "mlp.gate_proj.weight"),
            _impl->get(prefix + "mlp.up_proj.weight"),
            _impl->get(prefix + "mlp.down_proj.weight")};
        if (layer.input_norm == nullptr || layer.post_attention_norm == nullptr || layer.q_weight == nullptr || layer.q_bias == nullptr || layer.k_weight == nullptr || layer.k_bias == nullptr || layer.v_weight == nullptr || layer.v_bias == nullptr || layer.o_weight == nullptr || layer.gate_weight == nullptr || layer.up_weight == nullptr || layer.down_weight == nullptr) {
            return false;
        }
        _impl->layers.push_back(layer);
    }
    _impl->ready = true;
    return true;
}

size_t Qwen2NvidiaModel::generate(const int64_t *input_ids,
                                  size_t input_count,
                                  size_t max_new_tokens,
                                  int64_t *output_ids,
                                  size_t output_capacity) {
    if (!_impl->ready || input_ids == nullptr || output_ids == nullptr || input_count == 0 || output_capacity < input_count + max_new_tokens) {
        return 0;
    }
    std::copy(input_ids, input_ids + input_count, output_ids);
    _impl->cache_capacity = input_count + max_new_tokens;
    const size_t kv_size = _impl->config.num_key_value_heads * (_impl->config.hidden_size / _impl->config.num_attention_heads);
    const size_t cache_elements = _impl->config.num_hidden_layers * _impl->cache_capacity * kv_size;
    if (_impl->key_cache != nullptr) {
        checkCuda(cudaFree(_impl->key_cache), "cudaFree Qwen2 key cache");
        checkCuda(cudaFree(_impl->value_cache), "cudaFree Qwen2 value cache");
    }
    allocate(&_impl->key_cache, cache_elements);
    allocate(&_impl->value_cache, cache_elements);

    int64_t next = 0;
    for (size_t position = 0; position < input_count; ++position) {
        next = _impl->forward(input_ids[position], position);
    }
    size_t output_count = input_count;
    for (size_t generated = 0; generated < max_new_tokens; ++generated) {
        output_ids[output_count++] = next;
        if (next == _impl->config.eos_token_id) {
            break;
        }
        next = _impl->forward(next, input_count + generated);
    }
    return output_count;
}

} // namespace llaisys::models
