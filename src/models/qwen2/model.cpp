#include "model.hpp"
#ifdef ENABLE_NVIDIA_API
#include "model_nvidia.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__AVX2__)
#ifdef __C
#undef __C
#endif
#include <immintrin.h>
#endif

namespace llaisys::models {
namespace {

using bf16 = uint16_t;

inline float bf16ToFloat(bf16 value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline bf16 floatToBf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<bf16>((bits + rounding_bias) >> 16U);
}

#if defined(__AVX2__)
inline float horizontalSum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

float dotBf16(const bf16 *a, const bf16 *b, size_t count) {
    size_t i = 0;
    float result = 0.0F;
#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= count; i += 16) {
        const __m128i a0_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
        const __m128i b0_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
        const __m128i a1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i + 8));
        const __m128i b1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i + 8));

        const __m256 av0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a0_16), 16));
        const __m256 bv0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b0_16), 16));
        const __m256 av1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a1_16), 16));
        const __m256 bv1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b1_16), 16));
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, bv0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(av1, bv1));
#endif
    }
    result = horizontalSum(_mm256_add_ps(acc0, acc1));
#endif
    for (; i < count; ++i) {
        result += bf16ToFloat(a[i]) * bf16ToFloat(b[i]);
    }
    return result;
}

void linear(std::vector<bf16> &out,
            const std::vector<bf16> &in,
            const Weight &weight,
            const Weight *bias = nullptr) {
    const size_t rows = weight.shape[0];
    const size_t columns = weight.shape[1];
    out.resize(rows);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
        float value = dotBf16(in.data(), weight.data.data() + static_cast<size_t>(row) * columns, columns);
        if (bias != nullptr) {
            value += bf16ToFloat(bias->data[static_cast<size_t>(row)]);
        }
        out[static_cast<size_t>(row)] = floatToBf16(value);
    }
}

int64_t linearArgmax(const std::vector<bf16> &in, const Weight &weight) {
    const size_t rows = weight.shape[0];
    const size_t columns = weight.shape[1];
    float global_max = -std::numeric_limits<float>::infinity();
    int64_t global_index = 0;

#pragma omp parallel
    {
        float local_max = -std::numeric_limits<float>::infinity();
        int64_t local_index = 0;
#pragma omp for nowait schedule(static)
        for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
            const float value = bf16ToFloat(floatToBf16(dotBf16(
                in.data(),
                weight.data.data() + static_cast<size_t>(row) * columns,
                columns)));
            if (value > local_max) {
                local_max = value;
                local_index = row;
            }
        }
#pragma omp critical
        {
            if (local_max > global_max || (local_max == global_max && local_index < global_index)) {
                global_max = local_max;
                global_index = local_index;
            }
        }
    }
    return global_index;
}

void rmsNorm(std::vector<bf16> &out,
             const std::vector<bf16> &in,
             const Weight &weight,
             float eps) {
    float sum = 0.0F;
    for (bf16 value : in) {
        const float converted = bf16ToFloat(value);
        sum += converted * converted;
    }
    const float scale = 1.0F / std::sqrt(sum / static_cast<float>(in.size()) + eps);
    out.resize(in.size());
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(in.size()); ++i) {
        const bf16 normalized = floatToBf16(bf16ToFloat(in[static_cast<size_t>(i)]) * scale);
        out[static_cast<size_t>(i)] = floatToBf16(
            bf16ToFloat(normalized) * bf16ToFloat(weight.data[static_cast<size_t>(i)]));
    }
}

void addInPlace(std::vector<bf16> &left, const std::vector<bf16> &right) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(left.size()); ++i) {
        const size_t index = static_cast<size_t>(i);
        left[index] = floatToBf16(bf16ToFloat(left[index]) + bf16ToFloat(right[index]));
    }
}

void swiglu(std::vector<bf16> &out,
            const std::vector<bf16> &gate,
            const std::vector<bf16> &up) {
    out.resize(gate.size());
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(gate.size()); ++i) {
        const size_t index = static_cast<size_t>(i);
        const float gate_value = bf16ToFloat(gate[index]);
        const bf16 activated = floatToBf16(gate_value / (1.0F + std::exp(-gate_value)));
        out[index] = floatToBf16(bf16ToFloat(activated) * bf16ToFloat(up[index]));
    }
}

} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Config &config)
    : _config(config), _layers(config.num_hidden_layers), _cache(config.num_hidden_layers) {
    if (_config.device_type != LLAISYS_DEVICE_CPU
#ifdef ENABLE_NVIDIA_API
        && _config.device_type != LLAISYS_DEVICE_NVIDIA
#endif
    ) {
        throw std::invalid_argument("Qwen2 CPU model only supports the CPU device in this build");
    }
    if (_config.hidden_size == 0 || _config.num_attention_heads == 0 || _config.hidden_size % _config.num_attention_heads != 0 || _config.num_attention_heads % _config.num_key_value_heads != 0) {
        throw std::invalid_argument("Invalid Qwen2 configuration");
    }

#ifdef ENABLE_NVIDIA_API
    if (_config.device_type == LLAISYS_DEVICE_NVIDIA) {
        _nvidia = std::make_unique<Qwen2NvidiaModel>(_config);
        return;
    }
#endif

    _hidden.resize(_config.hidden_size);
    _norm.resize(_config.hidden_size);
    _q.resize(_config.hidden_size);
    const size_t kv_size = _config.num_key_value_heads * (_config.hidden_size / _config.num_attention_heads);
    _k.resize(kv_size);
    _v.resize(kv_size);
    _attention.resize(_config.hidden_size);
    _projection.resize(_config.hidden_size);
    _gate.resize(_config.intermediate_size);
    _up.resize(_config.intermediate_size);
    _mlp.resize(_config.intermediate_size);
}

Qwen2Model::~Qwen2Model() = default;

Weight *Qwen2Model::findWeight(const std::string &name) {
    if (name == "model.embed_tokens.weight") {
        return &_embedding;
    }
    if (name == "model.norm.weight") {
        return &_final_norm;
    }
    if (name == "lm_head.weight") {
        return &_lm_head;
    }

    constexpr const char *prefix = "model.layers.";
    if (name.compare(0, std::strlen(prefix), prefix) != 0) {
        return nullptr;
    }
    const size_t layer_begin = std::strlen(prefix);
    const size_t dot = name.find('.', layer_begin);
    if (dot == std::string::npos) {
        return nullptr;
    }
    size_t layer_index = 0;
    try {
        layer_index = static_cast<size_t>(std::stoul(name.substr(layer_begin, dot - layer_begin)));
    } catch (const std::exception &) {
        return nullptr;
    }
    if (layer_index >= _layers.size()) {
        return nullptr;
    }

    Qwen2LayerWeights &layer = _layers[layer_index];
    const std::string suffix = name.substr(dot + 1);
    if (suffix == "input_layernorm.weight") {
        return &layer.input_norm;
    }
    if (suffix == "post_attention_layernorm.weight") {
        return &layer.post_attention_norm;
    }
    if (suffix == "self_attn.q_proj.weight") {
        return &layer.q_weight;
    }
    if (suffix == "self_attn.q_proj.bias") {
        return &layer.q_bias;
    }
    if (suffix == "self_attn.k_proj.weight") {
        return &layer.k_weight;
    }
    if (suffix == "self_attn.k_proj.bias") {
        return &layer.k_bias;
    }
    if (suffix == "self_attn.v_proj.weight") {
        return &layer.v_weight;
    }
    if (suffix == "self_attn.v_proj.bias") {
        return &layer.v_bias;
    }
    if (suffix == "self_attn.o_proj.weight") {
        return &layer.o_weight;
    }
    if (suffix == "mlp.gate_proj.weight") {
        return &layer.gate_weight;
    }
    if (suffix == "mlp.up_proj.weight") {
        return &layer.up_weight;
    }
    if (suffix == "mlp.down_proj.weight") {
        return &layer.down_weight;
    }
    return nullptr;
}

bool Qwen2Model::loadWeight(const std::string &name,
                            const void *data,
                            const size_t *shape,
                            size_t ndim,
                            llaisysDataType_t dtype) {
    if (_ready || data == nullptr || dtype != LLAISYS_DTYPE_BF16) {
        return false;
    }
#ifdef ENABLE_NVIDIA_API
    if (_nvidia) {
        return _nvidia->loadWeight(name, data, shape, ndim, dtype);
    }
#endif
    Weight *weight = findWeight(name);
    if (weight == nullptr) {
        return false;
    }
    size_t elements = 1;
    weight->shape.assign(shape, shape + ndim);
    for (size_t dimension : weight->shape) {
        elements *= dimension;
    }
    weight->data.resize(elements);
    std::memcpy(weight->data.data(), data, elements * sizeof(bf16));
    return true;
}

bool Qwen2Model::finalize() {
#ifdef ENABLE_NVIDIA_API
    if (_nvidia) {
        _ready = _nvidia->finalize();
        return _ready;
    }
#endif
    bool complete = _embedding.loaded() && _final_norm.loaded() && _lm_head.loaded();
    for (const Qwen2LayerWeights &layer : _layers) {
        complete = complete && layer.input_norm.loaded() && layer.post_attention_norm.loaded() && layer.q_weight.loaded() && layer.q_bias.loaded() && layer.k_weight.loaded() && layer.k_bias.loaded() && layer.v_weight.loaded() && layer.v_bias.loaded() && layer.o_weight.loaded() && layer.gate_weight.loaded() && layer.up_weight.loaded() && layer.down_weight.loaded();
    }
    _ready = complete;
    return complete;
}

int64_t Qwen2Model::forwardToken(int64_t token, size_t position) {
    const size_t hidden_size = _config.hidden_size;
    const size_t head_count = _config.num_attention_heads;
    const size_t kv_head_count = _config.num_key_value_heads;
    const size_t head_size = hidden_size / head_count;
    const size_t kv_size = kv_head_count * head_size;
    const size_t heads_per_kv = head_count / kv_head_count;

    const bf16 *embedding = _embedding.data.data() + static_cast<size_t>(token) * hidden_size;
    std::copy(embedding, embedding + hidden_size, _hidden.begin());

    std::vector<bf16> rope_cos(head_size / 2);
    std::vector<bf16> rope_sin(head_size / 2);
    for (size_t i = 0; i < head_size / 2; ++i) {
        const float frequency = std::pow(
            _config.rope_theta,
            -2.0F * static_cast<float>(i) / static_cast<float>(head_size));
        const float angle = static_cast<float>(position) * frequency;
        rope_cos[i] = floatToBf16(std::cos(angle));
        rope_sin[i] = floatToBf16(std::sin(angle));
    }

    for (size_t layer_index = 0; layer_index < _layers.size(); ++layer_index) {
        const Qwen2LayerWeights &layer = _layers[layer_index];
        rmsNorm(_norm, _hidden, layer.input_norm, _config.rms_norm_eps);
        linear(_q, _norm, layer.q_weight, &layer.q_bias);
        linear(_k, _norm, layer.k_weight, &layer.k_bias);
        linear(_v, _norm, layer.v_weight, &layer.v_bias);

        auto rotate = [&](std::vector<bf16> &values, size_t number_of_heads) {
            for (size_t head = 0; head < number_of_heads; ++head) {
                bf16 *vector = values.data() + head * head_size;
                for (size_t i = 0; i < head_size / 2; ++i) {
                    const float first = bf16ToFloat(vector[i]);
                    const float second = bf16ToFloat(vector[i + head_size / 2]);
                    const float cosine = bf16ToFloat(rope_cos[i]);
                    const float sine = bf16ToFloat(rope_sin[i]);
                    const bf16 first_product = floatToBf16(first * cosine);
                    const bf16 second_product = floatToBf16(second * sine);
                    const bf16 third_product = floatToBf16(second * cosine);
                    const bf16 fourth_product = floatToBf16(first * sine);
                    vector[i] = floatToBf16(bf16ToFloat(first_product) - bf16ToFloat(second_product));
                    vector[i + head_size / 2] = floatToBf16(bf16ToFloat(third_product) + bf16ToFloat(fourth_product));
                }
            }
        };
        rotate(_q, head_count);
        rotate(_k, kv_head_count);

        LayerCache &cache = _cache[layer_index];
        std::copy(_k.begin(), _k.end(), cache.key.begin() + static_cast<ptrdiff_t>(position * kv_size));
        std::copy(_v.begin(), _v.end(), cache.value.begin() + static_cast<ptrdiff_t>(position * kv_size));

        const size_t context = position + 1;
        _scores.resize(context);
        const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));
        for (size_t head = 0; head < head_count; ++head) {
            const size_t kv_head = head / heads_per_kv;
            const bf16 *query = _q.data() + head * head_size;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t sequence = 0; sequence < context; ++sequence) {
                const bf16 *key = cache.key.data() + sequence * kv_size + kv_head * head_size;
                const bf16 dot = floatToBf16(dotBf16(query, key, head_size));
                const bf16 scaled = floatToBf16(bf16ToFloat(dot) * attention_scale);
                _scores[sequence] = bf16ToFloat(scaled);
                max_score = std::max(max_score, _scores[sequence]);
            }
            float score_sum = 0.0F;
            for (float &score : _scores) {
                score = std::exp(score - max_score);
                score_sum += score;
            }
            for (float &score : _scores) {
                score = bf16ToFloat(floatToBf16(score / score_sum));
            }
            bf16 *attention_head = _attention.data() + head * head_size;
            for (size_t dimension = 0; dimension < head_size; ++dimension) {
                float value = 0.0F;
                for (size_t sequence = 0; sequence < context; ++sequence) {
                    const bf16 cached_value = cache.value[sequence * kv_size + kv_head * head_size + dimension];
                    value += _scores[sequence] * bf16ToFloat(cached_value);
                }
                attention_head[dimension] = floatToBf16(value);
            }
        }

        linear(_projection, _attention, layer.o_weight);
        addInPlace(_hidden, _projection);
        rmsNorm(_norm, _hidden, layer.post_attention_norm, _config.rms_norm_eps);
        linear(_gate, _norm, layer.gate_weight);
        linear(_up, _norm, layer.up_weight);
        swiglu(_mlp, _gate, _up);
        linear(_projection, _mlp, layer.down_weight);
        addInPlace(_hidden, _projection);
    }

    rmsNorm(_norm, _hidden, _final_norm, _config.rms_norm_eps);
    return linearArgmax(_norm, _lm_head);
}

size_t Qwen2Model::generate(const int64_t *input_ids,
                            size_t input_count,
                            size_t max_new_tokens,
                            int64_t *output_ids,
                            size_t output_capacity) {
    if (!_ready || input_ids == nullptr || output_ids == nullptr || input_count == 0 || output_capacity < input_count + max_new_tokens) {
        return 0;
    }
#ifdef ENABLE_NVIDIA_API
    if (_nvidia) {
        return _nvidia->generate(
            input_ids, input_count, max_new_tokens, output_ids, output_capacity);
    }
#endif
    std::copy(input_ids, input_ids + input_count, output_ids);
    const size_t capacity = input_count + max_new_tokens;
    const size_t kv_size = _config.num_key_value_heads * (_config.hidden_size / _config.num_attention_heads);
    for (LayerCache &cache : _cache) {
        cache.key.assign(capacity * kv_size, 0);
        cache.value.assign(capacity * kv_size, 0);
    }

    int64_t next_token = 0;
    for (size_t position = 0; position < input_count; ++position) {
        next_token = forwardToken(input_ids[position], position);
    }

    size_t output_count = input_count;
    for (size_t generated = 0; generated < max_new_tokens; ++generated) {
        output_ids[output_count++] = next_token;
        if (next_token == _config.eos_token_id) {
            break;
        }
        next_token = forwardToken(next_token, input_count + generated);
    }
    return output_count;
}

} // namespace llaisys::models
