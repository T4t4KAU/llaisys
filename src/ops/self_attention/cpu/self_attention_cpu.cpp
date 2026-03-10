#include "self_attention_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../../../utils.hpp"

// Layout (contiguous):
// Q:   [seqlen, nhead, d]
// K:   [total_len, nkvhead, d]
// V:   [total_len, nkvhead, dv]
// out: [seqlen, nhead, dv]
//
// Causal rule: query index t can attend to keys up to (past_len + t), inclusive.
// Typically total_len == past_len + seqlen.
//
// GQA/MQA mapping: kvh = h % nkvhead (when nkvhead != nhead).

template <typename T>
void self_attention_(
    T *attn_val,    // [L, nhead, dv]
    const T *query, // [L, nhead, d]
    const T *key,   // [S, nkvhead, d]
    const T *value, // [S, nkvhead, dv]
    size_t L,       // seqlen
    size_t S,       // total_len
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale) {

    if (L <= 0 || S <= 0 || nhead <= 0 || nkvhead <= 0 || d <= 0 || dv <= 0) {
        return;
    }
    if (nhead % nkvhead != 0) {
        return;
    }
    const size_t group = nhead / nkvhead;

    // torch mask: tril(diagonal=S-L)
    // allow keys j <= i + (S - L)
    const size_t shift = S - L; // can be 0 or positive in typical cache case

    for (size_t i = 0; i < L; ++i) { // query position
        size_t j_max = i + shift;    // inclusive
        if (j_max < 0) {
            continue;
        }
        if (j_max >= S) {
            j_max = S - 1;
        }
        const size_t n_keys = j_max + 1;

        for (size_t h = 0; h < nhead; ++h) { // head
            const size_t kvh = h / group;    // repeat_interleave mapping

            const T *q_ptr = query + (static_cast<size_t>(i) * nhead + h) * d;
            T *out_ptr = attn_val + (static_cast<size_t>(i) * nhead + h) * dv;

            // 1) compute logits for allowed keys (masked others are -inf)
            float max_logit = -std::numeric_limits<float>::infinity();
            std::vector<float> exps(static_cast<size_t>(n_keys));

            for (size_t j = 0; j < n_keys; ++j) {
                const T *k_ptr = key + (static_cast<size_t>(j) * nkvhead + kvh) * d;

                float dot = 0.0f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t t = 0; t < d; ++t) {
                        dot += llaisys::utils::cast<float>(q_ptr[t]) * llaisys::utils::cast<float>(k_ptr[t]);
                    }
                } else {
                    for (size_t t = 0; t < d; ++t) {
                        dot += static_cast<float>(q_ptr[t]) * static_cast<float>(k_ptr[t]);
                    }
                }

                float logit = dot * scale;
                exps[static_cast<size_t>(j)] = logit;
                if (logit > max_logit) {
                    max_logit = logit;
                }
            }

            // 2) softmax over n_keys
            float sum_exp = 0.0f;
            for (size_t j = 0; j < n_keys; ++j) {
                float e = std::exp(exps[static_cast<size_t>(j)] - max_logit);
                exps[static_cast<size_t>(j)] = e; // reuse buffer to store exp
                sum_exp += e;
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            // 3) weighted sum of V
            // accumulate in float
            std::vector<float> acc(static_cast<size_t>(dv), 0.0f);

            for (size_t j = 0; j < n_keys; ++j) {
                float w = exps[static_cast<size_t>(j)] * inv_sum;
                const T *v_ptr = value + (static_cast<size_t>(j) * nkvhead + kvh) * dv;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t t = 0; t < dv; ++t) {
                        acc[static_cast<size_t>(t)] += w * llaisys::utils::cast<float>(v_ptr[t]);
                    }
                } else {
                    for (size_t t = 0; t < dv; ++t) {
                        acc[static_cast<size_t>(t)] += w * static_cast<float>(v_ptr[t]);
                    }
                }
            }

            // write back
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for (size_t t = 0; t < dv; ++t) {
                    out_ptr[t] = llaisys::utils::cast<T>(acc[static_cast<size_t>(t)]);
                }
            } else {
                for (size_t t = 0; t < dv; ++t) {
                    out_ptr[t] = static_cast<T>(acc[static_cast<size_t>(t)]);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(
    std::byte *attn_val,
    const std::byte *Q, const std::byte *K, const std::byte *V, llaisysDataType_t type,
    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(Q),
                               reinterpret_cast<const float *>(K),
                               reinterpret_cast<const float *>(V),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(Q),
                               reinterpret_cast<const llaisys::bf16_t *>(K),
                               reinterpret_cast<const llaisys::bf16_t *>(V),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(Q),
                               reinterpret_cast<const llaisys::fp16_t *>(K),
                               reinterpret_cast<const llaisys::fp16_t *>(V),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu