#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->shape().size() == 3,
           "SelfAttention: q must be 3D tensor [seqlen, nhead, d]");
    ASSERT(k->shape().size() == 3,
           "SelfAttention: k must be 3D tensor [total_len, nkvhead, d]");
    ASSERT(v->shape().size() == 3,
           "SelfAttention: v must be 3D tensor [total_len, nkvhead, dv]");
    ASSERT(attn_val->shape().size() == 3,
           "SelfAttention: attn_val must be 3D tensor [seqlen, nhead, dv]");

    ASSERT(q->shape()[0] == attn_val->shape()[0],
           "SelfAttention: seqlen mismatch between q and attn_val");

    ASSERT(q->shape()[1] == attn_val->shape()[1],
           "SelfAttention: nhead mismatch between q and attn_val");

    ASSERT(q->shape()[2] == k->shape()[2],
           "SelfAttention: head dimension d mismatch between q and k");

    ASSERT(k->shape()[0] == v->shape()[0],
           "SelfAttention: total_len mismatch between k and v");

    ASSERT(k->shape()[1] == v->shape()[1],
           "SelfAttention: nkvhead mismatch between k and v");

    ASSERT(v->shape()[2] == attn_val->shape()[2],
           "SelfAttention: value dimension dv mismatch between v and attn_val");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), q->shape()[0], k->shape()[0],
                                   q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2], scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), q->shape()[0], k->shape()[0],
                                   q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2], scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
