#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "../musa/ops_musa.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2-D.");
    ASSERT(index->ndim() == 1, "Embedding: index must be 1-D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2-D.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    ASSERT(out->shape()[0] == index->numel(), "Embedding: out.shape[0] must equal index.numel().");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out.shape[1] must equal weight.shape[1].");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1] * weight->elementSize());
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1] * weight->elementSize());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(), index->data(), weight->data(), weight->dtype(),
            index->numel(), weight->shape()[1]);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
