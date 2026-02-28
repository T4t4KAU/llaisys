#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    ASSERT(out->ndim() == 3 && in->ndim() == 3, "Rope: out/in must be 3-D [seq_len, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "Rope: pos_ids must be 1-D [seq_len].");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1] && out->shape()[2] == in->shape()[2], "Rope: out shape must match in shape.");
    ASSERT(pos_ids->shape()[0] == out->shape()[0], "Rope: pos_ids length must equal seq_len.");
    ASSERT((out->shape()[2] % 2) == 0, "Rope: d must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Rope: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->shape()[0], out->shape()[1], out->shape()[2]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->shape()[0], out->shape()[1], out->shape()[2]);
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
