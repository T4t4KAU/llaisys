#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    ASSERT(in->ndim() == 2 && out->ndim() == 2, "RMSNorm: in/out must be 2-D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1-D.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1],
           "RMSNorm: out shape must match in shape.");
    ASSERT(weight->shape()[0] == in->shape()[1],
           "RMSNorm: weight length must equal D.");
    ASSERT(eps > 0.0f, "RMSNorm: eps must be > 0.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in->shape()[0], in->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in->shape()[0], in->shape()[1]);
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
