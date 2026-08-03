#include "../runtime_api.hpp"

#include "cuda_utils.cuh"

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace runtime_api {
namespace {
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

cudaMemcpyKind convertMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Invalid CUDA memcpy kind");
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
}

void deviceSynchronize() {
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    check(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *pointer = nullptr;
    check(cudaMalloc(&pointer, size), "cudaMalloc");
    return pointer;
}

void freeDevice(void *ptr) {
    check(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    void *pointer = nullptr;
    check(cudaMallocHost(&pointer, size), "cudaMallocHost");
    return pointer;
}

void freeHost(void *ptr) {
    check(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check(cudaMemcpy(dst, src, size, convertMemcpyKind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst,
                 const void *src,
                 size_t size,
                 llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    check(cudaMemcpyAsync(
              dst,
              src,
              size,
              convertMemcpyKind(kind),
              reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
