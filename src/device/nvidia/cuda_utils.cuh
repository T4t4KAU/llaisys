#pragma once

#ifdef __C
#undef __C
#endif

#ifdef LLAISYS_MUSA
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

using cudaError_t = musaError_t;
using cudaMemcpyKind = musaMemcpyKind;
using cudaStream_t = musaStream_t;
using __nv_bfloat16 = __mt_bfloat16;

#define cudaSuccess musaSuccess
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaGetDeviceCount musaGetDeviceCount
#define cudaSetDevice musaSetDevice
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaStreamCreate musaStreamCreate
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMallocHost musaMallocHost
#define cudaFreeHost musaFreeHost
#define cudaMemcpy musaMemcpy
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpyHostToHost musaMemcpyHostToHost
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

inline void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline void checkCudaLaunch(const char *operation) {
    checkCuda(cudaGetLastError(), operation);
}

template <typename T>
__device__ inline float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ inline T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

} // namespace llaisys::device::nvidia
