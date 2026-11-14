#pragma once
#include <hip/hip_runtime.h>
#include <stdexcept>

// Keep the frozen CUDA sources shared by both compilers.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaDeviceProp hipDeviceProp_t
#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
  hipOccupancyMaxActiveBlocksPerMultiprocessor

inline hipError_t cudaGetDeviceProperties(hipDeviceProp_t* properties,
                                          int device) {
  const hipError_t error = hipGetDeviceProperties(properties, device);

  // The planner's groups, masks and shared state assume exactly 32 lanes.
  if (error == hipSuccess && properties->warpSize != 32) {
    throw std::invalid_argument(
        "tenkai HIP planners require a wave32 GPU (such as RX 9070 XT)");
  }

  return error;
}

// HIP requires 64-bit masks even on wave32; widening is valid only with that
// restriction.
__device__ inline void __syncwarp(unsigned mask) {
  __syncwarp(static_cast<unsigned long long>(mask));
}

__device__ inline int __any_sync(unsigned mask, int predicate) {
  return __any_sync(static_cast<unsigned long long>(mask), predicate);
}

__device__ inline unsigned __ballot_sync(unsigned mask, int predicate) {
  return static_cast<unsigned>(
      __ballot_sync(static_cast<unsigned long long>(mask), predicate));
}

template <typename T>
__device__ inline T __shfl_sync(unsigned mask,
                                T value,
                                int source,
                                int width = 32) {
  return __shfl_sync(static_cast<unsigned long long>(mask), value, source,
                     width);
}

template <typename T>
__device__ inline T __shfl_down_sync(unsigned mask,
                                     T value,
                                     unsigned delta,
                                     int width = 32) {
  return __shfl_down_sync(static_cast<unsigned long long>(mask), value, delta,
                          width);
}
