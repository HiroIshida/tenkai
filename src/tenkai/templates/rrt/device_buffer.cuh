#ifndef tenkai_f32_DEVICE_BUFFER_INCLUDED
#define tenkai_f32_DEVICE_BUFFER_INCLUDED
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace tenkai_f32 {
inline void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

inline size_t storage_product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b) {
    throw std::length_error("RRT storage size overflow");
  }

  return a * b;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count = 0) { reserve(count); }

  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Scratch allocations can grow without retaining their old contents.
  void reserve(size_t count) {
    if (count <= count_) {
      return;
    }
    const size_t bytes = storage_product(count, sizeof(T));
    T* replacement = nullptr;
    cuda_check(cudaMalloc(&replacement, bytes), "cudaMalloc");
    if (data_) {
      cudaFree(data_);
    }
    data_ = replacement;
    count_ = count;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }

  void upload(const T* input, size_t count) {
    if (count > count_) {
      throw std::out_of_range("device upload exceeds buffer");
    }

    if (count) {
      cuda_check(
          cudaMemcpy(data_, input, count * sizeof(T), cudaMemcpyHostToDevice),
          "upload");
    }
  }

  void download(T* output, size_t count, size_t offset = 0) const {
    if (offset > count_ || count > count_ - offset) {
      throw std::out_of_range("device download exceeds buffer");
    }

    if (count) {
      cuda_check(cudaMemcpy(output, data_ + offset, count * sizeof(T),
                            cudaMemcpyDeviceToHost),
                 "download");
    }
  }

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

}  // namespace tenkai_f32
#endif
