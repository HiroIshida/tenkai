#ifndef tenkai_f32_RRT_PLANNER_INCLUDED
#define tenkai_f32_RRT_PLANNER_INCLUDED
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.cuh"
#include "generated_model.cuh"
#include "rrtc.cuh"
#include "types.hpp"

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
  explicit DeviceBuffer(size_t count) : count_(count) {
    const size_t bytes = storage_product(count, sizeof(T));
    if (bytes) {
      cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc");
    }
  }

  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

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
  size_t count_;
};

// Scene buffers are borrowed and must outlive the planner. All search storage
// is reused across solves; a new solve invalidates the previous tree/path.
// With host ranges supplied, each environment's set of types must stay fixed.
class RrtPlanner {
 public:
  RrtPlanner(const tenkai::RrtConfig& config,
             const RrtSpace& space,
             Scene scene,
             size_t environment_count,
             size_t problem_count,
             const Environment* host_environments = nullptr)
      : config_(
            checked_config(config, space, environment_count, problem_count)),
        space_(space),
        scene_(scene),
        environment_count_(environment_count),
        problem_count_(problem_count),
        environment_types_(environment_count, AllTypes),
        problems_(problem_count),
        nodes_(storage_product(storage_product(problem_count, config.max_nodes),
                               rrt_dimension)),
        parents_(storage_product(problem_count, config.max_nodes)),
        trees_(storage_product(problem_count, config.max_nodes)),
        paths_(storage_product(problem_count, config.max_nodes)),
        results_(problem_count),
        preparation_(problem_count),
        next_problem_(1) {
    // Without host ranges, borrowed device scenes must use the generic kernel.
    if (host_environments) {
      for (size_t i = 0; i < environment_count; ++i) {
        const auto& env = host_environments[i];
        environment_types_[i] = (env.boxes.count ? BoxType : 0) |
                                (env.spheres.count ? SphereType : 0) |
                                (env.cylinders.count ? CylinderType : 0) |
                                (env.grounds.count ? GroundType : 0);
      }
    }
  }

  std::vector<tenkai::RrtResult> solve(
      const std::vector<RrtProblem>& problems) {
    if (problems.size() != problem_count_) {
      throw std::invalid_argument("RRT problem count mismatch");
    }
    unsigned types = 0;
    for (const auto& problem : problems) {
      if (problem.environment_id >= environment_count_) {
        throw std::invalid_argument("RRT environment ID out of range");
      }
      types |= environment_types_[problem.environment_id];
    }

    solved_ = false;
    problems_.upload(problems.data(), problems.size());

    tenkai_rrtc_prepare<<<
        static_cast<unsigned int>((problem_count_ - 1) / 128 + 1), 128>>>(
        problems_.data(), problem_count_, space_, scene_, preparation_.data(),
        next_problem_.data());
    cuda_check(cudaGetLastError(), "prepare RRT-Connect problems");

    if ((types & ~rrt_specialized_types) == 0) {
      launch_search<rrt_specialized_types>();
    } else {
      launch_search<AllTypes>();
    }
    cuda_check(cudaGetLastError(), "launch RRT-Connect");

    std::vector<tenkai::RrtResult> result(problem_count_);
    results_.download(result.data(), result.size());
    solved_ = true;
    return result;
  }

  std::vector<Scalar> path(size_t problem_id) const {
    if (!solved_ || problem_id >= problem_count_) {
      throw std::out_of_range("RRT path unavailable");
    }

    tenkai::RrtResult result;
    results_.download(&result, 1, problem_id);

    std::vector<uint32_t> indices(result.path_length);
    paths_.download(indices.data(), indices.size(),
                    problem_id * config_.max_nodes);

    std::vector<Scalar> nodes(size_t(result.node_count) * rrt_dimension);
    nodes_.download(nodes.data(), nodes.size(),
                    problem_id * config_.max_nodes * rrt_dimension);

    std::vector<Scalar> path(size_t(result.path_length) * rrt_dimension);
    for (size_t i = 0; i < indices.size(); ++i) {
      for (uint32_t d = 0; d < rrt_dimension; ++d) {
        path[i * rrt_dimension + d] =
            nodes[size_t(indices[i]) * rrt_dimension + d];
      }
    }

    return path;
  }

 private:
  template <unsigned Types>
  void launch_search() {
    tenkai_rrtc<Types>
        <<<static_cast<unsigned int>((problem_count_ - 1) / 4 + 1), 128>>>(
            problems_.data(), problem_count_, space_, scene_, config_,
            nodes_.data(), parents_.data(), trees_.data(), paths_.data(),
            results_.data(), preparation_.data(), next_problem_.data());
  }

  static tenkai::RrtConfig checked_config(const tenkai::RrtConfig& config,
                                          const RrtSpace& space,
                                          size_t environments,
                                          size_t problems) {
    validate_rrt(config, space);
    if (!environments || !problems) {
      throw std::invalid_argument("invalid RRT execution dimensions");
    }

    int device;
    cudaDeviceProp properties;
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device),
               "get CUDA properties");
    const void* kernels[] = {
        reinterpret_cast<const void*>(tenkai_rrtc<AllTypes>),
        reinterpret_cast<const void*>(tenkai_rrtc<rrt_specialized_types>)};
    const size_t kernel_count = rrt_specialized_types == AllTypes ? 1 : 2;
    for (size_t i = 0; i < kernel_count; ++i) {
      cudaFuncAttributes attributes;
      cuda_check(cudaFuncGetAttributes(&attributes, kernels[i]),
                 "get RRT kernel attributes");
      if (problems > size_t(properties.maxGridSize[0]) ||
          attributes.maxThreadsPerBlock < 128) {
        throw std::invalid_argument("RRT launch exceeds device/kernel limits");
      }

      int active_blocks;
      cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &active_blocks, kernels[i], 128, 0),
                 "check RRT occupancy");
      if (!active_blocks) {
        throw std::invalid_argument("RRT block cannot fit on an SM");
      }
    }

    return config;
  }

  tenkai::RrtConfig config_;
  RrtSpace space_;
  Scene scene_;
  size_t environment_count_;
  size_t problem_count_;
  std::vector<unsigned> environment_types_;
  bool solved_ = false;

  DeviceBuffer<RrtProblem> problems_;
  DeviceBuffer<Scalar> nodes_;
  DeviceBuffer<uint32_t> parents_;
  DeviceBuffer<RrtTree> trees_;
  DeviceBuffer<uint32_t> paths_;
  DeviceBuffer<tenkai::RrtResult> results_;
  DeviceBuffer<RrtPreparation> preparation_;
  DeviceBuffer<uint32_t> next_problem_;
};

}  // namespace tenkai_f32
#endif
