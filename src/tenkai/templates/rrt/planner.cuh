#ifndef tenkai_f32_RRT_PLANNER_INCLUDED
#define tenkai_f32_RRT_PLANNER_INCLUDED
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common.cuh"
#include "generated_model.cuh"
#include "path_batch.cuh"
#include "rrtc.cuh"
#include "types.hpp"

namespace tenkai_f32 {
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
             const Environment* host_environments = nullptr,
             size_t batch_size = 0)
      : config_(
            checked_config(config, space, environment_count, problem_count)),
        space_(space),
        scene_(scene),
        problem_count_(problem_count),
        worker_count_(batch_size ? std::min(batch_size, problem_count)
                                 : problem_count),
        environment_types_(environment_count, AllTypes),
        problems_(problem_count),
        nodes_(storage_product(storage_product(worker_count_, config.max_nodes),
                               rrt_dimension)),
        parents_(storage_product(worker_count_, config.max_nodes)),
        trees_(storage_product(worker_count_, config.max_nodes)),
        paths_(storage_product(worker_count_, config.max_nodes)),
        results_(problem_count),
        preparation_(problem_count),
        next_problem_(1),
        coordinate_capacity_(storage_product(problem_count, 2)),
        coordinates_(storage_product(coordinate_capacity_, rrt_dimension)),
        offsets_(problem_count),
        next_vertex_(1) {
    // Without host ranges, borrowed device scenes must use the generic kernel.
    if (host_environments) {
      for (size_t i = 0; i < environment_count; ++i) {
        const auto& env = host_environments[i];
        environment_types_[i] = (env.boxes.count ? BoxType : 0) |
                                (env.spheres.count ? SphereType : 0) |
                                (env.cylinders.count ? CylinderType : 0);
      }
    }
  }

  std::vector<tenkai::RrtResult> solve(
      const std::vector<RrtProblem>& problems,
      const tenkai::ShortcutConfig* shortcut = nullptr) {
    unsigned types = 0;
    for (const auto& problem : problems) {
      types |= environment_types_[problem.environment_id];
    }

    solved_ = false;
    problems_.upload(problems.data(), problems.size());

    for (;;) {
      tenkai_rrtc_prepare<<<
          static_cast<unsigned int>((problem_count_ - 1) / 128 + 1), 128>>>(
          problems_.data(), problem_count_, space_, scene_, preparation_.data(),
          next_problem_.data(), next_vertex_.data());
      cuda_check(cudaGetLastError(), "prepare RRT-Connect problems");

      if ((types & ~rrt_specialized_types) == 0) {
        launch_search<rrt_specialized_types>();
      } else {
        launch_search<AllTypes>();
      }
      cuda_check(cudaGetLastError(), "launch RRT-Connect");

      unsigned long long vertices;
      next_vertex_.download(&vertices, 1);
      if (vertices <= coordinate_capacity_) {
        break;
      }
      // Replay deterministic searches only when their packed paths outgrow
      // storage.
      coordinates_.reserve(storage_product(vertices, rrt_dimension));
      coordinate_capacity_ = vertices;
    }

    std::vector<tenkai::RrtResult> result(problem_count_);
    results_.download(result.data(), result.size());
    path_batch_.build(problems, result, coordinates_.data(), offsets_.data(),
                      shortcut, space_, scene_);
    solved_ = true;
    return result;
  }

  std::vector<Scalar> path(size_t problem_id) const {
    if (!solved_) {
      throw std::out_of_range("RRT path unavailable");
    }

    return path_batch_.path(problem_id);
  }

  std::vector<Scalar> paths() const {
    require_solved();
    return path_batch_.paths();
  }

  size_t path_vertex_count() const {
    require_solved();
    return path_batch_.length();
  }

 private:
  void require_solved() const {
    if (!solved_) {
      throw std::out_of_range("RRT path unavailable; call solve first");
    }
  }

  template <unsigned Types>
  void launch_search() {
    tenkai_rrtc<Types>
        <<<static_cast<unsigned int>((worker_count_ - 1) / 4 + 1), 128>>>(
            problems_.data(), problem_count_, worker_count_, space_, scene_,
            config_, nodes_.data(), parents_.data(), trees_.data(),
            paths_.data(), results_.data(), preparation_.data(),
            next_problem_.data(), coordinates_.data(), coordinate_capacity_,
            offsets_.data(), next_vertex_.data());
  }

  static tenkai::RrtConfig checked_config(const tenkai::RrtConfig& config,
                                          const RrtSpace& space,
                                          size_t environments,
                                          size_t problems) {
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
  size_t problem_count_;
  size_t worker_count_;
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
  size_t coordinate_capacity_;
  DeviceBuffer<Scalar> coordinates_;
  DeviceBuffer<unsigned long long> offsets_;
  DeviceBuffer<unsigned long long> next_vertex_;
  PathBatch path_batch_;
};

}  // namespace tenkai_f32
#endif
