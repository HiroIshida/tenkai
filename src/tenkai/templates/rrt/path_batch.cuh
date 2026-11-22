#ifndef tenkai_f32_PATH_BATCH_INCLUDED
#define tenkai_f32_PATH_BATCH_INCLUDED
#include <vector>

#include "device_buffer.cuh"
#include "shortcut.cuh"

namespace tenkai_f32 {
__global__ void tenkai_gather_paths(const Scalar* nodes,
                                    const uint32_t* indices,
                                    uint32_t max_nodes,
                                    const ShortcutProblem* paths,
                                    Scalar* coordinates) {
  const size_t p = blockIdx.x;
  const ShortcutProblem path = paths[p];
  for (size_t i = threadIdx.x; i < size_t(path.length) * rrt_dimension;
       i += blockDim.x) {
    const uint32_t node = indices[p * max_nodes + i / rrt_dimension];
    coordinates[path.offset * rrt_dimension + i] =
        nodes[(p * max_nodes + node) * rrt_dimension + i % rrt_dimension];
  }
}

class PathBatch {
 public:
  void build(const std::vector<RrtProblem>& problems,
             std::vector<tenkai::RrtResult>& results,
             const Scalar* nodes,
             const uint32_t* indices,
             uint32_t max_nodes,
             const tenkai::ShortcutConfig* shortcut,
             const RrtSpace& space,
             Scene scene) {
    const bool shorten =
        shortcut && shortcut->max_steps && shortcut->max_empty_steps;
    paths_.resize(problems.size());
    vertices_ = 0;
    for (size_t p = 0; p < paths_.size(); ++p) {
      const uint32_t length = results[p].path_length;
      uint32_t steps = 0;
      if (shorten && length >= 3) {
        steps =
            shortcut->max_steps == UINT32_MAX ? length : shortcut->max_steps;
      }
      const uint64_t capacity = uint64_t(length) + steps;
      if (capacity > UINT32_MAX ||
          vertices_ > std::numeric_limits<size_t>::max() - capacity) {
        throw std::length_error("shortcut path capacity overflow");
      }
      paths_[p] = {vertices_, length, problems[p].environment_id,
                   problems[p].seed};
      vertices_ += size_t(capacity);
    }

    // Every solve rebuilds coordinates from the tree, so buffer growth needs no
    // repacking.
    coordinates_.reserve(storage_product(vertices_, rrt_dimension));
    device_paths_.reserve(paths_.size());
    device_paths_.upload(paths_.data(), paths_.size());
    if (!vertices_) {
      return;
    }
    tenkai_gather_paths<<<static_cast<unsigned int>(paths_.size()), 128>>>(
        nodes, indices, max_nodes, device_paths_.data(), coordinates_.data());
    cuda_check(cudaGetLastError(), "gather RRT paths");

    if (shorten) {
      distances_.reserve(vertices_);
      device_lengths_.reserve(paths_.size());
      tenkai_shortcut<<<static_cast<unsigned int>(paths_.size()), 128>>>(
          coordinates_.data(), distances_.data(), device_paths_.data(), space,
          scene, *shortcut, device_lengths_.data());
      cuda_check(cudaGetLastError(), "shortcut RRT paths");
      std::vector<uint32_t> lengths(paths_.size());
      device_lengths_.download(lengths.data(), lengths.size());
      for (size_t p = 0; p < paths_.size(); ++p) {
        paths_[p].length = results[p].path_length = lengths[p];
      }
    } else {
      cuda_check(cudaStreamSynchronize(0), "finish gathering RRT paths");
    }
  }

  std::vector<Scalar> path(size_t problem) const {
    const auto& path = paths_.at(problem);
    std::vector<Scalar> result(size_t(path.length) * rrt_dimension);
    coordinates_.download(result.data(), result.size(),
                          path.offset * rrt_dimension);
    return result;
  }

  size_t length() const {
    size_t count = 0;
    for (const auto& path : paths_) {
      count += path.length;
    }
    return count;
  }

  std::vector<Scalar> paths() const {
    std::vector<Scalar> stored(storage_product(vertices_, rrt_dimension));
    coordinates_.download(stored.data(), stored.size());

    std::vector<Scalar> result;
    result.reserve(storage_product(length(), rrt_dimension));
    for (const auto& path : paths_) {
      const size_t begin = path.offset * rrt_dimension;
      const size_t size = size_t(path.length) * rrt_dimension;
      result.insert(result.end(), stored.begin() + begin,
                    stored.begin() + begin + size);
    }
    return result;
  }

 private:
  size_t vertices_ = 0;
  std::vector<ShortcutProblem> paths_;
  DeviceBuffer<Scalar> coordinates_;
  DeviceBuffer<ShortcutProblem> device_paths_;
  DeviceBuffer<double> distances_;
  DeviceBuffer<uint32_t> device_lengths_;
};
}  // namespace tenkai_f32
#endif
