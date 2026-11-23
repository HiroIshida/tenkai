#ifndef tenkai_f32_SHORTCUT_INCLUDED
#define tenkai_f32_SHORTCUT_INCLUDED
#include "common.cuh"

namespace tenkai {
struct ShortcutConfig {
  // UINT32_MAX selects the input vertex count independently for each path.
  uint32_t max_steps = UINT32_MAX;
  uint32_t max_empty_steps = UINT32_MAX;
  double range_ratio = 0.33;
  double snap_to_vertex = 0.005;
};
}  // namespace tenkai

namespace tenkai_f32 {
struct ShortcutProblem {
  size_t offset;
  uint32_t length;
  uint32_t environment_id;
  uint64_t seed;
};

struct ShortcutPoint {
  Scalar q[rrt_dimension];
  uint32_t left;
  uint32_t right;
};

__device__ inline double shortcut_distance(const Scalar* a, const Scalar* b) {
  double squared = 0;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    const double delta = double(b[d]) - double(a[d]);
    squared += delta * delta;
  }
  return sqrt(squared);
}

__device__ inline void shortcut_distances(const Scalar* path,
                                          uint32_t length,
                                          double* distances) {
  distances[0] = 0;
  for (uint32_t i = 1; i < length; ++i) {
    distances[i] = distances[i - 1] +
                   shortcut_distance(path + size_t(i - 1) * rrt_dimension,
                                     path + size_t(i) * rrt_dimension);
  }
}

__device__ inline ShortcutPoint shortcut_point(const Scalar* path,
                                               const double* distances,
                                               uint32_t length,
                                               double position,
                                               double snap_distance) {
  uint32_t begin = 0;
  uint32_t end = length;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    if (distances[middle] <= position) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }

  ShortcutPoint point;
  point.left = begin ? begin - 1 : 0;
  point.right = point.left + (point.left + 1 < length);
  const double before = position - distances[point.left];
  const double after = distances[point.right] - position;
  if (before == 0 || (before <= after && before < snap_distance)) {
    point.right = point.left;
  } else if (after == 0 || after < snap_distance) {
    point.left = point.right;
  }

  const Scalar* a = path + size_t(point.left) * rrt_dimension;
  const Scalar* b = path + size_t(point.right) * rrt_dimension;
  const double ratio =
      point.left == point.right
          ? 0
          : (position - distances[point.left]) /
                (distances[point.right] - distances[point.left]);
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    point.q[d] = Scalar(double(a[d]) + ratio * (double(b[d]) - double(a[d])));
  }
  return point;
}

__device__ inline uint32_t shortcut_splice(Scalar* path,
                                           uint32_t length,
                                           const ShortcutPoint& a,
                                           const ShortcutPoint& b) {
  const uint32_t insert_a = a.left != a.right;
  const uint32_t insert_b = b.left != b.right;
  const uint32_t destination = a.left + 1 + insert_a + insert_b;
  const size_t tail_size = size_t(length - b.right) * rrt_dimension;
  Scalar* tail = path + size_t(b.right) * rrt_dimension;
  Scalar* output = path + size_t(destination) * rrt_dimension;
  // Adjacent-edge interior samples can add a vertex, so the tail may move
  // right.
  if (destination > b.right) {
    for (size_t i = tail_size; i > 0; --i) {
      output[i - 1] = tail[i - 1];
    }
  } else {
    for (size_t i = 0; i < tail_size; ++i) {
      output[i] = tail[i];
    }
  }
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    if (insert_a) {
      path[size_t(a.left + 1) * rrt_dimension + d] = a.q[d];
    }
    if (insert_b) {
      path[size_t(a.left + 1 + insert_a) * rrt_dimension + d] = b.q[d];
    }
  }
  return destination + length - b.right;
}

// One block owns each path. Collision samples run in parallel; accepted edits
// are serialized because the next trial must sample the updated arc lengths.
// Each path and distance slot needs initial_length + max_steps entries.
__global__ void tenkai_shortcut(Scalar* paths,
                                double* distances,
                                const ShortcutProblem* problems,
                                RrtSpace space,
                                Scene scene,
                                tenkai::ShortcutConfig config,
                                uint32_t* lengths) {
  const ShortcutProblem problem = problems[blockIdx.x];
  Scalar* path = paths + problem.offset * rrt_dimension;
  double* cumulative = distances + problem.offset;
  const uint32_t max_steps =
      config.max_steps == UINT32_MAX ? problem.length : config.max_steps;
  const uint32_t max_empty = config.max_empty_steps == UINT32_MAX
                                 ? problem.length
                                 : config.max_empty_steps;

  __shared__ uint32_t length;
  __shared__ uint32_t empty_steps;
  __shared__ uint32_t segments;
  __shared__ bool candidate;
  __shared__ ShortcutPoint a;
  __shared__ ShortcutPoint b;
  uint64_t random = problem.seed;
  if (threadIdx.x == 0) {
    length = problem.length;
    empty_steps = 0;
    if (length) {
      shortcut_distances(path, length, cumulative);
    }
  }
  __syncthreads();

  for (uint32_t step = 0; step < max_steps && empty_steps < max_empty; ++step) {
    if (length < 3 || cumulative[length - 1] == 0) {
      break;
    }

    if (threadIdx.x == 0) {
      const double total = cumulative[length - 1];
      const double s0 = rrt_uniform(random) * total;
      const double lower = fmax(0.0, s0 - config.range_ratio * total);
      const double upper = fmin(total, s0 + config.range_ratio * total);
      const double s1 = lower + rrt_uniform(random) * (upper - lower);
      a = shortcut_point(path, cumulative, length, fmin(s0, s1),
                         config.snap_to_vertex * total);
      b = shortcut_point(path, cumulative, length, fmax(s0, s1),
                         config.snap_to_vertex * total);

      candidate = a.left + 1 < b.right;
      if (candidate) {
        const double replacement =
            shortcut_distance(path + size_t(a.left) * rrt_dimension, a.q) +
            shortcut_distance(a.q, b.q) +
            shortcut_distance(b.q, path + size_t(b.right) * rrt_dimension);
        candidate = replacement <= cumulative[b.right] - cumulative[a.left];
      }
      if (candidate) {
        double count = 1;
        for (uint32_t d = 0; d < rrt_dimension; ++d) {
          count = fmax(count, ceil(fabs(double(b.q[d]) - double(a.q[d])) /
                                   double(space.resolution[d])));
        }
        segments = static_cast<uint32_t>(count);
      }
    }
    __syncthreads();

    bool collision = false;
    if (candidate) {
      for (uint64_t sample = threadIdx.x; sample <= segments;
           sample += blockDim.x) {
        Scalar q[rrt_dimension];
        const double ratio = double(sample) / double(segments);
        for (uint32_t d = 0; d < rrt_dimension; ++d) {
          q[d] = sample == 0 ? a.q[d]
                 : sample == segments
                     ? b.q[d]
                     : Scalar(double(a.q[d]) +
                              ratio * (double(b.q[d]) - double(a.q[d])));
        }
        if (rrt_collision(q, space, scene, problem.environment_id)) {
          collision = true;
          break;
        }
      }
    }
    const bool blocked = __syncthreads_or(collision);
    if (threadIdx.x == 0) {
      if (candidate && !blocked) {
        length = shortcut_splice(path, length, a, b);
        shortcut_distances(path, length, cumulative);
        empty_steps = 0;
      } else {
        ++empty_steps;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    lengths[blockIdx.x] = length;
  }
}
}  // namespace tenkai_f32
#endif
