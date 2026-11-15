#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "planner.cuh"

namespace model = tenkai_f32;
using Scalar = model::Scalar;

// nvcc propagates hidden visibility from anonymous namespace parameter types
// to the C entry point, preventing ctypes from finding it.
struct SceneInput {
  const uint32_t* ranges;
  const double* boxes;
  const double* spheres;
  const double* cylinders;
  const double* grounds;
  uint32_t box_count;
  uint32_t sphere_count;
  uint32_t cylinder_count;
  uint32_t ground_count;
};

namespace {
thread_local std::string last_error;

template <typename F>
int boundary(F operation) {
  try {
    operation();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

Scalar scene_scalar(double value) {
  const Scalar result = Scalar(value);
  if (!std::isfinite(value) || !std::isfinite(result)) {
    throw std::invalid_argument("nonfinite scene geometry");
  }
  return result;
}

model::Vec3 scene_vector(const double* values) {
  return {scene_scalar(values[0]), scene_scalar(values[1]),
          scene_scalar(values[2])};
}

model::Pose scene_pose(const double* values) {
  model::Pose pose{};
  pose.position = scene_vector(values);
  for (size_t i = 0; i < 9; ++i) {
    pose.world_to_local[i] = scene_scalar(values[3 + i]);
  }

  const Scalar* r = pose.world_to_local;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      double dot = 0;
      for (size_t k = 0; k < 3; ++k) {
        dot += double(r[3 * i + k]) * double(r[3 * j + k]);
      }
      if (std::abs(dot - (i == j ? 1.0 : 0.0)) > 1e-5) {
        throw std::invalid_argument("scene rotation must be orthonormal");
      }
    }
  }
  const double determinant =
      double(r[0]) * (double(r[4]) * r[8] - double(r[5]) * r[7]) -
      double(r[1]) * (double(r[3]) * r[8] - double(r[5]) * r[6]) +
      double(r[2]) * (double(r[3]) * r[7] - double(r[4]) * r[6]);
  if (determinant < 0) {
    throw std::invalid_argument("scene rotation must have determinant +1");
  }
  return pose;
}

struct HostScene {
  std::vector<model::Environment> environments;
  std::vector<model::Box> boxes;
  std::vector<model::Sphere> spheres;
  std::vector<model::Cylinder> cylinders;
  std::vector<model::Ground> grounds;

  HostScene(size_t count, const SceneInput& input)
      : environments(count),
        boxes(input.box_count),
        spheres(input.sphere_count),
        cylinders(input.cylinder_count),
        grounds(input.ground_count) {
    if (!input.ranges || (input.box_count && !input.boxes) ||
        (input.sphere_count && !input.spheres) ||
        (input.cylinder_count && !input.cylinders) ||
        (input.ground_count && !input.grounds)) {
      throw std::invalid_argument("null scene input");
    }

    const size_t totals[] = {boxes.size(), spheres.size(), cylinders.size(),
                             grounds.size()};
    for (size_t i = 0; i < count; ++i) {
      model::Range ranges[4];
      for (size_t kind = 0; kind < 4; ++kind) {
        const size_t offset = input.ranges[i * 8 + kind * 2];
        const size_t size = input.ranges[i * 8 + kind * 2 + 1];
        if (offset > totals[kind] || size > totals[kind] - offset) {
          throw std::invalid_argument("scene range exceeds primitive array");
        }
        ranges[kind] = {offset, size};
      }
      environments[i] = {ranges[0], ranges[1], ranges[2], ranges[3]};
    }

    for (size_t i = 0; i < boxes.size(); ++i) {
      const double* row = input.boxes + i * 15;
      const model::Vec3 extents = scene_vector(row + 12);
      if (row[12] < 0 || row[13] < 0 || row[14] < 0) {
        throw std::invalid_argument("negative box extent");
      }
      boxes[i] = {scene_pose(row), extents};
    }
    for (size_t i = 0; i < spheres.size(); ++i) {
      const double* row = input.spheres + i * 4;
      if (row[3] < 0) {
        throw std::invalid_argument("negative sphere radius");
      }
      spheres[i] = {scene_vector(row), scene_scalar(row[3])};
    }
    for (size_t i = 0; i < cylinders.size(); ++i) {
      const double* row = input.cylinders + i * 14;
      if (row[12] < 0 || row[13] < 0) {
        throw std::invalid_argument("negative cylinder radius or height");
      }
      cylinders[i] = {scene_pose(row), scene_scalar(row[12]),
                      scene_scalar(row[13])};
    }
    for (size_t i = 0; i < grounds.size(); ++i) {
      grounds[i] = {scene_scalar(input.grounds[i])};
    }
  }
};

struct Context {
  size_t problem_count;
  size_t environment_count;

  model::DeviceBuffer<model::Environment> environments;
  model::DeviceBuffer<model::Box> boxes;
  model::DeviceBuffer<model::Sphere> spheres;
  model::DeviceBuffer<model::Cylinder> cylinders;
  model::DeviceBuffer<model::Ground> grounds;
  model::Scene scene;
  std::unique_ptr<model::RrtPlanner> planner;

  Context(const tenkai::RrtConfig& config,
          size_t problems,
          const HostScene& host,
          const double* lower,
          const double* upper,
          const double* resolution)
      : problem_count(problems),
        environment_count(host.environments.size()),
        environments(host.environments.size()),
        boxes(host.boxes.size()),
        spheres(host.spheres.size()),
        cylinders(host.cylinders.size()),
        grounds(host.grounds.size()) {
    model::RrtSpace space{};
    for (uint32_t d = 0; d < model::rrt_dimension; ++d) {
      space.lower[d] = Scalar(lower[d]);
      space.upper[d] = Scalar(upper[d]);
      space.resolution[d] = Scalar(resolution[d]);
    }
    model::validate_rrt(config, space);

    environments.upload(host.environments.data(), host.environments.size());
    boxes.upload(host.boxes.data(), host.boxes.size());
    spheres.upload(host.spheres.data(), host.spheres.size());
    cylinders.upload(host.cylinders.data(), host.cylinders.size());
    grounds.upload(host.grounds.data(), host.grounds.size());

    scene.environments = environments.data();
    scene.boxes = boxes.data();
    scene.spheres = spheres.data();
    scene.cylinders = cylinders.data();
    scene.grounds = grounds.data();
    planner.reset(new model::RrtPlanner(config, space, scene, environment_count,
                                        problems, host.environments.data()));
  }
};

Context& context(void* handle) {
  if (!handle) {
    throw std::invalid_argument("null RRT handle");
  }

  return *static_cast<Context*>(handle);
}

}  // namespace

extern "C" const char* rrt_error() {
  return last_error.c_str();
}

extern "C" int rrtc_create_scene(const tenkai::RrtConfig* config,
                                 uint32_t problems,
                                 uint32_t environments,
                                 const SceneInput* input,
                                 const double* lower,
                                 const double* upper,
                                 const double* resolution,
                                 void** output) {
  return boundary([&] {
    if (!output) {
      throw std::invalid_argument("null RRT output handle");
    }
    *output = nullptr;

    if (!config || !problems || !environments || !input || !lower || !upper ||
        !resolution) {
      throw std::invalid_argument("invalid RRT creation arguments");
    }

    const HostScene host(environments, *input);
    *output = new Context(*config, problems, host, lower, upper, resolution);
  });
}

extern "C" void rrt_destroy(void* handle) {
  delete static_cast<Context*>(handle);
}

extern "C" int rrt_solve(void* handle,
                         const double* starts,
                         const double* goals,
                         const uint32_t* environment_ids,
                         const uint64_t* seeds,
                         tenkai::RrtResult* output) {
  return boundary([&] {
    auto& ctx = context(handle);
    if (!starts || !goals || !environment_ids || !seeds || !output) {
      throw std::invalid_argument("null RRT solve input/output");
    }

    std::vector<model::RrtProblem> problems(ctx.problem_count);
    for (size_t p = 0; p < problems.size(); ++p) {
      problems[p].environment_id = environment_ids[p];
      problems[p].seed = seeds[p];
      for (uint32_t d = 0; d < model::rrt_dimension; ++d) {
        problems[p].start[d] = Scalar(starts[p * model::rrt_dimension + d]);
        problems[p].goal[d] = Scalar(goals[p * model::rrt_dimension + d]);
      }
    }

    const auto results = ctx.planner->solve(problems);
    std::copy(results.begin(), results.end(), output);
  });
}

extern "C" int rrt_path(void* handle,
                        uint32_t problem,
                        double* output,
                        uint32_t capacity,
                        uint32_t* length) {
  return boundary([&] {
    const auto path = context(handle).planner->path(problem);
    if (!length || (!path.empty() && !output)) {
      throw std::invalid_argument("null RRT path output");
    }

    *length = static_cast<uint32_t>(path.size() / model::rrt_dimension);
    if (*length > capacity) {
      throw std::length_error("path output buffer too small");
    }

    std::copy(path.begin(), path.end(), output);
  });
}

extern "C" int rrt_check(void* handle,
                         const double* configurations,
                         const uint32_t* environment_ids,
                         uint32_t count,
                         unsigned char* output) {
  return boundary([&] {
    const auto& ctx = context(handle);
    if (!count) {
      return;
    }
    if (!configurations || !environment_ids || !output) {
      throw std::invalid_argument("null collision input/output");
    }

    std::vector<Scalar> q(size_t(count) * model::rrt_fk_dof);
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = Scalar(configurations[i]);
      if (!std::isfinite(q[i])) {
        throw std::invalid_argument("nonfinite collision input");
      }
    }
    for (size_t i = 0; i < count; ++i) {
      if (environment_ids[i] >= ctx.environment_count) {
        throw std::invalid_argument("invalid environment ID");
      }
    }

    model::DeviceBuffer<Scalar> device_q(q.size());
    model::DeviceBuffer<uint32_t> ids(count);
    model::DeviceBuffer<unsigned char> flags(count);
    device_q.upload(q.data(), q.size());
    ids.upload(environment_ids, count);

    tenkai_collision<<<(size_t(count) + 127) / 128, 128>>>(
        device_q.data(), ids.data(), ctx.scene, flags.data(), count);
    model::cuda_check(cudaGetLastError(), "launch path validation");

    flags.download(output, count);
  });
}
