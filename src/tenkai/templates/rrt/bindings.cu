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

/* TPL_LAYOUT_ASSERTS */

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

template <typename T>
struct HostArray {
  const T* data;
  size_t count;
};

struct Context {
  size_t problem_count;

  model::DeviceBuffer<model::Environment> environments;
  model::DeviceBuffer<model::Box> boxes;
  model::DeviceBuffer<model::Sphere> spheres;
  model::DeviceBuffer<model::Cylinder> cylinders;
  model::DeviceBuffer<model::Ground> grounds;
  model::Scene scene;
  std::unique_ptr<model::RrtPlanner> planner;

  Context(const tenkai::RrtConfig& config,
          size_t problems,
          HostArray<model::Environment> host_environments,
          HostArray<model::Box> host_boxes,
          HostArray<model::Sphere> host_spheres,
          HostArray<model::Cylinder> host_cylinders,
          HostArray<model::Ground> host_grounds,
          const double* lower,
          const double* upper,
          const double* resolution)
      : problem_count(problems),
        environments(host_environments.count),
        boxes(host_boxes.count),
        spheres(host_spheres.count),
        cylinders(host_cylinders.count),
        grounds(host_grounds.count) {
    model::RrtSpace space{};
    for (uint32_t d = 0; d < model::rrt_dimension; ++d) {
      space.lower[d] = Scalar(lower[d]);
      space.upper[d] = Scalar(upper[d]);
      space.resolution[d] = Scalar(resolution[d]);
    }
    environments.upload(host_environments.data, host_environments.count);
    boxes.upload(host_boxes.data, host_boxes.count);
    spheres.upload(host_spheres.data, host_spheres.count);
    cylinders.upload(host_cylinders.data, host_cylinders.count);
    grounds.upload(host_grounds.data, host_grounds.count);

    scene.environments = environments.data();
    scene.boxes = boxes.data();
    scene.spheres = spheres.data();
    scene.cylinders = cylinders.data();
    scene.grounds = grounds.data();
    planner.reset(new model::RrtPlanner(config, space, scene,
                                        host_environments.count, problems,
                                        host_environments.data));
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
                                 uint32_t environment_count,
                                 const model::Environment* environments,
                                 const model::Box* boxes,
                                 uint32_t box_count,
                                 const model::Sphere* spheres,
                                 uint32_t sphere_count,
                                 const model::Cylinder* cylinders,
                                 uint32_t cylinder_count,
                                 const model::Ground* grounds,
                                 const double* lower,
                                 const double* upper,
                                 const double* resolution,
                                 void** output) {
  return boundary([&] {
    if (!output) {
      throw std::invalid_argument("null RRT output handle");
    }
    *output = nullptr;

    if (!config || !problems || !environment_count || !lower || !upper ||
        !resolution) {
      throw std::invalid_argument("invalid RRT creation arguments");
    }

    const size_t ground_count = model::has_ground ? environment_count : 0;
    *output = new Context(*config, problems, {environments, environment_count},
                          {boxes, box_count}, {spheres, sphere_count},
                          {cylinders, cylinder_count}, {grounds, ground_count},
                          lower, upper, resolution);
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
                         tenkai::RrtResult* output,
                         const tenkai::ShortcutConfig* shortcut) {
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

    const auto results = ctx.planner->solve(problems, shortcut);
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

extern "C" int rrt_paths(void* handle, double* output, uint64_t capacity) {
  return boundary([&] {
    const auto& planner = *context(handle).planner;
    const size_t length = planner.path_vertex_count();
    if (length > capacity) {
      throw std::length_error("path output buffer too small");
    }
    if (length && !output) {
      throw std::invalid_argument("null RRT paths output");
    }
    const auto paths = planner.paths();
    if (!paths.empty()) {
      std::copy(paths.begin(), paths.end(), output);
    }
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
