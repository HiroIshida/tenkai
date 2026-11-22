#include "rrt/bindings.cu"

extern "C" int test_shortcut(void* handle,
                             const tenkai::ShortcutConfig* config,
                             uint32_t count,
                             const uint64_t* offsets,
                             Scalar* paths,
                             uint32_t* lengths,
                             const uint32_t* ids,
                             const uint64_t* seeds,
                             double resolution) {
  return boundary([&] {
    model::validate_shortcut(*config);
    model::RrtSpace space{};
    for (uint32_t d = 0; d < model::rrt_dimension; ++d) {
      space.lower[d] = -1;
      space.upper[d] = 1;
      space.resolution[d] = Scalar(resolution);
    }
    std::vector<model::ShortcutProblem> problems(count);
    for (uint32_t p = 0; p < count; ++p) {
      problems[p] = {size_t(offsets[p]), lengths[p], ids[p], seeds[p]};
    }
    const size_t capacity = offsets[count];
    model::DeviceBuffer<Scalar> coordinates(capacity * model::rrt_dimension);
    model::DeviceBuffer<double> distances(capacity);
    model::DeviceBuffer<model::ShortcutProblem> descriptors(count);
    model::DeviceBuffer<uint32_t> output_lengths(count);
    coordinates.upload(paths, capacity * model::rrt_dimension);
    descriptors.upload(problems.data(), count);
    model::tenkai_shortcut<<<count, 32>>>(
        coordinates.data(), distances.data(), descriptors.data(), space,
        context(handle).scene, *config, output_lengths.data());
    model::cuda_check(cudaGetLastError(), "test shortcut kernel");
    coordinates.download(paths, capacity * model::rrt_dimension);
    output_lengths.download(lengths, count);
  });
}
