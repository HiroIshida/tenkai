#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace @NAMESPACE@ {
inline void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

inline size_t storage_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::length_error("RRT storage size overflow");
    return a * b;
}

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        const size_t bytes = storage_product(count, sizeof(T));
        if (bytes) cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() { return data_; }
    const T* data() const { return data_; }
    void upload(const T* input, size_t count) {
        if (count > count_) throw std::out_of_range("device upload exceeds buffer");
        if (count) cuda_check(cudaMemcpy(data_, input, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    void download(T* output, size_t count, size_t offset = 0) const {
        if (offset > count_ || count > count_ - offset)
            throw std::out_of_range("device download exceeds buffer");
        if (count) cuda_check(cudaMemcpy(output, data_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    }
private:
    T* data_ = nullptr;
    size_t count_;
};

struct CudaRrtConfig { uint32_t warps_per_problem = 4; };

// Scene buffers are borrowed and must outlive the planner. All search storage
// is reused across solves; a new solve invalidates the previous tree/path.
class RrtPlanner {
public:
    RrtPlanner(const rkern::RrtConfig& config, CudaRrtConfig execution,
               const RrtSpace& space, Scene scene, size_t environment_count, size_t problem_count)
        : config_(checked_config(config, execution, space, environment_count, problem_count)),
          threads_(execution.warps_per_problem * 32), space_(space), scene_(scene),
          environment_count_(environment_count), problem_count_(problem_count),
          problems_(problem_count),
          nodes_(storage_product(storage_product(problem_count, config.max_nodes), rrt_dimension)),
          parents_(storage_product(problem_count, config.max_nodes)),
          candidates_(storage_product(problem_count, config.samples_per_round)),
          paths_(storage_product(problem_count, config.max_nodes)), results_(problem_count) {}

    std::vector<rkern::RrtResult> solve(const std::vector<RrtProblem>& problems) {
        if (problems.size() != problem_count_) throw std::invalid_argument("RRT problem count mismatch");
        for (const auto& problem : problems)
            if (problem.environment_id >= environment_count_)
                throw std::invalid_argument("RRT environment ID out of range");
        solved_ = false;
        problems_.upload(problems.data(), problems.size());
        rkern_rrt<<<static_cast<unsigned int>(problem_count_), threads_>>>(
            problems_.data(), problem_count_, space_, scene_, config_, nodes_.data(),
            parents_.data(), candidates_.data(), paths_.data(), results_.data());
        cuda_check(cudaGetLastError(), "launch rkern_rrt");
        std::vector<rkern::RrtResult> result(problem_count_);
        results_.download(result.data(), result.size());
        solved_ = true;
        return result;
    }

    std::vector<Scalar> path(size_t problem_id) const {
        if (!solved_ || problem_id >= problem_count_) throw std::out_of_range("RRT path unavailable");
        rkern::RrtResult result;
        results_.download(&result, 1, problem_id);
        std::vector<uint32_t> indices(result.path_length);
        paths_.download(indices.data(), indices.size(), problem_id * config_.max_nodes);
        std::vector<Scalar> nodes(size_t(result.node_count) * rrt_dimension);
        nodes_.download(nodes.data(), nodes.size(), problem_id * config_.max_nodes * rrt_dimension);
        std::vector<Scalar> path(size_t(result.path_length) * rrt_dimension);
        for (size_t i = 0; i < indices.size(); ++i)
            for (uint32_t d = 0; d < rrt_dimension; ++d)
                path[i * rrt_dimension + d] = nodes[size_t(indices[i]) * rrt_dimension + d];
        return path;
    }

private:
    static rkern::RrtConfig checked_config(const rkern::RrtConfig& config, CudaRrtConfig execution,
                                          const RrtSpace& space, size_t environments, size_t problems) {
        validate_rrt(config, space);
        if (!execution.warps_per_problem || execution.warps_per_problem > 32 || !environments || !problems)
            throw std::invalid_argument("invalid CUDA RRT execution dimensions");
        int device;
        cudaDeviceProp properties;
        cudaFuncAttributes attributes;
        cuda_check(cudaGetDevice(&device), "get CUDA device");
        cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA properties");
        cuda_check(cudaFuncGetAttributes(&attributes, rkern_rrt), "get RRT kernel attributes");
        if (problems > size_t(properties.maxGridSize[0]) ||
            execution.warps_per_problem * 32 > uint32_t(attributes.maxThreadsPerBlock))
            throw std::invalid_argument("RRT launch exceeds device/kernel limits");
        int active_blocks;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, rkern_rrt, execution.warps_per_problem * 32, 0), "check RRT occupancy");
        if (!active_blocks) throw std::invalid_argument("RRT block cannot fit on an SM");
        return config;
    }

    rkern::RrtConfig config_;
    uint32_t threads_;
    RrtSpace space_;
    Scene scene_;
    size_t environment_count_, problem_count_;
    bool solved_ = false;
    DeviceBuffer<RrtProblem> problems_;
    DeviceBuffer<Scalar> nodes_;
    DeviceBuffer<uint32_t> parents_;
    DeviceBuffer<RrtCandidate> candidates_;
    DeviceBuffer<uint32_t> paths_;
    DeviceBuffer<rkern::RrtResult> results_;
};
}
