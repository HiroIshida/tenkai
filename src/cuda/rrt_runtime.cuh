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
               const RrtSpace& space, Scene scene, size_t environment_count, size_t problem_count,
               rkern::RrtAlgorithm algorithm = rkern::RrtAlgorithm::Rrt)
        : config_(checked_config(config, execution, space, environment_count, problem_count, algorithm)),
          algorithm_(algorithm), threads_(execution.warps_per_problem * 32), space_(space), scene_(scene),
          environment_count_(environment_count), problem_count_(problem_count),
          problems_(problem_count),
          nodes_(storage_product(storage_product(problem_count, config.max_nodes), rrt_dimension)),
          parents_(storage_product(problem_count, config.max_nodes)),
          trees_(algorithm == rkern::RrtAlgorithm::RrtConnect ? storage_product(problem_count, config.max_nodes) : 0),
          candidates_(storage_product(problem_count, config.samples_per_round)),
          paths_(storage_product(problem_count, config.max_nodes)), results_(problem_count),
          workspace_clear_(algorithm == rkern::RrtAlgorithm::RrtConnect && single_candidate(config, threads_) ?
                           environment_count : 0)
#if @FP32@
          , preparation_(algorithm == rkern::RrtAlgorithm::RrtConnect && single_candidate(config, threads_) ?
                         problem_count : 0)
#endif
          {}

    std::vector<rkern::RrtResult> solve(const std::vector<RrtProblem>& problems) {
        if (problems.size() != problem_count_) throw std::invalid_argument("RRT problem count mismatch");
        for (const auto& problem : problems)
            if (problem.environment_id >= environment_count_)
                throw std::invalid_argument("RRT environment ID out of range");
        solved_ = false;
        problems_.upload(problems.data(), problems.size());
#if @FP32@
        if (algorithm_ == rkern::RrtAlgorithm::RrtConnect && single_candidate(config_, threads_)) {
            const size_t blocks = (environment_count_ - 1) / 128 + 1;
            rkern_rrtc_workspace<<<static_cast<unsigned int>(blocks < 65535 ? blocks : 65535), 128>>>(
                space_, scene_, environment_count_, workspace_clear_.data());
            cuda_check(cudaGetLastError(), "check RRT workspace");
            rkern_rrtc_prepare<<<static_cast<unsigned int>((problem_count_ - 1) / 128 + 1), 128>>>(
                problems_.data(), problem_count_, space_, scene_, workspace_clear_.data(),
                preparation_.data());
            cuda_check(cudaGetLastError(), "prepare RRT-Connect problems");
            rkern_rrtc_streaming<<<static_cast<unsigned int>(problem_count_), 32>>>(
                problems_.data(), problem_count_, space_, scene_, config_, nodes_.data(),
                parents_.data(), trees_.data(), candidates_.data(), paths_.data(), results_.data(),
                workspace_clear_.data(), preparation_.data());
            cuda_check(cudaGetLastError(), "launch streaming RRT-Connect");
            rkern_rrtc_fallback<<<static_cast<unsigned int>(problem_count_), 32>>>(
                problems_.data(), problem_count_, space_, scene_, config_, nodes_.data(),
                parents_.data(), trees_.data(), candidates_.data(), paths_.data(), results_.data(),
                workspace_clear_.data(), preparation_.data());
        } else
#endif
        if (algorithm_ == rkern::RrtAlgorithm::RrtConnect)
            rkern_rrtc<<<static_cast<unsigned int>(problem_count_), threads_>>>(
                problems_.data(), problem_count_, space_, scene_, config_, nodes_.data(),
                parents_.data(), trees_.data(), candidates_.data(), paths_.data(), results_.data());
        else
            rkern_rrt<<<static_cast<unsigned int>(problem_count_), threads_>>>(
                problems_.data(), problem_count_, space_, scene_, config_, nodes_.data(),
                parents_.data(), candidates_.data(), paths_.data(), results_.data());
        cuda_check(cudaGetLastError(), "launch RRT planner");
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
    static bool single_candidate(const rkern::RrtConfig& config, uint32_t threads) {
        return sizeof(Scalar) == sizeof(float) && config.samples_per_round == 1 && threads == 32;
    }

    static rkern::RrtConfig checked_config(const rkern::RrtConfig& config, CudaRrtConfig execution,
                                          const RrtSpace& space, size_t environments, size_t problems,
                                          rkern::RrtAlgorithm algorithm) {
        validate_rrt(config, space);
        if (algorithm != rkern::RrtAlgorithm::Rrt && algorithm != rkern::RrtAlgorithm::RrtConnect)
            throw std::invalid_argument("invalid RRT algorithm");
        if (!execution.warps_per_problem || execution.warps_per_problem > 32 || !environments || !problems)
            throw std::invalid_argument("invalid CUDA RRT execution dimensions");
        int device;
        cudaDeviceProp properties;
        cudaFuncAttributes attributes;
        const void* kernel = algorithm == rkern::RrtAlgorithm::RrtConnect ?
            (single_candidate(config, execution.warps_per_problem * 32) ?
                reinterpret_cast<const void*>(@RRTC_SINGLE@) : reinterpret_cast<const void*>(rkern_rrtc)) :
            reinterpret_cast<const void*>(rkern_rrt);
        cuda_check(cudaGetDevice(&device), "get CUDA device");
        cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA properties");
        cuda_check(cudaFuncGetAttributes(&attributes, kernel), "get RRT kernel attributes");
        if (problems > size_t(properties.maxGridSize[0]) ||
            execution.warps_per_problem * 32 > uint32_t(attributes.maxThreadsPerBlock))
            throw std::invalid_argument("RRT launch exceeds device/kernel limits");
        int active_blocks;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, kernel, execution.warps_per_problem * 32, 0), "check RRT occupancy");
        if (!active_blocks) throw std::invalid_argument("RRT block cannot fit on an SM");
#if @FP32@
        if (algorithm == rkern::RrtAlgorithm::RrtConnect && single_candidate(config, execution.warps_per_problem * 32)) {
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, rkern_rrtc_streaming, 32, 0), "check streaming RRT occupancy");
            if (!active_blocks) throw std::invalid_argument("streaming RRT block cannot fit on an SM");
        }
#endif
        return config;
    }

    rkern::RrtConfig config_;
    rkern::RrtAlgorithm algorithm_;
    uint32_t threads_;
    RrtSpace space_;
    Scene scene_;
    size_t environment_count_, problem_count_;
    bool solved_ = false;
    DeviceBuffer<RrtProblem> problems_;
    DeviceBuffer<Scalar> nodes_;
    DeviceBuffer<uint32_t> parents_;
    DeviceBuffer<RrtTree> trees_;
    DeviceBuffer<RrtCandidate> candidates_;
    DeviceBuffer<uint32_t> paths_;
    DeviceBuffer<rkern::RrtResult> results_;
    DeviceBuffer<unsigned char> workspace_clear_;
#if @FP32@
    DeviceBuffer<RrtPreparation> preparation_;
#endif
};
}
