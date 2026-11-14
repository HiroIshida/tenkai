#pragma once

// Bypass streaming and workspace certificates to check their paths against
// the full collision predicate without adding a runtime execution selector.
class ReferencePlanner {
public:
    ReferencePlanner(const tenkai::RrtConfig& config, const m::RrtSpace& space, m::Scene scene,
                     size_t environments, size_t problems)
        : config_(config), space_(space), scene_(scene), count_(problems), problems_(problems),
          nodes_(problems * config.max_nodes * m::rrt_dimension),
          parents_(problems * config.max_nodes), trees_(problems * config.max_nodes),
          paths_(problems * config.max_nodes), results_(problems), clear_(environments) {
        std::vector<unsigned char> clear(environments, 0);
        clear_.upload(clear.data(), clear.size());
    }

    std::vector<tenkai::RrtResult> solve(const std::vector<m::RrtProblem>& problems) {
        assert(problems.size() == count_);
        problems_.upload(problems.data(), count_);
        tenkai_rrtc<<<static_cast<unsigned int>(count_), 32>>>(
            problems_.data(), count_, space_, scene_, config_, nodes_.data(), parents_.data(),
            trees_.data(), paths_.data(), results_.data(), clear_.data());
        m::cuda_check(cudaGetLastError(), "launch reference RRT-Connect");
        last_.resize(count_);
        results_.download(last_.data(), count_);
        return last_;
    }

    std::vector<Scalar> path(size_t id) const {
        const auto& result = last_.at(id);
        std::vector<uint32_t> indices(result.path_length);
        paths_.download(indices.data(), indices.size(), id * config_.max_nodes);
        std::vector<Scalar> nodes(size_t(result.node_count) * m::rrt_dimension);
        nodes_.download(nodes.data(), nodes.size(), id * config_.max_nodes * m::rrt_dimension);
        std::vector<Scalar> path(indices.size() * m::rrt_dimension);
        for (size_t i = 0; i < indices.size(); ++i)
            for (uint32_t d = 0; d < m::rrt_dimension; ++d)
                path[i * m::rrt_dimension + d] = nodes[size_t(indices[i]) * m::rrt_dimension + d];
        return path;
    }

private:
    tenkai::RrtConfig config_;
    m::RrtSpace space_;
    m::Scene scene_;
    size_t count_;
    m::DeviceBuffer<m::RrtProblem> problems_;
    m::DeviceBuffer<Scalar> nodes_;
    m::DeviceBuffer<uint32_t> parents_;
    m::DeviceBuffer<m::RrtTree> trees_;
    m::DeviceBuffer<uint32_t> paths_;
    m::DeviceBuffer<tenkai::RrtResult> results_;
    m::DeviceBuffer<unsigned char> clear_;
    std::vector<tenkai::RrtResult> last_;
};
