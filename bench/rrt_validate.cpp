#include <cstdint>
#include <iostream>
#include "plainmp_checker.hpp"

template <typename T> std::vector<T> read_values(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    const auto bytes = file.tellg();
    if (bytes < 0 || size_t(bytes) % sizeof(T)) throw std::runtime_error("invalid size: " + path);
    std::vector<T> values(size_t(bytes) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file) throw std::runtime_error("cannot read " + path);
    return values;
}

int main(int argc, char** argv) try {
    if (argc != 3) throw std::invalid_argument("usage: rrt_validate <rkern-root> <result-directory>");
    const std::string root = argv[1], directory = argv[2];
    const auto config = YAML::LoadFile(root + "/assets/fetch/collision.yaml");
    const auto names = config["control_joint_names"].as<std::vector<std::string>>();
    const auto q = read_values<double>(directory + "/validation_q.bin");
    const auto ids = read_values<uint32_t>(directory + "/validation_env_ids.bin");
    const auto geometry = read_values<double>(directory + "/geometry.bin");
    if (geometry.empty() || geometry.size() % 7 || q.size() != ids.size() * names.size())
        throw std::invalid_argument("invalid validation input dimensions");
    std::vector<std::unique_ptr<cst::SphereCollisionCst>> checkers;
    for (size_t e = 0; e < geometry.size() / 7; ++e)
        checkers.emplace_back(new cst::SphereCollisionCst(make_checker(root, config, names, &geometry[7*e])));
    Eigen::VectorXd configuration(names.size());
    size_t collisions = 0;
    for (size_t p = 0; p < ids.size(); ++p) {
        if (ids[p] >= checkers.size()) throw std::out_of_range("invalid environment ID");
        for (size_t d = 0; d < names.size(); ++d) {
            configuration[d] = q[p * names.size() + d];
            if (!std::isfinite(configuration[d])) throw std::invalid_argument("nonfinite configuration");
        }
        collisions += !checkers[ids[p]]->is_valid(configuration);
    }
    std::cout << "{\"points\":" << ids.size() << ",\"collisions\":" << collisions << "}\n";
    return collisions ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
