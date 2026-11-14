#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "plainmp_checker.hpp"
#include "workload.hpp"

using Clock = std::chrono::steady_clock;

std::vector<Eigen::VectorXd> load_configurations(const std::string& directory,
                                                const std::vector<std::string>& names) {
    const auto model = YAML::LoadFile(directory + "/build/model.json");
    const auto inputs = model["input_joint_names"].as<std::vector<std::string>>();
    if (inputs.size() != kDof) throw std::runtime_error("input joint count mismatch");
    std::vector<size_t> columns;
    for (const auto& name : names) {
        const auto found = std::find(inputs.begin(), inputs.end(), name);
        if (found == inputs.end()) throw std::runtime_error("missing control joint " + name);
        columns.push_back(found - inputs.begin());
    }
    std::vector<Scalar> raw(kPool * kDof);
    read_file(directory + "/configurations.bin", raw.data(), raw.size());
    std::vector<Eigen::VectorXd> configurations;
    configurations.reserve(kPool);
    for (size_t i = 0; i < kPool; ++i) {
        configurations.emplace_back(names.size());
        for (size_t j = 0; j < names.size(); ++j)
            configurations.back()[j] = raw[i*kDof + columns[j]];
    }
    return configurations;
}

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: plainmp_bench <tenkai-root> <data-directory>");
    const std::string root = argv[1], directory = argv[2];
    const auto config = YAML::LoadFile(root + "/assets/fetch/collision.yaml");
    const auto names = config["control_joint_names"].as<std::vector<std::string>>();
    const auto configurations = load_configurations(directory, names);
    Scalar geometry[7];
    read_file(directory + "/scene.bin", geometry, 7);
    auto checker = make_checker(root, config, names, geometry);
    std::vector<uint8_t> collision(kPool);

    auto measure = [&](size_t batch, size_t round, size_t repetitions) {
        const auto start = Clock::now();
        for (size_t iteration = 0; iteration < repetitions; ++iteration) {
            const size_t offset = input_offset(batch, round, iteration);
            for (size_t i = offset; i < offset + batch; ++i)
                collision[i] = !checker.is_valid(configurations[i]);
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / repetitions;
    };
    const auto warm_start = Clock::now();
    do { measure(std::min(kPool, size_t(65536)), 0, 8); }
    while (std::chrono::duration<double>(Clock::now() - warm_start).count() < kWarmupSeconds);

    std::cout << "mode,batch,round,repetitions,gpu_ms,wall_ms\n" << std::setprecision(12);
    const size_t count = sizeof(kBatches) / sizeof(kBatches[0]);
    for (size_t round = 0; round < kRounds; ++round) {
        for (size_t i = 0; i < count; ++i) {
            const size_t batch = kBatches[job_index(round, i, count)];
            const size_t repetitions = repetitions_for(batch);
            measure(batch, round, 2);
            const double elapsed = measure(batch, round, repetitions);
            std::cout << "plainmp," << batch << ',' << round << ',' << repetitions << ",," << elapsed << '\n';
        }
        std::cout.flush();
        std::cerr << "plainmp round " << round + 1 << '/' << kRounds << " complete\n";
    }

    // A full pass keeps validation coverage independent of the timed batch sizes.
    for (size_t i = 0; i < kPool; ++i)
        collision[i] = !checker.is_valid(configurations[i]);
    std::ofstream flags(directory + "/plainmp_collision.u8", std::ios::binary);
    flags.write(reinterpret_cast<const char*>(collision.data()), collision.size());
    flags.close();
    if (!flags || !std::cout) throw std::runtime_error("cannot write benchmark results");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
