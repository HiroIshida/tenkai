#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "plainmp/constraints/primitive_sphere_collision.hpp"
#include "workload.hpp"

namespace kin = plainmp::kinematics;
namespace cst = plainmp::constraint;
namespace sdf = plainmp::collision;
using Clock = std::chrono::steady_clock;

cst::SphereCollisionCst make_checker(const std::string& root, const YAML::Node& config,
                                    const std::vector<std::string>& names, const Scalar* geometry) {
    std::ifstream xml_file(root + "/assets/fetch/fetch.urdf");
    if (!xml_file) throw std::runtime_error("cannot open Fetch URDF");
    const std::string xml((std::istreambuf_iterator<char>(xml_file)), std::istreambuf_iterator<char>());
    auto model = std::make_shared<kin::KinematicModel<double>>(xml);
    std::vector<cst::SphereAttachmentSpec> attachments;
    for (const auto& entry : config["collision_spheres"]) {
        const auto definitions = entry.second["spheres"];
        cst::SphereAttachmentSpec spec;
        spec.parent_link_name = entry.first.as<std::string>();
        spec.relative_positions.resize(3, definitions.size());
        spec.radii.resize(definitions.size());
        // Excluding head/torso spheres would compare different environment workloads.
        spec.only_self_collision = false;
        for (size_t i = 0; i < definitions.size(); ++i) {
            for (size_t axis = 0; axis < 3; ++axis)
                spec.relative_positions(axis, i) = definitions[i][axis].as<double>();
            spec.radii[i] = definitions[i][3].as<double>();
        }
        attachments.push_back(std::move(spec));
    }
    cst::SphereCollisionCst checker(model, names, kin::BaseType::FIXED,
                                   attachments, {}, std::nullopt, true);
    auto box = std::make_shared<sdf::BoxSDF>(
        Eigen::Vector3d(geometry[3], geometry[4], geometry[5]),
        sdf::Pose(Eigen::Vector3d(geometry[0], geometry[1], geometry[2]), Eigen::Matrix3d::Identity()));
    auto ground = std::make_shared<sdf::GroundSDF>(geometry[6]);
    checker.set_sdf(std::make_shared<sdf::UnionSDF>(std::vector<sdf::SDFBase::Ptr>{box, ground}));
    return checker;
}

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
    if (argc != 3) throw std::runtime_error("usage: plainmp_bench <rkern-root> <data-directory>");
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
