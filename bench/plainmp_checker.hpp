#pragma once
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "plainmp/constraints/primitive_sphere_collision.hpp"

namespace kin = plainmp::kinematics;
namespace cst = plainmp::constraint;
namespace sdf = plainmp::collision;

template <typename GeometryScalar>
cst::SphereCollisionCst make_checker(const std::string& root, const YAML::Node& config,
                                    const std::vector<std::string>& names, const GeometryScalar* geometry) {
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

