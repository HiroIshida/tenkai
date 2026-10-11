cmake_minimum_required(VERSION 3.15)
project(plainmp_collision_bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT WORKLOAD_DIR)
    message(FATAL_ERROR "Pass -DWORKLOAD_DIR=<directory containing config.h>")
endif()

add_compile_options(-march=native)
add_compile_definitions(EIGEN_NO_DEBUG EIGEN_DONT_VECTORIZE)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(NOT ipo_supported)
    message(FATAL_ERROR "LTO unavailable: ${ipo_error}")
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
set(PLAINMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/plainmp")
add_subdirectory("${PLAINMP_ROOT}/third/urdf_parser" urdf_parser)
add_executable(plainmp_bench plainmp_bench.cpp
    "${PLAINMP_ROOT}/cpp/plainmp/kinematics/kinematics.cpp"
    "${PLAINMP_ROOT}/cpp/plainmp/kinematics/algorithm.cpp"
    "${PLAINMP_ROOT}/cpp/plainmp/kinematics/kinematic_model_wrapper.cpp"
    "${PLAINMP_ROOT}/cpp/plainmp/constraints/primitive_sphere_collision.cpp"
    "${PLAINMP_ROOT}/cpp/plainmp/collision/kdtree.cpp")
target_include_directories(plainmp_bench PRIVATE "${PLAINMP_ROOT}/cpp" "${WORKLOAD_DIR}")
target_link_libraries(plainmp_bench PRIVATE Eigen3::Eigen urdfdom_model yaml-cpp)

# Python.h normally supplies floating-point abs overloads; without them the
# unqualified abs in plainmp's SDF header can bind to the integer overload.
target_compile_options(plainmp_bench PRIVATE -include math.h)
