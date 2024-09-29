#include <iostream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <vector>
#include "cg.hpp"
#include "linalg.hpp"
#include "spatial.hpp"

using namespace tenkai;

auto get_jit_func() {
  auto x0 = Operation::make_var();
  auto x1 = Operation::make_var();
  auto x2 = Operation::make_var();
  auto x3 = Operation::make_var();
  auto x4 = Operation::make_var();
  auto x5 = Operation::make_var();
  auto x6 = Operation::make_var();

  auto trans = Vector({Operation::make_constant(1.0), Operation::make_constant(2.0), Operation::make_constant(3.0)});
  auto tf1 = SpatialTransform(Matrix::RotX(x0), trans);
  auto tf2 = SpatialTransform(Matrix::RotY(x1), trans);
  auto tf3 = SpatialTransform(Matrix::RotZ(x2), trans);
  auto tf4 = SpatialTransform(Matrix::RotX(x3), trans);
  auto tf5 = SpatialTransform(Matrix::RotY(x4), trans);
  auto tf6 = SpatialTransform(Matrix::RotZ(x5), trans);
  auto tf7 = SpatialTransform(Matrix::RotX(x6), trans);
  auto tf8 = tf4 * tf3 * tf2 * tf1;
  auto tf9 = tf7 * tf6 * tf5 * tf1;
  auto tf10 = tf9 * tf8;

  std::vector<Operation::Ptr> inputs = {x0, x1, x2, x3, x4, x5, x6};
  auto outputs = tf10.trans.elements;
  flatten("flattened", inputs, outputs, std::cout, "double"); // debug
  auto f_jit = jit_compile<double>(inputs, outputs, "g++");
  return f_jit;
}

void eigen_counterpart(double* input, double* output) {
    Eigen::Vector3d trans = Eigen::Vector3d(1.0, 2.0, 3.0);

    Eigen::Isometry3d tf1 = Eigen::Isometry3d(Eigen::AngleAxisd(input[0], Eigen::Vector3d::UnitX()));
    tf1.pretranslate(trans);

    Eigen::Isometry3d tf2 = Eigen::Isometry3d(Eigen::AngleAxisd(input[1], Eigen::Vector3d::UnitY()));
    tf2.pretranslate(trans);

    Eigen::Isometry3d tf3 = Eigen::Isometry3d(Eigen::AngleAxisd(input[2], Eigen::Vector3d::UnitZ()));
    tf3.pretranslate(trans);

    Eigen::Isometry3d tf4 = Eigen::Isometry3d(Eigen::AngleAxisd(input[3], Eigen::Vector3d::UnitX()));
    tf4.pretranslate(trans);

    Eigen::Isometry3d tf5 = Eigen::Isometry3d(Eigen::AngleAxisd(input[4], Eigen::Vector3d::UnitY()));
    tf5.pretranslate(trans);

    Eigen::Isometry3d tf6 = Eigen::Isometry3d(Eigen::AngleAxisd(input[5], Eigen::Vector3d::UnitZ()));
    tf6.pretranslate(trans);

    Eigen::Isometry3d tf7 = Eigen::Isometry3d(Eigen::AngleAxisd(input[6], Eigen::Vector3d::UnitX()));
    tf7.pretranslate(trans);

    Eigen::Isometry3d tf8 = tf4 * tf3 * tf2 * tf1;
    Eigen::Isometry3d tf9 = tf7 * tf6 * tf5 * tf1;
    Eigen::Isometry3d tf10 = tf9 * tf8;

    auto tmp = tf10.translation();
    output[0] = tmp[0];
    output[1] = tmp[1];
    output[2] = tmp[2];
}

int main() {
  auto f_jit = get_jit_func();
  std::vector<double> input(7);
  std::vector<double> output(3);
  std::vector<double> output_eigen(3);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-M_PI, M_PI);
  size_t n_trials = 1000000;
  for (int i = 0; i < 7; i++) {
    input[i] = dis(gen);
  }
  f_jit(input.data(), output.data());
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < n_trials; i++) {
    f_jit(input.data(), output.data());
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "jit: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / n_trials << " ns" << std::endl;

  eigen_counterpart(input.data(), output_eigen.data());
  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < n_trials; i++) {
    eigen_counterpart(input.data(), output_eigen.data());
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "eigen: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / n_trials << " ns" << std::endl;
}
