#include <iostream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <vector>
#include "cg.hpp"
#include "linalg.hpp"

using namespace tenkai;

auto gen_jit_func() {
  auto inp0 = Operation::make_var();
  auto inp1 = Operation::make_var();
  auto inp2 = Operation::make_var();

  auto A = Matrix3::RotX(inp0);
  auto B = Matrix3::RotY(inp1);
  auto C = Matrix3::RotZ(inp2);
  auto v = Vector3({inp0, inp1, inp2});
  auto Av = (A * v);
  auto BAv = (B * Av);
  auto CBAv = (C * BAv);
  auto out1 = Av.sum();
  auto out2 = BAv.sum();
  auto out3 = CBAv.sqnorm();
  auto out4 = CBAv(0);
  auto out5 = (Av + BAv + CBAv).sqnorm();

  std::vector<Operation::Ptr> inputs = {inp0, inp1, inp2};
  std::vector<Operation::Ptr> outputs = {out1, out2, out3, out4, out5};
  auto f = jit_compile<double>(inputs, outputs, "g++");
  return f;
}

void eigen_counterpart(double* input, double* output) {
  Eigen::Matrix3d A = Eigen::AngleAxisd(input[0], Eigen::Vector3d::UnitX()).toRotationMatrix();
  Eigen::Matrix3d B = Eigen::AngleAxisd(input[1], Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Matrix3d C = Eigen::AngleAxisd(input[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector3d v(input[0], input[1], input[2]);

  auto Av = A * v;
  auto BAv = B * Av;
  auto CBAv = C * BAv;
  output[0] = Av.sum();
  output[1] = BAv.sum();
  output[2] = CBAv.squaredNorm();
  output[3] = CBAv[0];
  output[4] = (Av + BAv + CBAv).squaredNorm();
}

int main() {
  size_t num_iterations = 1000000;
  auto jit_func = gen_jit_func();
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-M_PI, M_PI);

  std::vector<double> input(3);
  std::vector<double> output_jit(5);
  std::vector<double> output_eigen(5);

  // Warm-up
  for (int i = 0; i < 1000; ++i) {
      for (int j = 0; j < 3; ++j) {
          input[j] = dis(gen);
      }
      jit_func(input.data(), output_jit.data());
      eigen_counterpart(input.data(), output_eigen.data());
  }

  // JIT benchmark
  auto start_jit = std::chrono::high_resolution_clock::now();
  double out_sum_jit = 0.0;
  for (int i = 0; i < num_iterations; ++i) {
      for (int j = 0; j < 3; ++j) {
          input[j] = dis(gen);
      }
      jit_func(input.data(), output_jit.data());
      out_sum_jit += output_jit[4];
  }
  auto end_jit = std::chrono::high_resolution_clock::now();
  auto duration_jit = std::chrono::duration_cast<std::chrono::microseconds>(end_jit - start_jit);
  std::cout << "out_sum_jit: " << out_sum_jit << std::endl;

  // Eigen benchmark
  auto start_eigen = std::chrono::high_resolution_clock::now();
  double out_sum_eigen = 0.0;
  for (int i = 0; i < num_iterations; ++i) {
      for (int j = 0; j < 3; ++j) {
          input[j] = dis(gen);
      }
      eigen_counterpart(input.data(), output_eigen.data());
      out_sum_eigen += output_eigen[4];
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(end_eigen - start_eigen);
  std::cout << "out_sum_eigen: " << out_sum_eigen << std::endl;

  std::cout << "JIT time: " << duration_jit.count() / 1e6 << " seconds" << std::endl;
  std::cout << "Eigen time: " << duration_eigen.count() / 1e6 << " seconds" << std::endl;
  std::cout << "JIT/Eigen ratio: " << static_cast<double>(duration_jit.count()) / duration_eigen.count() << std::endl;
}
