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

  auto trans = Vector({Operation::make_constant(0.1), Operation::make_constant(0.2), Operation::make_constant(0.3)});
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
  // flatten("flattened", inputs, outputs, std::cout, "double"); // debug
  bool disassemble = true;
  auto f_jit = jit_compile<double>(inputs, outputs, "g++", disassemble);
  return f_jit;
}


struct QuatTrans {
  Eigen::Quaterniond quat;
  Eigen::Vector3d trans;
  void multiply_and_overwrite_self(const QuatTrans& other) {
    this->trans  = quat * other.trans + trans;
    this->quat = quat * other.quat;
  };
};


void eigen_counterpart(double* input, double* output) {
    Eigen::Quaterniond q1, q2, q3, q4, q5, q6, q7;
    q1.coeffs() << std::sin(input[0] / 2), 0, 0, std::cos(input[0] / 2);
    q2.coeffs() << 0, std::sin(input[1] / 2), 0, std::cos(input[1] / 2);
    q3.coeffs() << 0, 0, std::sin(input[2] / 2), std::cos(input[2] / 2);
    q4.coeffs() << std::sin(input[3] / 2), 0, 0, std::cos(input[3] / 2);
    q5.coeffs() << 0, std::sin(input[4] / 2), 0, std::cos(input[4] / 2);
    q6.coeffs() << 0, 0, std::sin(input[5] / 2), std::cos(input[5] / 2);
    q7.coeffs() << std::sin(input[6] / 2), 0, 0, std::cos(input[6] / 2);
    Eigen::Vector3d trans = Eigen::Vector3d(0.1, 0.2, 0.3);

    QuatTrans tf1{q1, trans};
    QuatTrans tf2{q2, trans};
    QuatTrans tf3{q3, trans};
    QuatTrans tf4{q4, trans};
    QuatTrans tf5{q5, trans};
    QuatTrans tf6{q6, trans};
    QuatTrans tf7{q7, trans};

    tf4.multiply_and_overwrite_self(tf3);
    tf4.multiply_and_overwrite_self(tf2);
    tf4.multiply_and_overwrite_self(tf1);

    tf7.multiply_and_overwrite_self(tf6);
    tf7.multiply_and_overwrite_self(tf5);
    tf7.multiply_and_overwrite_self(tf1);

    tf7.multiply_and_overwrite_self(tf4);

    output[0] = tf7.trans.x();
    output[1] = tf7.trans.y();
    output[2] = tf7.trans.z();
}

int main() {
  auto f_jit = get_jit_func();
  std::vector<double> input(7);
  std::vector<double> output(3);
  std::vector<double> output_eigen(3);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-M_PI, M_PI);
  size_t n_trials = 10000000;
  for (int i = 0; i < 7; i++) {
    input[i] = dis(gen);
  }
  f_jit(input.data(), output.data());
  auto start = std::chrono::high_resolution_clock::now();
  double sum = 0;
  for(int i = 0; i < n_trials; i++) {
    f_jit(input.data(), output.data());
    sum += output[0];
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "jit: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / n_trials << " ns" << std::endl;
  std::cout << sum << std::endl;

  eigen_counterpart(input.data(), output_eigen.data());
  start = std::chrono::high_resolution_clock::now();
  double sum_eigen = 0.0;
  for(int i = 0; i < n_trials; i++) {
    eigen_counterpart(input.data(), output_eigen.data());
    sum_eigen += output_eigen[0];
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "eigen: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / n_trials << " ns" << std::endl;
  std::cout << sum_eigen << std::endl;
}
