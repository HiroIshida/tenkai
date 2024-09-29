#include <gtest/gtest.h>
#include "cg.hpp"
#include "linalg.hpp"
#include "spatial.hpp"

TEST(SpatialTest, MultInverse) {
  auto r = tenkai::Operation::make_var();
  auto p = tenkai::Operation::make_var();
  auto y = tenkai::Operation::make_var();
  auto rot = tenkai::Matrix::RotX(r) * tenkai::Matrix::RotY(p) * tenkai::Matrix::RotZ(y);
  auto vec = tenkai::Vector::Var(3);
  auto tf = tenkai::SpatialTransform(rot, vec);
  auto tf_inv = tf.inverse();
  auto tf2 = tf * tf_inv;

  std::vector<tenkai::Operation::Ptr> inputs = {r, p, y};
  inputs.insert(inputs.end(), vec.elements.begin(), vec.elements.end());
  auto fn = tenkai::jit_compile<double>(inputs, {tf2.rot(0, 0), tf2.rot(1, 1), tf2.rot(2, 2), tf2.trans(0), tf2.trans(1), tf2.trans(2)});
  double input[6] = {0.1, 0.2, 0.3, 1, 2, 3};
  double output[6];
  fn(input, output);
  ASSERT_NEAR(output[0], 1, 1e-6);
  ASSERT_NEAR(output[1], 1, 1e-6);
  ASSERT_NEAR(output[2], 1, 1e-6);
  ASSERT_NEAR(output[3], 0, 1e-6);
  ASSERT_NEAR(output[4], 0, 1e-6);
  ASSERT_NEAR(output[5], 0, 1e-6);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
