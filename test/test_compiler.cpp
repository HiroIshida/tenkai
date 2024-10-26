#include "cg.hpp"
#include "compile.hpp"
#include <iostream>
#include <gtest/gtest.h>

using namespace tenkai;

TEST(Compiler, Basic) {
  auto x = Operation::make_var();
  auto y = Operation::make_var();
  auto z = Operation::make_var();
  auto w = Operation::make_var();

  auto i1 = x + y + z + w;
  auto i2 = x - y - z - w;
  auto i3 = x * y * z * w;
  auto i1_sq = i1 * i1;
  auto i2_sq = i2 * i2;
  auto i3_sq = i3 * i3;
  auto i4 = cos(i1_sq) + i2_sq + i3_sq;
  auto i5 = i4 + i2 + Operation::make_constant(0.3);
  auto i6 = (i1_sq * i2_sq) + i3_sq;
  auto i7 = i1 + sin(i2 * i3);
  auto ret = i5 + i3 + i6 + i7;
  auto ret2 = -ret;

  // custom compiler
  std::vector<Operation::Ptr> output = {i7, ret, ret2};
  tenkai::JitFunc<double> func = compiler::compile({x, y, z, w}, output);
  double input[4] = {1.0, 2.0, 3.0, 4.0};
  double output_custom[5];
  func(input, output_custom, {});

  // gcc
  double output_gcc[5];
  func = jit_compile<double>({x, y, z, w}, output);
  func(input, output_gcc, {});

  // compare
  for (int i = 0; i < 3; i++) {
    // print
    std::cout << "output_custom[" << i << "] = " << output_custom[i] << std::endl;
    std::cout << "output_gcc[" << i << "] = " << output_gcc[i] << std::endl;
    ASSERT_NEAR(output_custom[i], output_gcc[i], 1e-6);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
