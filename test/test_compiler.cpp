#include "cg.hpp"
#include "compile.hpp"
#include <iostream>

using namespace tenkai;

int main() {
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
  auto i5 = i4 + i2;
  auto i6 = (i1_sq * i2_sq) + i3_sq;
  auto i7 = i1 + sin(i2 * i3);
  auto ret = i5 + i3 + i6 + i7;

  // custom compiler
  tenkai::JitFunc<double> func = compile({x, y, z, w}, {ret, i5, i3, i6, i7});
  double input[4] = {1.0, 2.0, 3.0, 4.0};
  double output[5];
  std::cout << "start calling => " << std::endl;
  func(input, output, nullptr);
  for(int i = 0; i < 5; i++) {
    std::cout << output[i] << std::endl;
  }

  // gcc
  func = jit_compile<double>({x, y, z, w}, {ret, i5, i3, i6, i7});
  std::cout << "jit => " << std::endl;
  func(input, output, {});
  for(int i = 0; i < 5; i++) {
    std::cout << output[i] << std::endl;
  }
}
