#include <iostream>
#include <string>
#include <vector>
#include "cg.hpp"
#include "linalg.hpp"

int main() {
  auto inp0 = Operation::make_var("q0");
  auto inp1 = Operation::make_var("q1");
  auto inp2 = Operation::make_var("q2");

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
  auto out4 = CBAv.get(0);
  auto out5 = (Av + BAv + CBAv).sqnorm();

  std::vector<Operation::Ptr> inputs = {inp0, inp1, inp2};
  std::vector<Operation::Ptr> outputs = {out1, out2, out3, out4, out5};

  std::string func_name = "example";
  flatten(func_name, inputs, outputs, std::cout, "double");

  auto f = jit_compile<double>(inputs, outputs, "g++"); // or clang
  double input[3] = {0.1, 0.2, 0.3};
  double output[5];
  f(input, output);
  std::cout << "out1: " << output[0] << std::endl;
  std::cout << "out2: " << output[1] << std::endl;
  std::cout << "out3: " << output[2] << std::endl;
  std::cout << "out4: " << output[3] << std::endl;
  std::cout << "out5: " << output[4] << std::endl;
}
