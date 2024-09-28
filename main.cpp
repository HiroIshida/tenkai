#include <iostream>
#include <string>
#include <vector>
#include "cg.hpp"
#include "linalg.hpp"

int main() {
  auto q0 = Operation::make_var("q0");
  auto q1 = Operation::make_var("q1");
  auto q2 = Operation::make_var("q2");
  std::vector<std::string> inputs = {"q0", "q1", "q2"};

  {
    std::cout << "ARITHMETIC EXAMPLE >>" << std::endl;
    auto m = q0 + q1 + q2;
    auto mm = m * m;
    auto mmm = m * mm + Operation::make_one();
    auto mmmm = m * mmm + Operation::make_zero();
    auto tmp = m + mm + mmm + mmmm;
    auto ret = cos(tmp) + sin(tmp) + cos(Operation::make_zero()) + sin(Operation::make_zero());
    ret->unroll(inputs);
  }

  {
    std::cout << "LINALG EXAMPLE >>" << std::endl;
    auto A = Matrix3::RotX(q0);
    auto B = Matrix3::RotY(q1);
    auto C = Matrix3::RotZ(q2);
    auto b = Vector3({q0, q1, q2});
    auto c = ((A * b) + (B * b) + (C * b)).sum();
    c->unroll(inputs);
  }
}
