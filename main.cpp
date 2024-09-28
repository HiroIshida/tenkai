#include <iostream>
#include <string>
#include <vector>
#include "cg.hpp"
#include "linalg.hpp"

int main() {
  // simple arithmetic
  std::cout << "ARITHMETIC EXAMPLE >>" << std::endl;
  auto q0 = Operation::make_value("q0");
  auto q1 = Operation::make_value("q1");
  auto q2 = Operation::make_value("q2");
  auto m = q0 + q1 + q2;
  auto mm = m * m;
  auto mmm = m * mm + Operation::make_one();
  auto mmmm = m * mmm + Operation::make_zero();
  auto tmp = m + mm + mmm + mmmm;
  auto ret = cos(tmp) + sin(tmp) + cos(Operation::make_zero()) + sin(Operation::make_zero());
  std::vector<std::string> inputs = {"q0", "q1", "q2"};
  ret->unroll(inputs);

  // simple linalg
  std::cout << "LINALG EXAMPLE >>" << std::endl;
  auto A = Matrix3::RotX(q0);
  auto b = Vector3({q0, q1, q2});
  auto c = (A * b).sum();
  c->unroll(inputs);
}
