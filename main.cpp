#include <string>
#include <vector>
#include "cg.hpp"

int main() {
  auto q0 = Operation::make_value("q0");
  auto q1 = Operation::make_value("q1");
  auto q2 = Operation::make_value("q2");
  auto m = q0 + q1 + q2;
  auto mm = m * m;
  auto mmm = m * mm;
  auto mmmm = m * mmm;
  auto tmp = m + mm + mmm + mmmm;
  auto ret = cos(tmp) + sin(tmp);
  std::vector<std::string> inputs = {"q0", "q1", "q2"};
  ret->unroll(inputs);
}
