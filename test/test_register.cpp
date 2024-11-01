#include "cg.hpp"
#include "operation_scheduler.hpp"
#include "register_alloc.hpp"
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

  auto opseq = tenkai::compiler::DepthFirstScheduler().flatten({x, y, z, w}, {ret});
  register_alloc::RegisterAllocator regalloc(opseq, {x, y, z, w}, {ret}, 8);
  auto result = regalloc.allocate();
  for(auto& ss : result) {
    for(auto& s : ss) {
      std::cout << s;
    }
  }
}

int _main() {
  auto a = Operation::make_var();
  auto b = Operation::make_var();
  auto c = a + b; 
  auto d = a - b;
  std::vector<Operation::Ptr> inputs = {a, b};
  auto opseq = tenkai::compiler::DepthFirstScheduler().flatten(inputs, {c, d});
  register_alloc::RegisterAllocator regalloc(opseq, inputs, {c, d}, 2);
  auto result = regalloc.allocate();
}
