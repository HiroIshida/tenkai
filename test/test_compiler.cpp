#include "cg.hpp"
#include "linalg.hpp"
#include "compile.hpp"
#include <iostream>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>
#include <cstdint>

using namespace tenkai;

// int main() {
//    auto x = Operation::make_var();
//    auto y = Operation::make_var();
//    auto z = Operation::make_var();
//    auto w = Operation::make_var();
//    auto ret = x + y + z + w;
//    auto ret2 = x * y;
//    auto fin = ret + ret2;
//    auto code = compile({x, y, z, w}, {fin});
// 
//    size_t page_size = getpagesize();
//    void *mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
//    uint8_t *instruction = static_cast<uint8_t *>(mem);
//  
//    std::memcpy(instruction, code.data(), code.size());
//    mprotect(mem, page_size, PROT_READ | PROT_EXEC) == -1;
//  
//    using Func = void(*)(double*, double*);
//    Func add_func = reinterpret_cast<Func>(instruction);
// 
//    double input[4] = {1.0, 2.0, 3.0, 4.0};
//    double output;
//    add_func(input, &output);
//    std::cout << output << std::endl;
// 
//    // jit version
// 
//    auto fjit = jit_compile<double>({x, y, z, w}, {fin}, "g++", true);
//    std::cout << "jit => " << std::endl;
//    fjit(input, &output, {});
//    std::cout << output << std::endl;
// }

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
  auto i4 = i1_sq + i2_sq + i3_sq;
  auto i5 = i4 + i2;
  auto i6 = (i1_sq * i2_sq) + i3_sq;
  auto i7 = i1 + (i2 * i3);
  auto ret = i5 + i3 + i6 + i7;
  std::vector<uint8_t> code = compile({x, y, z, w}, {ret, i5, i3, i6, i7});
  std::cout << "code size is : " << code.size() << std::endl;

  size_t page_size = getpagesize();
  void *mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  uint8_t *instruction = static_cast<uint8_t *>(mem);

  std::memcpy(instruction, code.data(), code.size());
  mprotect(mem, page_size, PROT_READ | PROT_EXEC) == -1;

  using Func = void(*)(double*, double*);
  Func add_func = reinterpret_cast<Func>(instruction);

  double input[4] = {1.0, 2.0, 3.0, 4.0};
  double output[5];
  std::cout << "start calling => " << std::endl;
  add_func(input, output);
  for(int i = 0; i < 5; i++) {
    std::cout << output[i] << std::endl;
  }

  // jit 
  // tenkai::flatten("tmp", {x, y, z, w}, {ret}, std::cout, "double");
  auto fjit = jit_compile<double>({x, y, z, w}, {ret, i5, i3, i6, i7});
  std::cout << "jit => " << std::endl;
  fjit(input, output, {});
  for(int i = 0; i < 5; i++) {
    std::cout << output[i] << std::endl;
  }
}
