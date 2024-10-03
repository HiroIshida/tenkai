#include "cg.hpp"
#include "compile.hpp"
#include <iostream>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>
#include <cstdint>

using namespace tenkai;

int main() {
  auto a = Operation::make_var();
  auto b = Operation::make_var();
  auto c = a + b;
  std::vector<uint8_t> code = compile({a, b}, {c});

  size_t page_size = getpagesize();
  void *mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  uint8_t *instruction = static_cast<uint8_t *>(mem);

  std::memcpy(instruction, code.data(), code.size());
  mprotect(mem, page_size, PROT_READ | PROT_EXEC) == -1;

  using Func = void(*)(double*, double*);
  Func add_func = reinterpret_cast<Func>(instruction);

  double input[2] = {1.0, 2.0};
  double output;
  add_func(input, &output);
  std::cout << output << std::endl;
}
