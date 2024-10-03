## tenkai
Given a computation graph, tenkai performs runtime C++ code generation and conducts a super-naive just-in-time (JIT) compilation. The generated 'flattened' C++ code is easily optimizable by C++ compilers because it's fully inlined and all needless operations have been removed. It's super-naive because we call `system("g++ ..")` to create a .so file and load it using `dlopen` and `dlsym` runtime;c. I'm honestly not sure if this qualifies as JIT compilation in the normal sense. An obvious todo is to generate IR code and apply LLVM compilation, but the current implementation is actually sufficient for my research purposes. 

### Internal mechanism of `tenkai::jit_compile`
Define computation graphs.
```cpp
#include "cg.hpp"
#include "linalg.hpp"

using namespace tenkai;

int main(){
  auto inp0 = Operation::make_var();
  auto inp1 = Operation::make_var();
  auto inp2 = Operation::make_var();
  {
    // check common ubexpression elimination
    auto A = Matrix::RotX(inp0);
    auto B = Matrix::RotY(inp1);
    auto v = Vector({inp0, inp1, inp2});
    auto out1 = sin((A * B * v)(0));
    auto out2 = sin((A * B * v)(0));
    auto out3 = cos((A * B * v)(1));
    auto v2 = Vector({out1, out2, out3});
    auto C = Matrix::RotZ(inp2);
    auto out4 = (C * v2)(0);
    std::vector<Operation::Ptr> inputs = {inp0, inp1, inp2};
    std::vector<Operation::Ptr> outputs = {out1, out2, out3, out4};
    tenkai::flatten("tmp1", inputs, outputs, std::cout, "double");
  }
  {
    // check constant folding
    auto out1 = (inp0 * Operation::make_constant(2.0) + inp1) * Operation::make_constant(3.0) + inp2 * Operation::make_zero();
    tenkai::flatten("tmp2", {inp0, inp1, inp2}, {out1}, std::cout, "double");
  }

}
```
The computation graph then converted into the following flattened form, which are easily be exploited by C++ compilers.
Although, the most modern c++ compiler can do, but `tenkai` does CSE and constant folding in house (for my fun).
Then the generated code is compiled and loaded as a shared object file.
```cpp
#include <cmath>
#include <functional>
extern "C" {
void tmp1(const double* input, double* output, void** extfns){
  auto var_m1470877065 = cos(input[1]);
  auto var_m285649023 = std::multiplies<double>()(var_m1470877065, input[0]);
  auto var_m574550020 = sin(input[1]);
  auto var_574550020 = std::negate<double>()(var_m574550020);
  auto var_m217038012 = std::multiplies<double>()(var_574550020, input[2]);
  auto var_m502687035 = std::plus<double>()(var_m285649023, var_m217038012);
  output[0] = sin(var_m502687035);
  output[1] = sin(var_m502687035);
  auto var_m920328699 = sin(input[0]);
  auto var_m460212244 = std::multiplies<double>()(var_m920328699, var_m574550020);
  auto var_m144106380 = std::multiplies<double>()(var_m460212244, input[0]);
  auto var_m434350112 = cos(input[0]);
  auto var_m2114739200 = std::multiplies<double>()(var_m434350112, input[1]);
  auto var_2036121716 = std::plus<double>()(var_m144106380, var_m2114739200);
  auto var_1991763539 = std::multiplies<double>()(var_m920328699, var_m1470877065);
  auto var_300680643 = std::multiplies<double>()(var_1991763539, input[2]);
  auto var_m1958164937 = std::plus<double>()(var_2036121716, var_300680643);
  output[2] = cos(var_m1958164937);
  auto var_m1122489541 = cos(input[2]);
  auto var_m1325945477 = std::multiplies<double>()(var_m1122489541, output[0]);
  auto var_m1608468128 = sin(input[2]);
  auto var_m1661828768 = std::multiplies<double>()(var_m1608468128, output[1]);
  output[3] = std::plus<double>()(var_m1325945477, var_m1661828768);
}
}
#include <cmath>
#include <functional>
extern "C" {
void tmp2(const double* input, double* output, void** extfns){
  auto var_1284742335 = std::multiplies<double>()(input[0], 2.000000);
  auto var_1784014751 = std::plus<double>()(var_1284742335, input[1]);
  output[0] = std::multiplies<double>()(var_1784014751, 3.000000);
}
}
```
