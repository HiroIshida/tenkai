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
    auto out2 = sin((A * B * v)(0)) + Operation::make_constant(1.0);
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
  auto var_m1479264002 = cos(input[1]);
  auto var_m1916414764 = std::multiplies<double>()(var_m1479264002, input[0]);
  auto var_m1965242589 = sin(input[1]);
  auto var_1965242589 = std::negate<double>()(var_m1965242589);
  auto var_1667816900 = std::multiplies<double>()(var_1965242589, input[2]);
  auto var_m248597864 = std::plus<double>()(var_m1916414764, var_1667816900);
  auto var_1989960210 = sin(var_m248597864);
  output[0] = var_1989960210;
  auto var_m683411657 = std::plus<double>()(var_1989960210, 1.000000);
  output[1] = var_m683411657;
  auto var_m1943809953 = sin(input[0]);
  auto var_1199789565 = std::multiplies<double>()(var_m1943809953, var_m1965242589);
  auto var_687540798 = std::multiplies<double>()(var_1199789565, input[0]);
  auto var_m1457831366 = cos(input[0]);
  auto var_1932391052 = std::multiplies<double>()(var_m1457831366, input[1]);
  auto var_m1675035446 = std::plus<double>()(var_687540798, var_1932391052);
  auto var_1758037570 = std::multiplies<double>()(var_m1943809953, var_m1479264002);
  auto var_m1094366680 = std::multiplies<double>()(var_1758037570, input[2]);
  auto var_1525565170 = std::plus<double>()(var_m1675035446, var_m1094366680);
  auto var_1099533184 = cos(var_1525565170);
  output[2] = var_1099533184;
  auto var_m303140554 = cos(input[2]);
  auto var_m486366772 = std::multiplies<double>()(var_m303140554, var_1989960210);
  auto var_m789119141 = sin(input[2]);
  auto var_2000857485 = std::multiplies<double>()(var_m789119141, var_m683411657);
  auto var_1514490713 = std::plus<double>()(var_m486366772, var_2000857485);
  output[3] = var_1514490713;
}
}
#include <cmath>
#include <functional>
extern "C" {
void tmp2(const double* input, double* output, void** extfns){
  auto var_1705639868 = std::multiplies<double>()(input[0], 2.000000);
  auto var_m1340616390 = std::plus<double>()(var_1705639868, input[1]);
  auto var_1199161088 = std::multiplies<double>()(var_m1340616390, 3.000000);
  output[0] = var_1199161088;
}
}
```
