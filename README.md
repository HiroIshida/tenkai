## tenkai
Given a computation graph, tenkai performs runtime C++ code generation and conducts a super-naive just-in-time (JIT) compilation. The generated 'flattened' C++ code is easily optimizable by C++ compilers because it's fully inlined and all needless operations have been removed. It's super-naive because we call `system("g++ ..")` to create a .so file and load it using `dlopen` and `dlsym` runtime;c. I'm honestly not sure if this qualifies as JIT compilation in the normal sense. An obvious todo is to generate IR code and apply LLVM compilation, but the current implementation is actually sufficient for my research purposes. 

### example
Create computation graph
```cpp
#include <string>
#include <vector>
#include "cg.hpp"
#include "linalg.hpp"

using namespace tenkai;

int main() {
  // define computation graph
  auto inp0 = Operation::make_var();
  auto inp1 = Operation::make_var();
  auto inp2 = Operation::make_var();

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

  // define inputs and outputs order
  std::vector<Operation::Ptr> inputs = {inp0, inp1, inp2};
  std::vector<Operation::Ptr> outputs = {out1, out2, out3, out4, out5};

  // compile
  auto f = jit_compile<double>(inputs, outputs, "g++"); // or clang

  // run
  double input[3] = {0.1, 0.2, 0.3};
  double output[5];
  f(input, output);
```

Also you can check the intermediate flattend c++ code by calling `tenkai::flatten` function as
```cpp
void example(double* input, double* output) {
  auto dwbzUTzq = cos(input[0]);
  auto lwAxgRDg = dwbzUTzq * input[1];
  auto qNdstjvw = sin(input[0]);
  auto eXHyTQTO = -qNdstjvw;
  auto elopaBBm = eXHyTQTO * input[2];
  auto Bnupuqsp = lwAxgRDg + elopaBBm;
  auto TSnbJWKU = input[0] + Bnupuqsp;
  auto hQGcnOKD = qNdstjvw * input[1];
  auto zfIHkjFV = dwbzUTzq * input[2];
  auto FSLskBio = hQGcnOKD + zfIHkjFV;
  output[0] = TSnbJWKU + FSLskBio;
  auto uJFXGMRH = cos(input[1]);
  auto eGwvzjyz = uJFXGMRH * input[0];
  auto EQuNFvMA = sin(input[1]);
  auto lYsDjtaE = EQuNFvMA * FSLskBio;
  auto eaxqUlXa = eGwvzjyz + lYsDjtaE;
  auto CsXNJEPw = eaxqUlXa + Bnupuqsp;
  auto EXmcHEAE = -EQuNFvMA;
  auto jeeZzbFM = EXmcHEAE * input[0];
  auto ptZexfWE = uJFXGMRH * FSLskBio;
  auto IGPdaMmu = jeeZzbFM + ptZexfWE;
  output[1] = CsXNJEPw + IGPdaMmu;
  auto DZkEKwsk = cos(input[2]);
  auto IrfDMJtL = DZkEKwsk * eaxqUlXa;
  auto vLdyzOCi = sin(input[2]);
  auto iTgxlcMO = -vLdyzOCi;
  auto yuZLwGvJ = iTgxlcMO * Bnupuqsp;
  output[3] = IrfDMJtL + yuZLwGvJ;
  auto SSHbGoZS = output[3] * output[3];
  auto zEQaFNol = vLdyzOCi * eaxqUlXa;
  auto TYuLmFKd = DZkEKwsk * Bnupuqsp;
  auto hwboVRFx = zEQaFNol + TYuLmFKd;
  auto baMuUnQO = hwboVRFx * hwboVRFx;
  auto DwqCExfX = SSHbGoZS + baMuUnQO;
  auto ZPXOwIZw = IGPdaMmu * IGPdaMmu;
  output[2] = DwqCExfX + ZPXOwIZw;
  auto UpEXHGtz = input[0] + eaxqUlXa;
  auto NYUkqbFB = UpEXHGtz + output[3];
  auto FnCMIKDs = NYUkqbFB * NYUkqbFB;
  auto LcYQePZe = Bnupuqsp + Bnupuqsp;
  auto fYzyZZFY = LcYQePZe + hwboVRFx;
  auto TAICPxng = fYzyZZFY * fYzyZZFY;
  auto TFoMIwyz = FnCMIKDs + TAICPxng;
  auto JBnLAebQ = FSLskBio + IGPdaMmu;
  auto keTiCZUh = JBnLAebQ + IGPdaMmu;
  auto WdNJPUcP = keTiCZUh * keTiCZUh;
  output[4] = TFoMIwyz + WdNJPUcP;
}
```
