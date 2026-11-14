#include <math.h>
#include <stddef.h>

extern "C" __global__ void TPL_NAME(const TPL_SCALAR* configurations,
                                    TPL_SCALAR* output,
                                    size_t count) {
  const size_t batch =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch >= count) {
    return;
  }

  /* TPL_BODY */
}
