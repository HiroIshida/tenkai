#include <math.h>
#include <stddef.h>

extern "C" __global__ void TENKAI_TEMPLATE_NAME(
    const TENKAI_TEMPLATE_SCALAR* configurations,
    TENKAI_TEMPLATE_SCALAR* output,
    size_t count) {
  const size_t batch =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch >= count)
    return;
  /* TENKAI_TEMPLATE_BODY */
}
