#include <stddef.h>
#include <math.h>

extern "C" __global__ void @NAME@(
    const @SCALAR@* configurations, @SCALAR@* output, size_t count) {
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
@BODY@}
