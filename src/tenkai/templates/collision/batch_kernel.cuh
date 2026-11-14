
extern "C" __global__ void tenkai_collision(
    const TENKAI_TEMPLATE_SCALAR* configurations,
    const uint32_t* environment_ids,
    TENKAI_TEMPLATE_NAMESPACE::Scene scene,
    unsigned char* output,
    size_t count) {
  const size_t batch =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch >= count)
    return;
  output[batch] = static_cast<unsigned char>(TENKAI_TEMPLATE_COLLISION);
}
