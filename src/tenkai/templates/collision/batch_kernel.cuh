extern "C" __global__ void tenkai_collision(const float* configurations,
                                            const uint32_t* environment_ids,
                                            tenkai_f32::Scene scene,
                                            unsigned char* output,
                                            size_t count) {
  const size_t batch =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch >= count) {
    return;
  }

  output[batch] = static_cast<unsigned char>(TPL_COLLISION);
}
