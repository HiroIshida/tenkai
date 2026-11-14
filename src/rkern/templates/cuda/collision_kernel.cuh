
extern "C" __global__ void rkern_collision(
    const @SCALAR@* configurations, const uint32_t* environment_ids,
    @NAMESPACE@::Scene scene, unsigned char* output, size_t count) {
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
    output[batch] = static_cast<unsigned char>(@COLLISION@);
}
