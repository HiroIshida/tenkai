
__device__ inline bool rkern_collision_streamable(rkern_f32::Scene scene, uint32_t environment_id) {
@STREAMABLE_BODY@}

__device__ __forceinline__ rkern_f32::StreamingCollision rkern_collision_streaming(
    const float* q, rkern_f32::Scene scene, uint32_t environment_id, unsigned mask) {
    using namespace rkern_f32;
    const Environment environment = scene.environments[environment_id];
    const Box& box = scene.boxes[environment.boxes.offset];
    const Ground& ground = scene.grounds[environment.grounds.offset];
    bool collision = false, retry = false;
    Vec3 center;
@BODY@    return retry ? StreamingCollision::Retry :
        (collision ? StreamingCollision::Blocked : StreamingCollision::Clear);
}
