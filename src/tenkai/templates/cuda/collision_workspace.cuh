
__device__ inline bool tenkai_collision_in_workspace(const float* q, const float* lower, const float* upper) {
@BOUNDS_CHECKS@    return true;
}
@INTERVAL_HELPERS@
__device__ __noinline__ bool tenkai_collision_workspace_clear(
    const float* lower, const float* upper, const float* reference,
    tenkai_f32::Scene scene, uint32_t environment_id) {
    using namespace tenkai_f32;
    const Environment environment = scene.environments[environment_id];
@BODY@    return true;
}
