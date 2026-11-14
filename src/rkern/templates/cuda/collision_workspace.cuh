
__device__ inline bool rkern_collision_in_workspace(const float* q, const float* lower, const float* upper) {
@BOUNDS_CHECKS@    return true;
}
@INTERVAL_HELPERS@
__device__ __noinline__ bool rkern_collision_workspace_clear(
    const float* lower, const float* upper, const float* reference,
    rkern_f32::Scene scene, uint32_t environment_id) {
    using namespace rkern_f32;
    const Environment environment = scene.environments[environment_id];
@BODY@    return true;
}
