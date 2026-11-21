#ifndef tenkai_f32_BROAD_INCLUDED
#define tenkai_f32_BROAD_INCLUDED
#include "sdf.cuh"

namespace tenkai_f32 {
template <typename Obstacle>
__device__ inline bool may_intersect(const Obstacle& obstacle,
                                     const Sphere& enclosing) {
  return intersects(obstacle, enclosing.position, enclosing.radius);
}

template <unsigned Types = AllTypes>
__device__ inline bool may_intersect(const Scene& scene,
                                     const Environment& env,
                                     const Sphere& enclosing) {
  for (size_t i = 0; i < (Types & BoxType ? env.boxes.count : 0); ++i) {
    if (may_intersect(scene.boxes[env.boxes.offset + i], enclosing))
      return true;
  }
  for (size_t i = 0; i < (Types & SphereType ? env.spheres.count : 0); ++i) {
    if (may_intersect(scene.spheres[env.spheres.offset + i], enclosing))
      return true;
  }
  for (size_t i = 0; i < (Types & CylinderType ? env.cylinders.count : 0);
       ++i) {
    if (may_intersect(scene.cylinders[env.cylinders.offset + i], enclosing))
      return true;
  }
  for (size_t i = 0; i < (Types & GroundType ? env.grounds.count : 0); ++i) {
    if (may_intersect(scene.grounds[env.grounds.offset + i], enclosing))
      return true;
  }
  return false;
}

}  // namespace tenkai_f32
#endif
