constexpr unsigned int broad_phase_case_count = 2 + 3 * 4 * 9 * 9 * 9 * 4;

__device__ inline bool check_broad_phase_case(unsigned int index) {
    if (index < 2) {
        const sdf::Box box = {{{0,0,0},{1,0,0,0,1,0,0,0,1}}, {2,2,2}};
        const sdf::SphereBounds bounds = {{3,0,0},{4,1,1},index ? Scalar(2) : Scalar(0.5)};
        return sdf::overlaps(box,bounds) == bool(index);
    }
    index -= 2;
    const unsigned int radius_index = index % 4; index /= 4;
    const int z = int(index % 9) - 4; index /= 9;
    const int y = int(index % 9) - 4; index /= 9;
    const int x = int(index % 9) - 4; index /= 9;
    const unsigned int pose_index = index % 4; index /= 4;
    // Exact scaling keeps contact cases independent of rounding margins.
    const Scalar scales[] = {Scalar(1), Scalar(1) / Scalar(1048576), Scalar(1048576)};
    const sdf::Pose poses[] = {
        {{0,0,0}, {1,0,0,0,1,0,0,0,1}},
        {{2,-3,1}, {0,1,0,-1,0,0,0,0,1}},
        {{-1,2,-3}, {Scalar(0.6),Scalar(0.8),0,Scalar(-0.8),Scalar(0.6),0,0,0,1}},
        {{1,-2,3}, {0,Scalar(0.6),Scalar(0.8),0,Scalar(-0.8),Scalar(0.6),1,0,0}}
    };
    const Scalar scale = scales[index];
    auto pose = poses[pose_index];
    pose.position.x *= scale;
    pose.position.y *= scale;
    pose.position.z *= scale;
    const sdf::Box box = {pose, {2*scale,4*scale,6*scale}};
    const sdf::Sphere sphere = {pose.position, 2*scale};
    const sdf::Cylinder cylinder = {pose, scale, 4*scale};
    const sdf::Vec3 point = {pose.position.x + Scalar(x)*scale,
                             pose.position.y + Scalar(y)*scale,
                             pose.position.z + Scalar(z)*scale};
    const Scalar radii[] = {Scalar(0), scale/Scalar(4), scale, 2*scale};
    const Scalar radius = radii[radius_index];
    const sdf::SphereBounds bounds = {point, point, radius};
    return !(sdf::intersects(box,point,radius) && !sdf::overlaps(box,bounds)) &&
           !(sdf::intersects(sphere,point,radius) && !sdf::overlaps(sphere,bounds)) &&
           !(sdf::intersects(cylinder,point,radius) && !sdf::overlaps(cylinder,bounds));
}
