    const auto environment = scene.environments[environment_id];
    if (environment.boxes.count != 1 || environment.grounds.count != 1 ||
        environment.spheres.count || environment.cylinders.count) return false;
    const auto& box = scene.boxes[environment.boxes.offset];
    for (unsigned i = 0; i < 9; ++i)
        if (box.pose.world_to_local[i] != (i % 4 == 0 ? 1.0f : 0.0f)) return false;
    return true;
