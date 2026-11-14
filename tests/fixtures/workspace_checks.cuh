struct WorkspaceCase {
    float lower[2], upper[2], z;
    uint32_t environment;
    bool clear;
};

__global__ void check_workspace_trigonometry() {
    const float limit = float(100 * std::acos(-1.0));
    const float inside = nextafterf(limit, 0.0f);
    for (bool cosine : {false, true}) {
        const auto bounds = m::workspace_trig({-inside, inside}, cosine);
        assert(m::workspace_finite(bounds));
        for (unsigned i = threadIdx.x; i <= 4096; i += blockDim.x) {
            const float angle = inside * (float(i) / 2048 - 1);
#ifdef TEST_STANDARD_FP32
            const float value = cosine ? cosf(angle) : sinf(angle);
#else
            const float value = cosine ? __cosf(angle) : __sinf(angle);
#endif
            assert(bounds.lower <= value && value <= bounds.upper);
        }
        assert(!m::workspace_finite(m::workspace_trig({-INFINITY, INFINITY}, cosine)));
        assert(!m::workspace_finite(m::workspace_trig({NAN, NAN}, cosine)));
        const auto zero = m::workspace_trig({-0.0f, 0.0f}, cosine);
        assert(zero.lower == (cosine ? 1 : 0) && zero.upper == zero.lower);
    }
}

__global__ void check_workspace_cases(const WorkspaceCase* cases, uint32_t count, m::Scene scene) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const WorkspaceCase& c = cases[i];
    const float reference[3] = {0, 0, c.z};
    const bool clear = tenkai_collision_workspace_clear(c.lower, c.upper, reference, scene, c.environment);
    assert(clear == c.clear);
    if (clear) {
        for (unsigned y = 0; y <= 16; ++y) {
            for (unsigned x = 0; x <= 16; ++x) {
                float q[3] = {c.lower[1] + float(x) / 16 * (c.upper[1] - c.lower[1]),
                              c.lower[0] + float(y) / 16 * (c.upper[0] - c.lower[0]), c.z};
                assert(tenkai_collision_edge(q, scene, c.environment) ==
                       tenkai_collision_moving(q, scene, c.environment));
            }
        }
    }
    const m::WorkspaceInterval unknown = m::workspace_unknown();
    assert(!m::workspace_finite(m::workspace_mul(unknown, {0, 0})));
    assert(!m::workspace_finite(m::workspace_add({FLT_MAX, FLT_MAX}, {FLT_MAX, FLT_MAX})));
    const float tiny = FLT_MIN / 4;
    const auto sum = m::workspace_add({tiny, tiny}, {tiny, tiny});
    assert(sum.lower <= 2 * tiny && sum.upper >= 2 * tiny);
}

void check_workspace_bounds() {
    check_workspace_trigonometry<<<1, 32>>>();
    m::cuda_check(cudaGetLastError(), "check workspace trigonometry");
    m::cuda_check(cudaDeviceSynchronize(), "validate workspace trigonometry");

    const float s = std::sqrt(0.5f);
    m::Box boxes[] = {
        {{{0, 0, 0}, {1,0,0,0,1,0,0,0,1}}, {0.2f,0.8f,0.2f}},
        {{{0, 0, 0}, {s,s,0,-s,s,0,0,0,1}}, {0.2f,0.8f,0.2f}},
    };
    const m::Sphere sphere{{0,0,0}, 0.1f};
    const m::Cylinder cylinder{{{0,0,0}, {1,0,0,0,1,0,0,0,1}}, 0.1f, 0.2f};
    const m::Ground ground{0};
    const m::Environment environments[] = {
        {}, {{0,1},{},{},{}}, {{},{0,1},{},{}}, {{},{},{0,1},{}},
        {{},{},{},{0,1}}, {{1,1},{},{},{}}, {{0,2},{0,1},{0,1},{0,1}},
    };
    m::DeviceBuffer<m::Box> device_boxes(2);
    m::DeviceBuffer<m::Sphere> device_sphere(1);
    m::DeviceBuffer<m::Cylinder> device_cylinder(1);
    m::DeviceBuffer<m::Ground> device_ground(1);
    m::DeviceBuffer<m::Environment> device_environments(7);
    device_boxes.upload(boxes, 2);
    device_sphere.upload(&sphere, 1);
    device_cylinder.upload(&cylinder, 1);
    device_ground.upload(&ground, 1);
    device_environments.upload(environments, 7);
    const m::Scene scene{device_environments.data(), device_boxes.data(), device_sphere.data(),
                         device_cylinder.data(), device_ground.data()};
    std::vector<WorkspaceCase> cases;
    for (uint32_t environment = 0; environment < 7; ++environment) {
        cases.push_back({{-0.1f,-1}, {0.1f,-0.5f}, 0.25f, environment, true});
        cases.push_back({{-0.1f,-0.1f}, {0.1f,0.1f}, 0, environment, environment == 0});
        cases.push_back({{-0.1f,-0.1f}, {0.1f,0.1f}, 2, environment, true});
    }
    cases.push_back({{-0.01f,0.15f}, {0.01f,0.2f}, 0, 1, false});
    cases.push_back({{-0.1f,-1}, {0.1f,-0.5f}, 0.05f, 4, false});
    cases.push_back({{-FLT_MAX,-FLT_MAX}, {FLT_MAX,FLT_MAX}, 0, 1, false});
    m::DeviceBuffer<WorkspaceCase> device_cases(cases.size());
    device_cases.upload(cases.data(), cases.size());
    check_workspace_cases<<<1, 32>>>(device_cases.data(), cases.size(), scene);
    m::cuda_check(cudaGetLastError(), "check certified workspace bounds");
    m::cuda_check(cudaDeviceSynchronize(), "validate certified workspace bounds");
    // The same scene pointers must observe updated geometry on the next solve.
    boxes[0].pose.position.x = 5;
    device_boxes.upload(boxes, 2);
    WorkspaceCase changed{{-0.1f,-0.1f}, {0.1f,0.1f}, 0, 1, true};
    device_cases.upload(&changed, 1);
    check_workspace_cases<<<1, 1>>>(device_cases.data(), 1, scene);
    m::cuda_check(cudaGetLastError(), "check updated workspace geometry");
    m::cuda_check(cudaDeviceSynchronize(), "validate updated workspace geometry");
}
