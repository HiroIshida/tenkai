#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "collision.cuh"
#include "workload.hpp"

using Clock = std::chrono::steady_clock;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}
#define CUDA(operation) check((operation), #operation)

enum class Mode { Kernel, Transfer };
struct Job { size_t batch; Mode mode; size_t repetitions; };
struct Timing { double gpu_ms, wall_ms; };

int main(int argc, char** argv) try {
    if (argc != 2) {
        std::cerr << "usage: collision_bench <data-directory>\n";
        return 2;
    }
    const std::string directory = argv[1];
    cudaDeviceProp device;
    CUDA(cudaGetDeviceProperties(&device, 0));
    if (kBlock > device.maxThreadsPerBlock) {
        std::cerr << "block size exceeds the device limit\n";
        return 2;
    }

    Scalar* host_q;
    unsigned char* host_result;
    CUDA(cudaMallocHost(&host_q, kPool * kDof * sizeof(Scalar)));
    CUDA(cudaMallocHost(&host_result, kPool));
    read_file(directory + "/configurations.bin", host_q, kPool * kDof);
    Scalar geometry[7];
    read_file(directory + "/scene.bin", geometry, 7);
    const sdf::Box box = {{{geometry[0], geometry[1], geometry[2]},
                          {1,0,0,0,1,0,0,0,1}},
                         {geometry[3], geometry[4], geometry[5]}};
    const sdf::Ground ground = {geometry[6]};
    const sdf::Environment environment = {{0,1},{0,0},{0,0},{0,1}};
    sdf::Box* d_box;
    sdf::Ground* d_ground;
    sdf::Environment* d_environment;
    Scalar *d_q, *d_transfer_q;
    unsigned char *d_result, *d_transfer_result;
    uint32_t* d_ids;
    CUDA(cudaMalloc(&d_box, sizeof(box)));
    CUDA(cudaMalloc(&d_ground, sizeof(ground)));
    CUDA(cudaMalloc(&d_environment, sizeof(environment)));
    CUDA(cudaMalloc(&d_q, kPool * kDof * sizeof(Scalar)));
    CUDA(cudaMalloc(&d_transfer_q, kMaxBatch * kDof * sizeof(Scalar)));
    CUDA(cudaMalloc(&d_result, kPool));
    CUDA(cudaMalloc(&d_transfer_result, kMaxBatch));
    CUDA(cudaMalloc(&d_ids, kPool * sizeof(uint32_t)));
    CUDA(cudaMemcpy(d_box, &box, sizeof(box), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_ground, &ground, sizeof(ground), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_environment, &environment, sizeof(environment), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_q, host_q, kPool * kDof * sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA(cudaMemset(d_ids, 0, kPool * sizeof(uint32_t)));
    const sdf::Scene scene = {d_environment, d_box, nullptr, nullptr, d_ground};

    CUDA(cudaMemset(d_result, 123, kPool));
    tenkai_collision<<<(kPool+kBlock-1)/kBlock, kBlock>>>(d_q, d_ids, scene, d_result, kPool);
    CUDA(cudaGetLastError());
    CUDA(cudaMemcpy(host_result, d_result, kPool, cudaMemcpyDeviceToHost));
    std::ofstream flags(directory + "/collision.u8", std::ios::binary);
    flags.write(reinterpret_cast<const char*>(host_result), kPool);
    flags.close();
    if (!flags) return 1;

    cudaFuncAttributes attributes;
    CUDA(cudaFuncGetAttributes(&attributes, tenkai_collision));
    int active_blocks, driver_version, runtime_version;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, tenkai_collision, kBlock, 0));
    CUDA(cudaDriverGetVersion(&driver_version));
    CUDA(cudaRuntimeGetVersion(&runtime_version));
    std::ofstream info(directory + "/device.json");
    info << "{\"name\":" << std::quoted(device.name)
         << ",\"compute_capability\":\"" << device.major << '.' << device.minor << '"'
         << ",\"sms\":" << device.multiProcessorCount
         << ",\"driver_version\":" << driver_version << ",\"runtime_version\":" << runtime_version
         << ",\"registers_per_thread\":" << attributes.numRegs
         << ",\"local_bytes_per_thread\":" << attributes.localSizeBytes
         << ",\"active_blocks_per_sm\":" << active_blocks << "}\n";
    info.close();
    if (!info) return 1;

    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
    auto measure = [&](const Job& job, size_t round, size_t repetitions) {
        const auto wall_start = Clock::now();
        CUDA(cudaEventRecord(start));
        for (size_t iteration = 0; iteration < repetitions; ++iteration) {
            const size_t offset = input_offset(job.batch, round, iteration);
            const size_t grid = (job.batch + kBlock - 1) / kBlock;
            if (job.mode == Mode::Transfer) {
                CUDA(cudaMemcpyAsync(d_transfer_q, host_q + offset*kDof,
                                     job.batch*kDof*sizeof(Scalar), cudaMemcpyHostToDevice));
                tenkai_collision<<<grid, kBlock>>>(d_transfer_q, d_ids, scene, d_transfer_result, job.batch);
                CUDA(cudaMemcpyAsync(host_result, d_transfer_result, job.batch, cudaMemcpyDeviceToHost));
            } else {
                tenkai_collision<<<grid, kBlock>>>(d_q + offset*kDof, d_ids + offset, scene, d_result + offset, job.batch);
            }
        }
        CUDA(cudaEventRecord(stop));
        CUDA(cudaEventSynchronize(stop));
        CUDA(cudaGetLastError());
        float elapsed;
        CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        const double wall = std::chrono::duration<double, std::milli>(Clock::now()-wall_start).count();
        return Timing{elapsed / repetitions, wall / repetitions};
    };

    const Job warmup = {std::min(kPool, size_t(65536)), Mode::Kernel, 8};
    const auto warm_start = Clock::now();
    do { measure(warmup, 0, warmup.repetitions); }
    while (std::chrono::duration<double>(Clock::now()-warm_start).count() < kWarmupSeconds);

    std::vector<Job> jobs;
    for (size_t batch : kBatches) {
        const size_t repetitions = repetitions_for(batch);
        for (Mode mode : {Mode::Kernel, Mode::Transfer})
            jobs.push_back({batch, mode, repetitions});
    }
    std::cout << "mode,batch,round,repetitions,gpu_ms,wall_ms\n" << std::setprecision(12);
    for (size_t round = 0; round < kRounds; ++round) {
        for (size_t i = 0; i < jobs.size(); ++i) {
            const Job& job = jobs[job_index(round, i, jobs.size())];
            measure(job, round, 2);
            const Timing timing = measure(job, round, job.repetitions);
            std::cout << (job.mode == Mode::Kernel ? "kernel" : "transfer") << ',' << job.batch << ','
                      << round << ',' << job.repetitions << ',' << timing.gpu_ms << ',' << timing.wall_ms << '\n';
        }
        std::cout.flush();
        std::cerr << "measurement round " << round+1 << '/' << kRounds << " complete\n";
    }
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
    CUDA(cudaFree(d_box)); CUDA(cudaFree(d_ground)); CUDA(cudaFree(d_environment));
    CUDA(cudaFree(d_q)); CUDA(cudaFree(d_transfer_q)); CUDA(cudaFree(d_result));
    CUDA(cudaFree(d_transfer_result)); CUDA(cudaFree(d_ids));
    CUDA(cudaFreeHost(host_q)); CUDA(cudaFreeHost(host_result));
    return std::cout ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
