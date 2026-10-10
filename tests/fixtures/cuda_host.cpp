#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

// CPU execution lets codegen indexing and arithmetic be tested without a GPU.
#define __global__
struct Dim { unsigned int x; };
Dim blockIdx{}, blockDim{}, threadIdx{};

#include "kernels.cuh"
#include "expected.h"

int main() {
    constexpr size_t capacity = 16;
    constexpr double sentinel = 12345.0;
    blockDim.x = 4;
    for (size_t kernel = 0; kernel < kernel_count; ++kernel) {
        for (size_t count : {size_t(0), size_t(1), configuration_count}) {
            double output[capacity * 7];
            std::fill(output, output + capacity * 7, sentinel);
            for (blockIdx.x = 0; blockIdx.x < 4; ++blockIdx.x) {
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                    kernels[kernel](inputs, output, count);
                }
            }
            for (size_t i = 0; i < capacity * 7; ++i) {
                const double value = i < count * 7
                    ? expected[kernel * configuration_count * 7 + i] : sentinel;
                if (!std::isfinite(output[i]) || std::abs(output[i] - value) > 1e-12) {
                    std::fprintf(stderr, "kernel=%zu count=%zu element=%zu: %.17g != %.17g\n",
                                 kernel, count, i, output[i], value);
                    return 1;
                }
            }
        }
        double output[7];
        std::fill(output, output + 7, sentinel);
        blockIdx.x = 1u << 30;
        threadIdx.x = 0;
        kernels[kernel](inputs, output, 1);
        if (!std::all_of(output, output + 7, [](double v) { return v == sentinel; })) {
            std::fprintf(stderr, "thread index overflow in kernel %zu\n", kernel);
            return 1;
        }
    }
}
