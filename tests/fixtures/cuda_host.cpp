#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

// CPU execution lets codegen indexing and arithmetic be tested without a GPU.
#define __global__
struct Dim { unsigned int x; };
Dim blockIdx{}, blockDim{}, threadIdx{};

#include "kernels.cuh"
#include "expected.h"

int main() {
    constexpr size_t capacity = 16;
    constexpr Scalar sentinel = 12345;
    blockDim.x = 4;
    for (const Case& test : cases) {
        for (size_t count : {size_t(0), size_t(1), test.count}) {
            std::vector<Scalar> output(capacity * test.components + 2, sentinel);
            for (blockIdx.x = 0; blockIdx.x < 4; ++blockIdx.x) {
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                    test.kernel(test.inputs, output.data() + 1, count);
                }
            }
            for (size_t i = 0; i < output.size(); ++i) {
                const double value = i > 0 && i <= count * test.components
                    ? test.expected[i - 1] : sentinel;
                if (!std::isfinite(output[i]) || std::abs(output[i] - value) > tolerance) {
                    std::fprintf(stderr, "%s count=%zu element=%zu: %.17g != %.17g\n",
                                 test.name, count, i, double(output[i]), value);
                    return 1;
                }
            }
        }
        std::vector<Scalar> output(test.components + 2, sentinel);
        blockIdx.x = 1u << 30;
        threadIdx.x = 0;
        test.kernel(test.inputs, output.data() + 1, 1);
        if (!std::all_of(output.begin(), output.end(), [](Scalar v) { return v == sentinel; })) {
            std::fprintf(stderr, "thread index overflow in %s\n", test.name);
            return 1;
        }
    }
}
