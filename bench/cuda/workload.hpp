#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "config.h"

inline size_t repetitions_for(size_t batch) {
    return std::max(size_t(2), std::min(kMaxRepetitions,
        (kQueriesPerRound + batch - 1) / batch));
}

inline size_t input_offset(size_t batch, size_t round, size_t iteration) {
    return ((round * 104729 + iteration) % (kPool / batch)) * batch;
}

inline size_t job_index(size_t round, size_t index, size_t count) {
    return (index + round * 5) % count;
}

template <typename T>
void read_file(const std::string& path, T* data, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != std::streamoff(count * sizeof(T)))
        throw std::runtime_error("unexpected size or missing file: " + path);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(data), count * sizeof(T)))
        throw std::runtime_error("cannot read " + path);
}
