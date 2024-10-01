#pragma once

#include "linalg.hpp"

namespace tenkai {

struct SpatialTransform {
  Matrix rot;
  Vector trans;
  size_t dim;

  SpatialTransform(Matrix rotation, Vector translation)
      : rot(rotation), trans(translation), dim(rotation.n_rows) {
    if (rotation.n_rows != rotation.n_cols) {
      throw std::runtime_error("Matrix must be square");
    }
    if (rotation.n_rows != translation.size()) {
      throw std::runtime_error("Matrix and vector must have the same size");
    }
    if (dim != 2 && dim != 3) {
      throw std::runtime_error("Matrix must be 2x2 or 3x3");
    }
  }
  SpatialTransform operator*(const SpatialTransform& other) {
    return SpatialTransform(rot * other.rot, trans + rot * other.trans);
  }
  Vector operator*(const Vector& v) { return rot * v + trans; }
  SpatialTransform inverse() {
    return SpatialTransform(rot.transpose(), rot.transpose() * (-trans));
  }
};

};  // namespace tenkai
