#pragma once
#include <array>
#include "cg.hpp"

struct Vector3 {
  std::array<Operation::Ptr, 3> elements;
  Operation::Ptr sum() { return elements[0] + elements[1] + elements[2]; };
  Operation::Ptr get(size_t i) const { return elements[i]; };
  Vector3 operator+(const Vector3& v);
};

struct Matrix3 {
  static Matrix3 identity();
  static Matrix3 RotX(Operation::Ptr angle);
  static Matrix3 RotY(Operation::Ptr angle);
  static Matrix3 RotZ(Operation::Ptr angle);
  Vector3 operator*(const Vector3& v);
  Matrix3 operator*(const Matrix3& m);
  Operation::Ptr get(size_t i, size_t j) const { return elements[i * 3 + j]; };
  std::array<Operation::Ptr, 9> elements;
};
