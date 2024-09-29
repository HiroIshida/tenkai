#include "linalg.hpp"

namespace tenkai {

Vector3 Vector3::operator+(const Vector3& v) {
  std::array<Operation::Ptr, 3> elements;
  for (size_t i = 0; i < 3; i++) {
    elements[i] = (*this)(i) + v(i);
  }
  return Vector3({elements});
}

Matrix3 Matrix3::identity() {
  std::array<Operation::Ptr, 9> elements = {
      Operation::make_one(),  Operation::make_zero(), Operation::make_zero(),
      Operation::make_zero(), Operation::make_one(),  Operation::make_zero(),
      Operation::make_zero(), Operation::make_zero(), Operation::make_one()};
  return Matrix3({elements});
}

Matrix3 Matrix3::RotX(Operation::Ptr angle) {
  auto c = cos(angle);
  auto s = sin(angle);
  std::array<Operation::Ptr, 9> elements = {Operation::make_one(),
                                            Operation::make_zero(),
                                            Operation::make_zero(),
                                            Operation::make_zero(),
                                            c,
                                            negate(s),
                                            Operation::make_zero(),
                                            s,
                                            c};
  return Matrix3({elements});
}

Matrix3 Matrix3::RotY(Operation::Ptr angle) {
  auto c = cos(angle);
  auto s = sin(angle);
  std::array<Operation::Ptr, 9> elements = {c,
                                            Operation::make_zero(),
                                            s,
                                            Operation::make_zero(),
                                            Operation::make_one(),
                                            Operation::make_zero(),
                                            negate(s),
                                            Operation::make_zero(),
                                            c};
  return Matrix3({elements});
}

Matrix3 Matrix3::RotZ(Operation::Ptr angle) {
  auto c = cos(angle);
  auto s = sin(angle);
  std::array<Operation::Ptr, 9> elements = {c,
                                            negate(s),
                                            Operation::make_zero(),
                                            s,
                                            c,
                                            Operation::make_zero(),
                                            Operation::make_zero(),
                                            Operation::make_zero(),
                                            Operation::make_one()};
  return Matrix3({elements});
}

Vector3 Matrix3::operator*(const Vector3& v) {
  auto e0 = (*this)(0, 0) * v(0) + (*this)(0, 1) * v(1) + (*this)(0, 2) * v(2);
  auto e1 = (*this)(1, 0) * v(0) + (*this)(1, 1) * v(1) + (*this)(1, 2) * v(2);
  auto e2 = (*this)(2, 0) * v(0) + (*this)(2, 1) * v(1) + (*this)(2, 2) * v(2);
  return Vector3({e0, e1, e2});
}

Matrix3 Matrix3::operator*(const Matrix3& m) {
  std::array<Operation::Ptr, 9> elements;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      elements[i * 3 + j] = (*this)(i, 0) * m(0, j) + (*this)(i, 1) * m(1, j) +
                            (*this)(i, 2) * m(2, j);
    }
  }
  return Matrix3({elements});
}

}  // namespace tenkai
