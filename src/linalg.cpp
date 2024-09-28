#include "linalg.hpp"

Vector3 Vector3::operator+(const Vector3& v) {
  std::array<Operation::Ptr, 3> elements;
  for (size_t i = 0; i < 3; i++) {
    elements[i] = this->get(i) + v.get(i);
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
  auto e0 = this->get(0, 0) * v.get(0) + this->get(0, 1) * v.get(1) +
            this->get(0, 2) * v.get(2);
  auto e1 = this->get(1, 0) * v.get(0) + this->get(1, 1) * v.get(1) +
            this->get(1, 2) * v.get(2);
  auto e2 = this->get(2, 0) * v.get(0) + this->get(2, 1) * v.get(1) +
            this->get(2, 2) * v.get(2);
  return Vector3({e0, e1, e2});
}

Matrix3 Matrix3::operator*(const Matrix3& m) {
  std::array<Operation::Ptr, 9> elements;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      elements[i * 3 + j] = this->get(i, 0) * m.get(0, j) +
                            this->get(i, 1) * m.get(1, j) +
                            this->get(i, 2) * m.get(2, j);
    }
  }
  return Matrix3({elements});
}
