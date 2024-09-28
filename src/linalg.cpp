#include "linalg.hpp"

Matrix3 Matrix3::identity() {
  std::array<Operation::Ptr, 9> elements = {
      Operation::make_one(),  Operation::make_zero(), Operation::make_zero(),
      Operation::make_zero(), Operation::make_one(),  Operation::make_zero(),
      Operation::make_zero(), Operation::make_zero(), Operation::make_one()};
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
