#include "linalg.hpp"

namespace tenkai {

Operation::Ptr Vector::sum() {
  Operation::Ptr result = Operation::make_zero();
  for (size_t i = 0; i < size(); i++) {
    result = result + elements[i];
  }
  return result;
};

Operation::Ptr Vector::sqnorm() {
  Operation::Ptr result = Operation::make_zero();
  for (size_t i = 0; i < size(); i++) {
    result = result + elements[i] * elements[i];
  }
  return result;
};

Vector Vector::operator+(const Vector& v) {
  std::vector<Operation::Ptr> elements(size());
  for (size_t i = 0; i < size(); i++) {
    elements[i] = (*this)(i) + v(i);
  }
  return Vector({elements});
}

Matrix Matrix::RotX(Operation::Ptr angle) {
  auto c = cos(angle);
  auto s = sin(angle);
  std::vector<Operation::Ptr> elements = {Operation::make_one(),
                                          Operation::make_zero(),
                                          Operation::make_zero(),
                                          Operation::make_zero(),
                                          c,
                                          negate(s),
                                          Operation::make_zero(),
                                          s,
                                          c};
  return Matrix(elements, 3, 3);
}

Matrix Matrix::RotY(Operation::Ptr angle) {
  auto c = cos(angle);
  auto s = sin(angle);
  std::vector<Operation::Ptr> elements = {c,
                                          Operation::make_zero(),
                                          s,
                                          Operation::make_zero(),
                                          Operation::make_one(),
                                          Operation::make_zero(),
                                          negate(s),
                                          Operation::make_zero(),
                                          c};
  return Matrix(elements, 3, 3);
}

Matrix Matrix::RotZ(Operation::Ptr angle) {
  auto c = cos(angle);
  auto s = sin(angle);
  std::vector<Operation::Ptr> elements = {c,
                                          negate(s),
                                          Operation::make_zero(),
                                          s,
                                          c,
                                          Operation::make_zero(),
                                          Operation::make_zero(),
                                          Operation::make_zero(),
                                          Operation::make_one()};
  return Matrix(elements, 3, 3);
}

Vector Matrix::operator*(const Vector& v) {
  std::vector<Operation::Ptr> elements(v.size());
  for (size_t i = 0; i < n_rows; i++) {
    Operation::Ptr sum = Operation::make_zero();
    for (size_t j = 0; j < n_cols; j++) {
      sum = sum + (*this)(i, j) * v(j);
    }
    elements[i] = sum;
  }
  return Vector(elements);
}

Matrix Matrix::operator*(const Matrix& other) {
  std::vector<Operation::Ptr> elements(n_rows * other.n_cols);
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < other.n_cols; j++) {
      Operation::Ptr sum = Operation::make_zero();
      for (size_t k = 0; k < n_cols; k++) {
        sum = sum + (*this)(i, k) * other(k, j);
      }
      elements[i * other.n_cols + j] = sum;
    }
  }
  return Matrix(elements, n_rows, other.n_cols);
}

}  // namespace tenkai
