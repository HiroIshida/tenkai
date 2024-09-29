#pragma once
#include <array>
#include "cg.hpp"

namespace tenkai {

// because these struct are used for code generation, we dont need to support
// neither static sizes nor static asserts

struct Vector {
  // variables
  std::vector<Operation::Ptr> elements;
  inline size_t size() const { return elements.size(); }

  // methods
  Vector(std::vector<Operation::Ptr> elements) : elements(elements) {}
  inline Operation::Ptr operator()(size_t i) const { return elements[i]; }
  Operation::Ptr sum();
  Operation::Ptr sqnorm();
  Vector operator+(const Vector& v);
  Vector operator*(Operation::Ptr scalar);
};

struct Matrix {
  // variables
  std::vector<Operation::Ptr> elements;
  size_t n_rows;
  size_t n_cols;

  // methods
  Matrix(std::vector<Operation::Ptr> elements, size_t n_rows, size_t n_cols)
      : elements(elements), n_rows(n_rows), n_cols(n_cols) {}
  static Matrix RotX(Operation::Ptr angle);
  static Matrix RotY(Operation::Ptr angle);
  static Matrix RotZ(Operation::Ptr angle);
  static Matrix Var(size_t n_rows, size_t n_cols);
  Vector operator*(const Vector& v);
  Matrix operator*(const Matrix& other);
  Matrix operator*(Operation::Ptr scalar);
  Matrix transpose();
  inline Operation::Ptr operator()(size_t i, size_t j) const {
    return elements[i + j * n_rows];
  };
};

}  // namespace tenkai
