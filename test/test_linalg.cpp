#include <gtest/gtest.h>
#include "cg.hpp"
#include "linalg.hpp"

TEST(LinalgTest, Vector) {
  auto var1 = tenkai::Operation::make_var();
  auto var2 = tenkai::Operation::make_var();
  auto var3 = tenkai::Operation::make_var();
  tenkai::Vector v1({var1, var2, var3});
  ASSERT_EQ(v1.size(), 3);

  { // addition
    auto v2 = v1 + v1;
    auto fn = tenkai::jit_compile<double>({var1, var2, var3}, v2.elements);
    double input[3] = {1, 2, 3};
    double output[3];
    fn(input, output, nullptr);
    ASSERT_EQ(output[0], 2);
    ASSERT_EQ(output[1], 4);
    ASSERT_EQ(output[2], 6);
  }

  { // sum
    auto fn = tenkai::jit_compile<double>({var1, var2, var3}, {v1.sum()});
    double input[3] = {1, 2, 3};
    double output;
    fn(input, &output, nullptr);
    ASSERT_EQ(output, 6);
  }

  { // sqnorm
    auto fn = tenkai::jit_compile<double>({var1, var2, var3}, {v1.sqnorm()});
    double input[3] = {1, 2, 3};
    double output;
    fn(input, &output, nullptr);
    ASSERT_EQ(output, 14.);
  }
}

TEST(LinalgTest, Matrix) {
  auto var1 = tenkai::Operation::make_var();
  auto var2 = tenkai::Operation::make_var();
  auto var3 = tenkai::Operation::make_var();
  auto var4 = tenkai::Operation::make_var();

  { // index access (column major)
    auto mat = tenkai::Matrix({var1, var2, var3, var4}, 2, 2);
    auto mat2 = mat * tenkai::Operation::make_constant(1.0);
    auto fn = tenkai::jit_compile<double>({var1, var2, var3, var4}, {mat2(0, 0), mat2(0, 1), mat2(1, 0), mat2(1, 1)});
    double input[4] = {1, 2, 3, 4};
    double output[4];
    fn(input, output, nullptr);
    ASSERT_EQ(output[0], 1);
    ASSERT_EQ(output[1], 3);
    ASSERT_EQ(output[2], 2);
    ASSERT_EQ(output[3], 4);
  }

  { // transpose
    auto mat = tenkai::Matrix({var1, var2, var3, var1, var2, var3}, 2, 3);
    auto mat2 = mat.transpose() * tenkai::Operation::make_constant(1.0);
    ASSERT_EQ(mat2.n_rows, 3);
    ASSERT_EQ(mat2.n_cols, 2);
    auto fn = tenkai::jit_compile<double>({var1, var2, var3}, {mat2(0, 0), mat2(0, 1), mat2(1, 0), mat2(1, 1), mat2(2, 0), mat2(2, 1)});
    double input[3] = {1, 2, 3};
    double output[6];
    fn(input, output, nullptr);
    ASSERT_EQ(output[0], 1);
    ASSERT_EQ(output[1], 2);
    ASSERT_EQ(output[2], 3);
    ASSERT_EQ(output[3], 1);
    ASSERT_EQ(output[4], 2);
    ASSERT_EQ(output[5], 3);
  }

  {
    auto mat1 = tenkai::Matrix::Var(2, 3);
    auto mat2 = tenkai::Matrix::Var(3, 2);
    auto mat3 = mat1 * mat2;

    std::vector<tenkai::Operation::Ptr> inputs;
    inputs.insert(inputs.end(), mat1.elements.begin(), mat1.elements.end());
    inputs.insert(inputs.end(), mat2.elements.begin(), mat2.elements.end());
    auto fn = tenkai::jit_compile<double>(inputs, {mat3(0, 0), mat3(0, 1), mat3(1, 0), mat3(1, 1)});
    double input[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    double output[4];
    fn(input, output, nullptr);
    ASSERT_EQ(output[0], 7 + 3 * 8 + 5 * 9);
    ASSERT_EQ(output[1], 10 + 3 * 11 + 5 * 12);
    ASSERT_EQ(output[2], 7 * 2 + 8 * 4 + 9 * 6);
    ASSERT_EQ(output[3], 10 * 2 + 11 * 4 + 12 * 6);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
