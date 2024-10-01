#include <gtest/gtest.h>
#include "cg.hpp"

TEST(BasicMathTest, ConstantFolding) {
  auto a = tenkai::Operation::make_constant(1.5);
  auto b = tenkai::Operation::make_constant(2.0);
  auto c = tenkai::Operation::make_constant(3.0);
  auto d = a * b + c;
  EXPECT_EQ(d->rhs, nullptr);
  EXPECT_EQ(d->lhs, nullptr);
  EXPECT_EQ(std::stod(d->name), 6.0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
