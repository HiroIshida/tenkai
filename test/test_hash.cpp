#include "cg.hpp"
#include <iostream>
#include <gtest/gtest.h>

TEST(HashTest, BasicMathTest) {
  auto a = tenkai::Operation::make_var();
  auto b = tenkai::Operation::make_var();
  auto c = tenkai::Operation::make_var();
  auto d = tenkai::Operation::make_var();
  { // test commutative property
    auto expr1 = a + b + c + d;
    auto expr2 = d + c + b + a;
    EXPECT_EQ(expr1->hash_id, expr2->hash_id);
  }

  { // test associative property
    auto expr1 = (a + b) + (c + d);
    auto expr2 = a + (b + c) + d;
    EXPECT_EQ(expr1->hash_id, expr2->hash_id);
  }

  { // test distributive property
    auto expr1 = (a + b) * c;
    auto expr2 = a * c + b * c;
    EXPECT_EQ(expr1->hash_id, expr2->hash_id);
  }
}

TEST(HashTest, MathFuncTest) {
  auto a = tenkai::Operation::make_var();
  auto b = tenkai::Operation::make_var();
  auto c = sin(a) + cos(a);
  auto d = cos(a) + sin(a);
  EXPECT_EQ(c->hash_id, d->hash_id);

  // nested case
  auto e = sin(sin((a + b) * c) + cos(a - b));
  auto f = sin(cos(a - b) + sin(c * (a + b)));
  EXPECT_EQ(e->hash_id, f->hash_id);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
