#include <catch2/catch_test_macros.hpp>
#include <detail/dense/matrixops.hpp>

#include <cstdint>

using namespace symprop;

TEST_CASE("test gemmtn") {
  std::vector<double> A = {1, 2, 2, 3, 3, 4}; // 3x2 I = 3 R = 2
  std::vector<double> B = {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6}; // 3x4 I = 3 L = 4
  std::vector<double> C(2 * 4, 0); // 2x4 R = 2 L = 4

  matrix::gemmtn(C, A, B, 3, 2, 4);

  std::vector<double> ref = {14, 20, 26, 32, 20, 29, 38, 47};
  REQUIRE(C == ref);
}


