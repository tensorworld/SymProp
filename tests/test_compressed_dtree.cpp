#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <detail/compressed_dtree.hpp>

#include "utils.h"
#include <vector>

using namespace symprop;

TEST_CASE("toy cd-tree") {

  std::vector<dim_t> inds{1, 2, 2, 3, 4};
  real_t val = 1.0;
  DTree dt;

  SECTION("constructing cd-tree from single nonzero") {
    // dt.insert(inds, val);
    dt.banerjee_insert(std::span(inds), val);
    dt.print(std::cout);
    CompressedDTree cdt(dt, inds.size() - 1);
    cdt.print();
  }

  SECTION("constructing cd-tree from two nonzero") {
    std::vector<dim_t> inds{1, 2, 3, 4, 6};
    real_t val = 2.0;
    dt.banerjee_insert(inds, val);
    inds[4] = 5;
    val = 1.0;
    dt.banerjee_insert(inds, val);
    dt.print(std::cout);
    CompressedDTree cdt(dt, inds.size() - 1);
    cdt.print();
  }
}

TEST_CASE("cdtree kronecker") {

  std::vector<dim_t> inds = {1, 2, 3, 4, 5};
  real_t val = 0.5;

  std::vector<real_t> U = {1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0};
  size_t R = 2;

  SECTION("no repeat index") {
    DTree dt;
    dt.banerjee_insert(inds, val);
    std::sort(inds.begin(), inds.end());
    CompressedDTree cdt(dt, inds.size() - 1);

    dt.kronecker_product(U, R);
    cdt.kronecker_product(U, R);

    for (size_t j = 0; j < inds.size(); j++) {
      auto *leaf = dt.exclude_search(inds, j);
      index_t cleaf = cdt.exclude_search(inds, j);

      auto tensor1 = std::span<const real_t>(*leaf->kptensor);
      auto tensor2 = std::span<const real_t>(*cdt.get_leaf_kptensors()[cleaf]);

      for (size_t i = 0; i < tensor1.size(); i++) {
        REQUIRE(tensor1[i] == Catch::Approx(tensor2[i]));
      }
    }
  }
}
