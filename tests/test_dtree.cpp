#include <catch2/catch_test_macros.hpp>
#include <detail/dtree.hpp>

#include "utils.h"
#include <vector>

using namespace symprop;

TEST_CASE("toy d-tree") {

  std::vector<dim_t> inds{1, 2, 2, 3, 4};
  real_t val = 1.0;
  DTree dt;

  SECTION("constructing d-tree from single nonzero") {
    // dt.insert(inds, val);
    dt.banerjee_insert(std::span(inds), val);
    dt.print(std::cout);
  }

  SECTION("constructing d-tree from two nonzero") {
    std::vector<dim_t> inds{1, 2, 3, 4, 6};
    real_t val = 2.0;
    dt.banerjee_insert(inds, val);
    inds[4] = 5;
    val = 1.0;
    dt.banerjee_insert(inds, val);
    dt.print(std::cout);
  }
}

TEST_CASE("d-tree kronecker") {
  std::cout << "d-tree kronecker" << std::endl;
  // std::vector<size_t> inds{1, 2, 2, 3, 4};
  std::vector<dim_t> inds = {1, 2, 3, 4, 5};
  real_t val = 1.0;
  std::vector<real_t> U = {1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0};
  //   size_t n = 5;
  size_t R = 2;

  SECTION("Single element TTMc") {
    DTree dt;
    // dt.insert(inds, val);
    dt.banerjee_insert(inds, val);
    // dt.print(std::cout);
    dt.kronecker_product(U, R);

    std::vector<kpnode> leaves = dt.leaves();
    for (auto &node : leaves) {
      std::cout << "leaf: ";
      for (auto &i : node.subindex) {
        std::cout << i << " ";
      }
      std::cout << "\n";

      auto res_symtens = single_elem_s3ttmc(U, node.subindex, R);
      auto res_dtree = node.n->kptensor->to_tensor();
      REQUIRE(res_symtens == res_dtree);

      auto res_fulltens = single_elem_s3ttmc_naive(U, node.subindex, R);
      REQUIRE(res_fulltens == res_symtens);
    }
  }

  // SECTION("Single element TTMc with repeated indices") {
  //   DTree dt;
  //   inds = {1, 1, 1, 1, 1};

  //   dt.banerjee_insert(inds, val);
  //   dt.kronecker_product(U, R);

  //   // int coeff = multinomial_coeff(std::array<index_t, 4>({1, 2, 1, 1}));
  //   int coeff = 1;

  //   for (index_t i = 0; i < inds.size(); i++) {
  //     std::vector<dim_t> subindex(inds.size() - 1);

  //     for (index_t j = 0; j < inds.size(); j++) {
  //       if (j < i) {
  //         subindex[j] = inds[j];
  //       } else if (j > i) {
  //         subindex[j - 1] = inds[j];
  //       }
  //     }
  //     auto *node = dt.search(std::views::all(subindex));
  //   }

  // for (auto &node : nodes) {
  //   auto res_symtens = single_elem_s3ttmc(U, node->, R);
  //   for (size_t i = 0; i < res_symtens.size(); i++) {
  //     res_symtens[i] *= coeff;
  //   }

  //   auto res_dtree = node.n->kptensor->to_tensor();
  //   REQUIRE(res_symtens == res_dtree);

  //   auto res_fulltens = single_elem_s3ttmc_naive(U, node.subindex, R);
  //   for (size_t i = 0; i < res_fulltens.size(); i++) {
  //     res_fulltens[i] *= coeff;
  //   }

  //   REQUIRE(res_fulltens == res_symtens);
  // }
}
