#ifndef SYMPROP_SPARSE_MATRIXOPS_HPP
#define SYMPROP_SPARSE_MATRIXOPS_HPP

#include <utils/types.hpp>

#include <cstddef>
#include <vector>

namespace symprop {
namespace matrix {

template <typename T> struct COO {
  std::vector<T> row_indices;
  std::vector<T> col_indices;
  std::vector<real_t> values;
  T rows;
  T cols;
};

void sparse_leftsvd(matrix::COO<int> &A, std::vector<real_t> &U, size_t rank);

} // namespace matrix
} // namespace symprop

#endif // SYMPROP_SPARSE_MATRIXOPS_HPP