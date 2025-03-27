#include <cstring>
#include <detail/dtree.hpp>
#include <utils/types.hpp>

#include "utils.h"

namespace symprop {

std::vector<real_t> single_elem_s3ttmc(const std::vector<real_t> &u_mat,
                                       const std::vector<dim_t> &indices,
                                       size_t dim) {

  std::vector<std::unique_ptr<SymtensorBase>> level_tensors(indices.size());
  auto Urow = [&](size_t i) {
    return std::span(u_mat).subspan((i - 1) * dim, dim);
  };

  for (size_t i = 0; i < indices.size(); i++) {
    level_tensors[i] = create_symtensor(dim, i + 1);
  }

  auto indices_copy = indices;

  do {
    dynamic_cast<Symtensor<1> *>(level_tensors[0].get())
        ->set(Urow(indices_copy[0]));

    for (size_t i = 1; i < indices.size() - 1; i++) {
      level_tensors[i]->set_zero();
      level_tensors[i]->sym_outer_prod(*level_tensors[i - 1],
                                       Urow(indices_copy[i]));
    }

    level_tensors.back()->sym_outer_prod(*level_tensors[indices.size() - 2],
                                         Urow(indices_copy[indices.size() - 1]));

  } while (std::next_permutation(indices_copy.begin(), indices_copy.end()));

  return level_tensors.back()->to_tensor();
}

void kronecker(real_t *res, const real_t *src, const real_t *row, size_t dim,
               size_t order) {
  size_t dim1 = dim;
  size_t dim2 = 1;
  for (size_t i = 0; i < order; i++) {
    dim2 *= dim;
  }

  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      res[i * dim2 + j] += src[j] * row[i];
    }
  }
}

std::vector<real_t> single_elem_s3ttmc_naive(const std::vector<real_t> &u_mat,
                                             const std::vector<dim_t> &indices,
                                             size_t dim) {
  size_t N = indices.size();
  std::vector<real_t *> level_ptrs(N);
  std::vector<size_t> level_sizes(N);

  auto tensor_size = [](size_t dim, size_t order) {
    return std::pow(dim, order);
  };

  std::vector<real_t> last_level(tensor_size(dim, N));
  for (size_t i = 1; i < N; i++) {
    size_t sym_size = tensor_size(dim, i);
    level_ptrs[i - 1] = new real_t[sym_size];
    level_sizes[i - 1] = sym_size;
  }

  auto indices_copy = indices;

  do {
    for (size_t i = 0; i < dim; i++) {
      level_ptrs[0][i] = u_mat[(indices_copy[0] - 1) * dim + i];
    }

    for (size_t i = 1; i < N - 1; i++) {
      std::memset(level_ptrs[i], 0, level_sizes[i] * sizeof(real_t));
      kronecker(level_ptrs[i], level_ptrs[i - 1],
                u_mat.data() + (indices_copy[i] - 1) * dim, dim, i);
    }

    kronecker(last_level.data(), level_ptrs[N - 2],
              u_mat.data() + (indices_copy[N - 1] - 1) * dim, dim, N - 1);
  } while (std::next_permutation(indices_copy.begin(), indices_copy.end()));

  for (size_t i = 0; i < N - 1; i++) {
    delete[] level_ptrs[i];
  }

  return last_level;
}

// template <size_t N>
// real_t *single_elem_s3ttmc_opt(real_t *u_mat, const std::array<size_t, N>
// &indices, size_t dim) {
//     std::vector<real_t *> level_ptrs(N);
//     std::array<size_t, N> level_sizes;

//     for (size_t i = 1; i < N + 1; i++) {
//         size_t sym_size = symtens_size(dim, i);
//         level_ptrs[i - 1] = new real_t[sym_size];
//         level_sizes[i - 1] = sym_size;
//     }

//     std::array<size_t, N> indices_copy = indices;

//     do {
//         for (int i = 0; i < dim; i++) {
//             level_ptrs[0][i] = u_mat[indices_copy[0] * dim + i];
//         }

//         for (size_t i = 1; i < N - 1; i++) {
//             memset(level_ptrs[i], 0, level_sizes[i] * sizeof(real_t));
//             symtens_prod(level_ptrs[i], level_ptrs[i - 1], u_mat +
//             indices_copy[i] * dim, dim, i);
//         }

//         symtens_prod(level_ptrs[N - 1], level_ptrs[N - 2], u_mat +
//         indices_copy[N - 1] * dim, dim, N - 1);
//     } while (std::next_permutation(indices_copy.begin(),
//     indices_copy.end()));

//     for (size_t i = 0; i < N - 1; i++) {
//         delete[] level_ptrs[i];
//     }

//     return level_ptrs[N - 1];
// }

}; // namespace symprop