#include "detail/sparse/spsymtensor.hpp"
#include "utils/types.hpp"
#include <detail/dense/matrixops.hpp>
#include <detail/sparse/matrixops.hpp>
#include <detail/tucker.hpp>
#include <ios>
#include <random>
#include <utils/function_timer.hpp>

#include <cassert>
#include <cstddef>
#include <iomanip>

namespace symprop {

SpSymtensor::SpSymtensor(dim_t dim, order_t order,
                         const std::vector<dim_t> &unnz_inds,
                         const std::vector<real_t> &unnzs)
    : _dim(dim), _order(order), _unnz(unnzs.size()), _norm(0),
      _dtree(new DTree()), _unnz_inds(unnz_inds), _unnzs(unnzs) {
  for (size_t i = 0; i < unnzs.size(); i++) {
    _dtree->banerjee_insert(_get_unnz_inds(i), unnzs[i]);
  }
  _norm = calc_norm();
  _cdtree = std::make_unique<CompressedDTree>(*_dtree, _order - 1);
  std::cout << "Finished creating compressed dtree\n";
  _dtree.reset(nullptr);
}

void SpSymtensor::S3TTMc(std::vector<real_t> &res, const std::vector<real_t> &U,
                         dim_t R, bool partial_sym_res) const {
  START_FUNCTION_TIMER();

  size_t sym_ncols = symtens_size(R, _order - 1);
  size_t res_ncols = partial_sym_res ? sym_ncols : std::pow(R, _order - 1);

  assert(res.size() == _dim * res_ncols);
  if (_dtree) {
    #pragma omp parallel for
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = 0.0;
    }
    _dtree->kronecker_product(U, R);
    std::vector<index_t> freqs(_order, 0);

    for (size_t i = 0; i < _unnz; i++) {
      auto unnzi = _get_unnz_inds(i);
      count_freq(unnzi, std::span(freqs));
      real_t coeff =
          static_cast<real_t>(factorial[_order]) / multinomial_coeff(freqs);

      for (size_t j = 0; j < _order; j++) {
        auto *leaf = _dtree->exclude_search(unnzi, j);
        assert(leaf);
        auto leaf_view = std::span<const real_t>(*leaf->kptensor);
        for (size_t k = 0; k < leaf_view.size(); k++) {
          res[(unnzi[j] - 1) * res_ncols + k] +=
              leaf_view[k] * _unnzs[i] / coeff;
        }
      }
    }
  }

  //   if (!_dtree && _cdtree) {
  //     _cdtree->kronecker_product_opt4(U, R);
  //     std::vector<index_t> freqs(_order, 0);
  // #pragma omp parallel for firstprivate(freqs)
  //     for (size_t i = 0; i < _unnz; i++) {
  //       auto unnzi = _get_unnz_inds(i);
  //       count_freq(unnzi, std::span(freqs));
  //       real_t coeff =
  //           static_cast<real_t>(factorial[_order]) /
  //           multinomial_coeff(freqs);

  //       for (size_t j = 0; j < _order; j++) {
  //         auto cleaf = _cdtree->exclude_search(unnzi, j);
  //         auto leaf_view =
  //             _cdtree->get_leaf_kptensors_array()->symtensor_data(cleaf);
  //         // auto leaf_view = std::span<const
  //         // real_t>(*_cdtree->get_leaf_kptensors()[cleaf]);

  //         for (size_t k = 0; k < leaf_view.size(); k++) {
  //           real_t val = leaf_view[k] * _unnzs[i] / coeff;
  // #pragma omp atomic
  //           res[(unnzi[j] - 1) * res_ncols + k] += val;
  //         }
  //       }
  //     }
  //   }

  if (!_dtree && _cdtree) {
#pragma omp parallel for
    for (index_t i = 0; i < res.size(); i++) {
      res[i] = 0.0;
    }

    _cdtree->kronecker_product_opt5(U, R, res, res_ncols);
  }

  if (!partial_sym_res) {
    std::vector<real_t> row(sym_ncols);
#pragma omp parallel for firstprivate(row)
    for (size_t i = 0; i < _dim; i++) {
      auto res_row = std::span(res).subspan(i * res_ncols, res_ncols);
      std::copy(res_row.begin(), res_row.begin() + sym_ncols, row.begin());

      unpack_symtens(std::span(row), R, _order - 1, res_row);
    }
  }
}

real_t SpSymtensor::norm() const { return this->_norm; }

real_t SpSymtensor::calc_norm() const {
  real_t norm = 0.0;
  std::vector<index_t> freq(_order, 0);
  for (size_t i = 0; i < _unnz; i++) {
    std::fill(freq.begin(), freq.end(), 0);
    auto nnz_idx = std::span(_unnz_inds).subspan(i * _order, _order);
    freq[0] = 1;
    dim_t prev = nnz_idx[0];
    dim_t prev_idx = 0;

    for (order_t j = 1; j < _order; j++) {
      if (nnz_idx[j] == prev) {
        freq[prev_idx] += 1;
      } else {
        prev = nnz_idx[j];
        prev_idx = j;
        freq[j] = 1;
      }
    }

    real_t coeff = multinomial_coeff(freq);
    norm += std::pow(_unnzs[i], 2) * coeff;
  }
  return std::sqrt(norm);
}

real_t residual(const std::vector<real_t> &x, const std::vector<real_t> &y) {
  real_t residual = 0.0f;
#pragma omp parallel for reduction(+ : residual)
  for (size_t i = 0; i < x.size(); i++)
    residual += std::pow((x[i] - y[i]), 2);
  return residual;
}

real_t rel_residual(const std::vector<real_t> &x,
                    const std::vector<real_t> &y) {
  real_t residual = 0.0f;
  real_t norm = 0.0f;
#pragma omp parallel for reduction(+ : residual, norm)
  for (size_t i = 0; i < x.size(); i++) {
    residual += std::pow((x[i] - y[i]), 2);
    norm += std::pow(x[i], 2);
  }
  return residual / norm;
}

real_t norm(std::vector<real_t> &x) {
  real_t norm = 0.0f;
#pragma omp parallel for reduction(+ : norm)
  for (size_t i = 0; i < x.size(); i++)
    norm += std::pow(x[i], 2);
  return std::sqrt(norm);
}

void regularize(std::vector<real_t> &U, size_t nrows, size_t ncols,
                real_t delta) {
  std::vector<real_t> rownorms(nrows, 0);
  real_t multiplier;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      rownorms[i] += U[i * ncols + j] * U[i * ncols + j];
    }
    rownorms[i] = sqrt(rownorms[i]);
    // std::cout << "Row norm " << i << " " << rownorms[i] << "\n";
    if (delta < rownorms[i]) {
      multiplier = delta / rownorms[i];
      for (size_t j = 0; j < ncols; j++) {
        U[i * ncols + j] *= multiplier;
      }
    }
  }
  std::vector<real_t> U_copy = U;
  // std::cout << std::setprecision(4) << std::fixed;
  // for (size_t i = 0; i < ncols; i++) {
  //   for (size_t j = 0; j < nrows; j++) {
  //     std::cout << U_copy[j * ncols + i] << ",";
  //   }
  // }
  // std::cout << "\n";
  matrix::leftsvd(U_copy, U, nrows, ncols, ncols);
  // for (size_t i = 0; i < ncols; i++) {
  //   for (size_t j = 0; j < nrows; j++) {
  //     std::cout << U[j * ncols + i] << ",";
  //   }
  // }
}

SymTuckerDecomp SpSymtensor::TuckerHOOI(const Config &c) const {
  START_FUNCTION_TIMER();
  SymTuckerDecomp decomp(c, _dim, _order);
  auto &core = decomp._core;
  auto &factor = decomp._factor;
  auto core_prev = core;

  hosvd_init(factor, c);
  // std::cout << std::fixed << std::setprecision(6);
  // for (size_t i = 0; i < _dim; i++) {
  //   for (size_t j = 0; j < c.rank; j++) {
  //     std::cout << factor[i * c.rank + j] << ",";
  //   }
  //   std::cout << "\n";
  // }

  size_t ncols = std::pow(c.rank, _order - 1);
  std::vector<real_t> Y(_dim * ncols);

  real_t prev_error = 0.0;
  for (size_t iter = 0; iter < c.max_iter; iter++) {
    regularize(factor, _dim, c.rank, c.delta);
    S3TTMc(Y, factor, c.rank);
    std::vector<real_t> Y_copy = Y;
    matrix::leftsvd(Y, factor, _dim, ncols, c.rank);
    matrix::gemmtn(core, factor, Y_copy, _dim, c.rank, ncols);

    // real_t res = residual(core_prev, core) / _norm;
    real_t rel_res = rel_residual(core_prev, core);
    real_t error = decomp.reconstruct_error(*this);
    std::cout << std::scientific << std::setprecision(12)
              << "Iteration: " << iter << " relative residual: " << rel_res
              << " error " << error << "\n";

    real_t error_diff = std::abs(prev_error - error);
    if (error_diff < c.rel_tol) {
      break;
    }
    prev_error = error;
    core_prev = core;
  }

  // std::cout << "Final core:\n";
  // for (size_t i = 0; i < ncols; i++) {
  //   for (size_t j = 0; j < rank; j++) {
  //     std::cout << core[i * rank + j] << ",";
  //   }
  //   std::cout << "\n";
  // }

  return decomp;
}

SymTuckerDecomp SpSymtensor::TuckerHOQRI(const Config &c) const {
  START_FUNCTION_TIMER();
  SymTuckerDecomp decomp(c, _dim, _order);
  auto &factor = decomp._factor;
  hosvd_init(factor, c);

  size_t ncols = symtens_size(c.rank, _order - 1);
  std::vector<real_t> core(c.rank * ncols);
  std::vector<real_t> core_prev(c.rank * ncols);
  std::vector<real_t> Y(_dim * ncols);
  std::vector<real_t> coeff = symtens_coeff(c.rank, _order - 1);
  real_t prev_error = 0.0;

  for (size_t iter = 0; iter < c.max_iter; iter++) {
    regularize(factor, _dim, c.rank, c.delta);
    S3TTMc(Y, factor, c.rank, true);
    matrix::gemmtn(core, factor, Y, _dim, c.rank, ncols);
    matrix::symgemm(factor, Y, core, coeff, _dim, c.rank);
    matrix::qr(factor, _dim, c.rank);
    real_t rel_res = symtens_rel_res(std::span(core), std::span(core_prev),
                                     std::span(coeff), c.rank);
    real_t core_norm = symtens_norm(std::span(core), std::span(coeff), c.rank);
    real_t error = std::sqrt(_norm * _norm - core_norm * core_norm) / _norm;

    std::cout << std::scientific << std::setprecision(12)
              << "Iteration: " << iter << " relative residual: " << rel_res
              << " error " << error << "\n";
    core_prev = core;

    real_t error_diff = std::abs(prev_error - error);
    if (error_diff < c.rel_tol) {
      break;
    }
    prev_error = error;
  }

  size_t unpacked_size = std::pow(c.rank, _order - 1);
  for (size_t i = 0; i < c.rank; i++) {
    auto core_view = std::span(core).subspan(i * ncols, ncols);
    auto decomp_core_view =
        std::span(decomp._core).subspan(i * unpacked_size, unpacked_size);
    unpack_symtens(core_view, c.rank, _order - 1, decomp_core_view);
  }

  return decomp;
}

SymTuckerDecomp SpSymtensor::TuckerAltHOOI(const Config &c) const {
  START_FUNCTION_TIMER();
  SymTuckerDecomp decomp(c, _dim, _order);
  auto &core = decomp._core;
  auto &factor = decomp._factor;
  auto factor_prev = factor;
  std::vector<real_t> coeff = symtens_coeff(c.rank, _order - 1);
  hosvd_init(factor, c);

  size_t ncols = symtens_size(c.rank, _order - 1);
  std::vector<real_t> Y(_dim * ncols);
  std::vector<real_t> A(_dim * _dim);
  real_t prev_error = 0.0;

  for (size_t iter = 0; iter < c.max_iter; iter++) {
    regularize(factor, _dim, c.rank, c.delta);
    S3TTMc(Y, factor, c.rank, true);
    matrix::symsyrk(A, Y, coeff, _dim);
    matrix::syevd(A, factor, _dim, c.rank);
    real_t rel_res = rel_residual(factor, factor_prev);
    matrix::gemmtn(core, factor, Y, _dim, c.rank, ncols);
    real_t core_norm = symtens_norm(std::span(core), std::span(coeff), c.rank);
    real_t error = std::sqrt(_norm * _norm - core_norm * core_norm) / _norm;
    std::cout << std::scientific << std::setprecision(10)
              << "Iteration: " << iter << " relative residual: " << rel_res
              << " error " << error << "\n";

    real_t error_diff = std::abs(prev_error - error);
    if (error_diff < c.rel_tol) {
      break;
    }
    prev_error = error;
    factor_prev = factor;
  }

  return decomp;
}

void SpSymtensor::hosvd_init(std::vector<real_t> &U, const Config &c) const {
  START_FUNCTION_TIMER();

  if (c.init == "hosvd") {
    auto coo = convert_to_coo();
    matrix::COO<int> coo_int;
    coo_int.col_indices.reserve(coo.entries.size());
    coo_int.row_indices.reserve(coo.entries.size());
    coo_int.values.reserve(coo.entries.size());
    for (const auto &entry : coo.entries) {
      coo_int.col_indices.push_back(std::get<0>(entry));
      coo_int.row_indices.push_back(std::get<1>(entry));
      coo_int.values.push_back(std::get<2>(entry));
    }

    coo_int.rows = _dim;
    coo_int.cols = 1;
    for (size_t i = 1; i < _order; i++) {
      coo_int.cols *= _dim;
    }

    matrix::sparse_leftsvd(coo_int, U, c.rank);
  }

  if (c.init == "hoevd") {
    auto coo = convert_to_coo();
    std::vector<real_t> A(_dim * _dim, 0.0);
    // std::normal_distribution<> distr(0.0, 0.1);
    // std::mt19937 engine{c.seed};
    // for (size_t i = 0; i < _dim; i++) {
    //   for (size_t j = 0; j < _dim; j++) {
    //     A[i * _dim + j] = distr(engine);
    //   }
    // }
    sparse_syrk(coo, A);
    for (size_t i = 0; i < _dim; i++) {
      for (size_t j = 0; j < _dim; j++) {
        std::cout << A[i * _dim + j] << ",";
      }
      std::cout << "\n";
    }

    matrix::syevd(A, U, _dim, c.rank);
  }
}

void SpSymtensor::sparse_syrk(const COO &A, std::vector<real_t> &result) const {
  START_FUNCTION_TIMER();
  index_t i = 0;
  while (i < A.entries.size()) {
    index_t col = std::get<0>(A.entries[i]);
    index_t j = i + 1;
    while (j < A.entries.size() && std::get<0>(A.entries[j]) == col)
      j++;

    for (index_t k = i; k < j; k++) {
      for (index_t l = k; l < j; l++) {
        index_t row1 = std::get<1>(A.entries[k]);
        index_t row2 = std::get<1>(A.entries[l]);
        double val1 = std::get<2>(A.entries[k]);
        double val2 = std::get<2>(A.entries[l]);
        result[row1 * _dim + row2] += val1 * val2;
        if (row1 != row2) {
          result[row2 * _dim + row1] += val1 * val2;
        }
      }
    }

    i = j;
  }
}

COO SpSymtensor::convert_to_coo() const {
  std::vector<std::tuple<index_t, index_t, double>> matrix_coo;
  matrix_coo.reserve(_unnz * factorial[_order]);
  std::vector<index_t> perm(_order);
  for (size_t i = 0; i < _unnz_inds.size(); i += _order) {
    for (index_t j = 0; j < _order; ++j) {
      perm[j] = _unnz_inds[i + j] - 1;
    }

    do {
      index_t row = perm[0];
      index_t col = 0;
      for (index_t j = 1; j < _order; ++j) {
        col = col * _dim + perm[j];
      }
      matrix_coo.emplace_back(col, row, _unnzs[i / _order]);
    } while (std::next_permutation(perm.begin(), perm.end()));
  }

  // Sort the COO entries by column, then by row
  std::sort(matrix_coo.begin(), matrix_coo.end(),
            [](const auto &a, const auto &b) {
              return std::make_pair(std::get<0>(a), std::get<1>(a)) <
                     std::make_pair(std::get<0>(b), std::get<1>(b));
            });
  COO coo;
  coo.entries = std::move(matrix_coo);
  return coo;
}

}; // namespace symprop
