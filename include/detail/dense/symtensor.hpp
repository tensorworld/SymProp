#ifndef SYMPROP_SYMTENSOR_HPP
#define SYMPROP_SYMTENSOR_HPP

#include <detail/dense/symtensor_base.hpp>
#include <utils/function_timer.hpp>
#include <utils/types.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <span>
#include <vector>

namespace symprop {

template <order_t N> class SymtensorOps {

public:
  constexpr static dim_t VEC_DIM = 8;
  __attribute__((optimize("no-tree-vectorize"))) static void
  SymOuterProd(std::span<real_t> dst, std::span<const real_t> src,
               std::span<const real_t> row)
    requires(N > 1)
  {
    dim_t _dim = row.size();

    index_t res_idx = 0;
    SymtensorOps<N - 1>::Iterate(
        _dim, [&](const std::array<dim_t, N - 1> &indices, index_t oned_idx) {
          for (dim_t last = indices[N - 2]; last < _dim; last++) {
            dst[res_idx] += src[oned_idx] * row[last];
            res_idx++;
          }
        });
  }

  static void SymOuterProdVec(std::span<real_t> dst,
                              std::span<const real_t> src,
                              std::span<const real_t> row)
    requires(N > 1)
  {
    dim_t _dim = row.size();

    index_t res_idx = 0;
    SymtensorOps<N - 1>::Iterate(
        _dim, [&](const std::array<dim_t, N - 1> &indices, index_t oned_idx) {
          for (dim_t last = indices[N - 2]; last < _dim; last++) {
            dst[res_idx] += src[oned_idx] * row[last];
            res_idx++;
          }
        });
  }

  template <typename Func> static void Iterate(dim_t dim, Func &&f) {
    std::array<dim_t, N> indices{};
    index_t oned_idx = 0;
    iterate_impl<Func, N>(dim, std::forward<Func>(f), indices, oned_idx);
  }

  static void MemToIndex(index_t loc, dim_t _dim,
                         std::array<dim_t, N> &indices) {
    index_t remaining = loc;
    index_t start = 0;
    for (order_t j = 0; j < N; ++j) {
      dim_t i = start;
      while (_dim >= i) {
        index_t lvl_size = symtens_size(_dim - i, N - 1 - j);
        if (remaining < lvl_size) {
          break;
        }
        remaining -= lvl_size;
        i++;
      }
      indices[j] = i;
      start = i;
    }
  }

  static index_t At(const std::array<dim_t, N> &indices, dim_t _dim) {
    index_t oned_idx = 0;
    dim_t start = 0;

    for (order_t j = 0; j < N; ++j) {
      for (dim_t i = start; i < indices[j]; ++i) {
        oned_idx += symtens_size(_dim - i, N - 1 - j);
      }
      start = indices[j];
    }
    return oned_idx;
  }

  static void ToTensor(const std::span<const real_t> data, dim_t dim,
                       std::span<real_t> tensor) {
    assert(data.size() == symtens_size(dim, N));
    assert(tensor.size() == std::pow(dim, N));

    Iterate(dim, [&](const std::array<dim_t, N> &indices, index_t oned_idx) {
      std::array<dim_t, N> tensor_indices = indices;

      do {
        index_t tensor_oned_idx = 0;

        for (order_t j = 0; j < N; ++j) {
          tensor_oned_idx *= dim;
          tensor_oned_idx += tensor_indices[j];
        }

        tensor[tensor_oned_idx] = data[oned_idx];
      } while (
          std::next_permutation(tensor_indices.begin(), tensor_indices.end()));
    });
  }

  static void SymtensCoeff(std::vector<real_t> &coeffs, size_t dim) {
    std::array<size_t, N> freqs = {0};

    Iterate(dim, [&](const std::array<dim_t, N> &indices, size_t oned_idx) {
      count_freq(std::span<const dim_t>(indices), std::span<size_t>(freqs));
      real_t coeff = static_cast<real_t>(multinomial_coeff<N>(freqs));
      coeffs[oned_idx] = coeff;
    });
  }

private:
  template <typename Func, size_t R>
  static void iterate_impl(dim_t dim, Func &&f, std::array<dim_t, N> &indices,
                           index_t &oned_idx, dim_t start = 0) {
    if constexpr (R == 0) {
      f(indices, oned_idx);
      oned_idx++;
    } else {
      for (dim_t i = start; i < dim; ++i) {
        indices[N - R] = i;
        iterate_impl<Func, R - 1>(dim, std::forward<Func>(f), indices, oned_idx,
                                  i);
      }
    }
  }
};

template <order_t N> class Symtensor : public SymtensorBase {

public:
  Symtensor(dim_t dim) : SymtensorBase(dim, N), _data(_packed_size) {}

  Symtensor(std::span<const real_t> row)
    requires(N == 1)
      : SymtensorBase(row.size(), 1), _data(row.begin(), row.end()) {}

  void set(std::span<const real_t> row)
    requires(N == 1)
  {
    assert(row.size() == _dim);
    std::copy(row.begin(), row.end(), _data.begin());
  }

  void sym_outer_prod(const SymtensorBase &src, std::span<const real_t> row) {
    if constexpr (N == 1) {
      throw std::runtime_error("Outer product not supported for order 1 and 0");
    } else {
      const Symtensor<N - 1> &src_casted =
          dynamic_cast<const Symtensor<N - 1> &>(src);

      if (_dim >= SymtensorOps<N>::VEC_DIM) {
        SymtensorOps<N>::SymOuterProdVec(
            std::span<real_t>(_data), std::span<const real_t>(src_casted), row);
      } else {
        SymtensorOps<N>::SymOuterProd(std::span<real_t>(_data),
                                      std::span<const real_t>(src_casted), row);
      }
    }
  }

  operator std::span<const real_t>() const {
    return std::span<const real_t>(_data);
  }

  operator std::span<real_t>() { return std::span<real_t>(_data); }

  const real_t &operator[](index_t idx) const { return _data[idx]; }

  std::vector<real_t> to_tensor() const {
    std::vector<real_t> tensor(std::pow(_dim, N));
    SymtensorOps<N>::ToTensor(std::span<const real_t>(_data), _dim,
                              std::span<real_t>(tensor));
    return tensor;
  }

  void set_zero() { std::fill(_data.begin(), _data.end(), 0.0f); }

  real_t at(const std::array<dim_t, N> &indices) {
    return _data[SymtensorOps<N>::At(indices, _dim)];
  }

  void indices(index_t loc, std::array<dim_t, N> &indices) {
    SymtensorOps<N>::MemToIndex(loc, _dim, indices);
  }

private:
  std::vector<real_t> _data;
};

void sym_outer_prod(std::span<real_t> dst, std::span<const real_t> src1,
                    std::span<const real_t> src2, order_t order);
std::unique_ptr<SymtensorBase> create_symtensor(dim_t dim, order_t order);

void unpack_symtens(const std::span<const real_t> data, dim_t dim,
                    order_t order, std::span<real_t> tensor);

std::vector<real_t> symtens_coeff(size_t dim, size_t N);
real_t symtens_residual(const std::span<const real_t> core1,
                        const std::span<const real_t> core2,
                        const std::span<const real_t> coeff, size_t nrows);
real_t symtens_rel_res(const std::span<const real_t> core1,
                       const std::span<const real_t> core2,
                       const std::span<const real_t> coeff, size_t nrows);
real_t symtens_norm(const std::span<const real_t> core,
                    const std::span<const real_t> coeff, size_t nrows);

}; // namespace symprop

#endif // SYMPROP_SYMTENSOR_HPP