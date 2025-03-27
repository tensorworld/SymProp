#ifndef SYMPROP_SYMTENSOR_BASE_HPP
#define SYMPROP_SYMTENSOR_BASE_HPP

#include <array>
#include <cstdint>
#include <span>
#include <utils/types.hpp>
#include <vector>

namespace symprop {

inline index_t binomial_coefficient(index_t n, index_t k) {
  if (k > n)
    return 0;
  if (k * 2 > n)
    k = n - k;
  if (k == 0)
    return 1;

  index_t result = n;
  for (index_t i = 2; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}

inline index_t symtens_size(index_t dim, index_t order) {
  return binomial_coefficient(order + dim - 1, order);
}

template <index_t L> constexpr std::array<index_t, L> factorial_table() {
  std::array<index_t, L> table = {1};
  for (index_t i = 1; i < L; i++) {
    table[i] = table[i - 1] * i;
  }
  return table;
}

constexpr auto factorial = factorial_table<20>();

template <index_t N>
index_t multinomial_coeff(const std::array<index_t, N> &freqs) {
  index_t denom = 1;
  for (index_t i = 0; i < N; i++) {
    denom *= factorial[freqs[i]];
  }
  return factorial[N] / denom;
}

inline index_t multinomial_coeff(const std::span<index_t> freqs) {
  index_t denom = 1;
  for (index_t i = 0; i < freqs.size(); i++) {
    denom *= factorial[freqs[i]];
  }
  return factorial[freqs.size()] / denom;
}

template <typename T1, typename T2>
inline void count_freq(const std::span<T1> indices,
                       std::span<T2> freqs) {
  std::fill(freqs.begin(), freqs.end(), 0);
  index_t prev = indices[0];
  index_t prev_idx = 0;
  freqs[0] = 1;

  for (index_t j = 1; j < indices.size(); j++) {
    if (indices[j] == prev) {
      freqs[prev_idx] += 1;
    } else {
      prev = indices[j];
      prev_idx = j;
      freqs[j] = 1;
    }
  }
}

class SymtensorBase {
public:
  SymtensorBase(dim_t dim, order_t order)
      : _dim(dim), _order(order), _packed_size(symtens_size(dim, order)) {}

  order_t order() const { return _order; }
  dim_t dim() const { return _dim; }
  index_t packed_size() const { return _packed_size; }

  virtual void sym_outer_prod(const SymtensorBase &src,
                              std::span<const real_t> row) = 0;
  virtual void set_zero() = 0;
  virtual std::vector<real_t> to_tensor() const = 0;
  virtual operator std::span<const real_t>() const = 0;
  virtual operator std::span<real_t>() = 0;
  virtual ~SymtensorBase() = default;

protected:
  dim_t _dim;
  order_t _order;
  index_t _packed_size;
};

} // namespace symprop

#endif // SYMPROP_SYMTENSOR_BASE_HPP