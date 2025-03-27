#include <detail/dense/symtensor.hpp>
#include <utils/dispatcher.hpp>
#include <utils/function_timer.hpp>

#include <functional>

namespace symprop {

template <size_t Order> struct SymtensProdFunctor {
  void operator()(std::span<real_t> dst, std::span<const real_t> src1,
                  std::span<const real_t> src2) const {
    SymtensorOps<Order>::SymOuterProd(dst, src1, src2);
  }
};

void sym_outer_prod(std::span<real_t> dst, std::span<const real_t> src1,
                    std::span<const real_t> src2, order_t order) {
  static const auto dispatcher = make_dispatcher<SymtensProdFunctor, 2, 20>();
  dispatcher(order, dst, src1, src2);
}

template <size_t Order> struct SymtensorCreator {
  std::unique_ptr<SymtensorBase> operator()(dim_t dim) const {
    return std::make_unique<Symtensor<Order>>(dim);
  }
};

std::unique_ptr<SymtensorBase> create_symtensor(dim_t dim, order_t order) {
  static const auto dispatcher = make_dispatcher<SymtensorCreator, 1, 20>();
  return dispatcher(order, dim);
}

template <size_t Order> struct SymtensorUnpackFunctor {
  void operator()(const std::span<const real_t> data, dim_t dim,
                  std::span<real_t> tensor) const {
    SymtensorOps<Order>::ToTensor(data, dim, tensor);
  }
};

void unpack_symtens(const std::span<const real_t> data, dim_t dim,
                    order_t order, std::span<real_t> tensor) {
  static const auto dispatcher =
      make_dispatcher<SymtensorUnpackFunctor, 1, 20>();
  dispatcher(order, data, dim, tensor);
}

template <size_t N> struct SymtensorCoeffFunctor {
  void operator()(std::vector<real_t> &coeffs, size_t dim) const {
    SymtensorOps<N>::SymtensCoeff(coeffs, dim);
  }
};

std::vector<real_t> symtens_coeff(size_t dim, size_t N) {
  std::vector<real_t> coeffs(symtens_size(dim, N));
  static const auto dispatcher =
      make_dispatcher<SymtensorCoeffFunctor, 1, 20>();
  dispatcher(N, coeffs, dim);
  return coeffs;
}

real_t symtens_residual(const std::span<const real_t> core1,
                        const std::span<const real_t> core2,
                        const std::span<const real_t> coeff, size_t nrows) {
  real_t residual = 0.0f;

  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < coeff.size(); j++) {
      residual +=
          std::pow(core1[i * coeff.size() + j] - core2[i * coeff.size() + j],
                   2) *
          coeff[j];
    }
  }

  return residual;
}

real_t symtens_rel_res(const std::span<const real_t> core1,
                       const std::span<const real_t> core2,
                       const std::span<const real_t> coeff, size_t nrows) {
  real_t residual = 0.0f;
  real_t norm = 0.0f;

  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < coeff.size(); j++) {
      residual +=
          std::pow(core1[i * coeff.size() + j] - core2[i * coeff.size() + j],
                   2) *
          coeff[j];
      norm += std::pow(core1[i * coeff.size() + j], 2) * coeff[j];
    }
  }

  return residual / norm;
}

real_t symtens_norm(const std::span<const real_t> core,
                    const std::span<const real_t> coeff, size_t nrows) {
  real_t norm = 0.0f;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < coeff.size(); j++) {
      norm += std::pow(core[i * coeff.size() + j], 2) * coeff[j];
    }
  }
  return std::sqrt(norm);
}

} // namespace symprop
