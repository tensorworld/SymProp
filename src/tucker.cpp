#include "detail/dense/matrixops.hpp"
#include <detail/sparse/spsymtensor.hpp>
#include <detail/tucker.hpp>

#include <algorithm>
#include <random>

namespace symprop {

auto normal() {
  static std::normal_distribution<real_t> distr(0.0, 1.0);
  static std::random_device device;
  static std::mt19937 engine{device()};
  return distr(engine);
}

auto norm(const std::vector<real_t> &x) {
  real_t norm = 0.0f;
#pragma omp parallel for reduction(+ : norm)
  for (size_t i = 0; i < x.size(); i++)
    norm += std::pow(x[i], 2);
  return std::sqrt(norm);
}

SymTuckerDecomp::SymTuckerDecomp(const Config &config, dim_t dim, order_t order)
    : _dim(dim), _rank(config.rank), _order(order) {
  _core.resize(std::pow(_rank, order));
  _factor.resize(dim * _rank);

  std::normal_distribution<real_t> distr(0.0, 1.0);
  std::mt19937 engine{config.seed};
  // todo: different initialization methods
  // std::generate(_factor.begin(), _factor.end(), distr(engine));
  for (size_t i = 0; i < _factor.size(); i++)
    _factor[i] = distr(engine);
}

// formula from On the best rank-l and rank-(Rl, R2,..., Rn) approximation of
// higher order tensors
real_t SymTuckerDecomp::reconstruct_error(const SpSymtensor &target) const {
  real_t sp_norm = target.norm();
  real_t core_norm = norm(_core);
  // return sp_norm * sp_norm - core_norm * core_norm;
  return std::sqrt(sp_norm * sp_norm - core_norm * core_norm) / sp_norm;
}

void SymTuckerDecomp::print(std::ostream &os) const {
  os << "Mode-1 matricized C:\n";
  for (size_t i = 0; i < _rank; i++) {
    for (size_t j = 0; j < _core.size() / _rank; j++)
      os << _core[i * _rank + j] << ",";
    os << "\n";
  }

  os << "Factor U:\n";
  for (size_t i = 0; i < _dim; i++) {
    for (size_t j = 0; j < _rank; j++)
      os << _factor[i * _rank + j] << ",";
    os << "\n";
  }
}


} // namespace symprop
