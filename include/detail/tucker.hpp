#ifndef SYMPROP_TUCKER_HPP
#define SYMPROP_TUCKER_HPP

#include <utils/types.hpp>

#include <string>
#include <cmath>
#include <vector>

namespace symprop {

class SpSymtensor;

struct Config {
  dim_t rank = 2;
  index_t max_iter = 10;
  real_t rel_tol = 1e-6;
  real_t delta = 0.5;
  index_t seed = 0;
  std::string init = "rnd";

  Config() = default;
};

class SymTuckerDecomp {
public:
  real_t reconstruct_error(const SpSymtensor &target) const;
  real_t reconstruct(const std::vector<dim_t> &inds) const;
  std::vector<real_t> reconstruct_nnz(const std::vector<dim_t> &inds,
                                      index_t num_nnz) const;

  void print(std::ostream &os) const;

  friend class SpSymtensor;

private:
  SymTuckerDecomp(const Config &config, dim_t dim, order_t order);

  dim_t _dim;
  dim_t _rank;
  order_t _order;

  std::vector<real_t> _core;
  std::vector<real_t> _factor;
};


} // namespace symprop

#endif // SYMPROP_TUCKER_HPP