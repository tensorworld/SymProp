#ifndef SYMPROP_SPSYMTENSOR_HPP
#define SYMPROP_SPSYMTENSOR_HPP

#include <detail/dtree.hpp>
#include <detail/compressed_dtree.hpp>
#include <detail/tucker.hpp>
#include <utils/types.hpp>

#include <cassert>
#include <vector>

namespace symprop {

struct CSC {
  std::vector<double> values;
  std::vector<index_t> row_indices;
  std::vector<index_t> col_pointers;
  index_t rows;
  index_t cols;
};

struct COO {
  std::vector<std::tuple<index_t, index_t, double>> entries;
};

class SpSymtensor {
public:
  SpSymtensor(dim_t dim, order_t order, const std::vector<dim_t> &unnz_inds,
              const std::vector<real_t> &unnzs);

  order_t order() const { return _order; }
  dim_t dim() const { return _dim; }
  index_t size() const { return _unnz; }

  const std::vector<dim_t> &unnz_inds() const { return _unnz_inds; }
  const std::vector<real_t> &unnzs() const { return _unnzs; }

  void S3TTMc(std::vector<real_t> &res, const std::vector<real_t> &U, dim_t R,
              bool partial_sym_res = false) const;
  SymTuckerDecomp TuckerHOOI(const Config &c) const;
  SymTuckerDecomp TuckerHOQRI(const Config &c) const;
  SymTuckerDecomp TuckerAltHOOI(const Config &c) const;

  real_t norm() const;
private:
  real_t calc_norm() const;
  void hosvd_init(std::vector<real_t> &U, const Config &c) const;
  // CSC convert_to_csc() const;
  COO convert_to_coo() const;
  void sparse_syrk(const COO &A, std::vector<real_t> &result) const;

  std::span<const dim_t> _get_unnz_inds(size_t i) const {
    return std::span<const dim_t>(_unnz_inds).subspan(i * _order, _order);
  }

  dim_t _dim;
  order_t _order;
  index_t _unnz;
  real_t _norm;
  std::unique_ptr<DTree> _dtree;
  std::unique_ptr<CompressedDTree> _cdtree;

  std::vector<dim_t> _unnz_inds;
  std::vector<real_t> _unnzs;
};

}; // namespace symprop

#endif // SYMPROP_SPSYMTENSOR_HPP