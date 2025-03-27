#ifndef SYMPROP_COMPRESSED_DTREE_HPP
#define SYMPROP_COMPRESSED_DTREE_HPP

#include <detail/dense/symtensor_array.hpp>
#include <detail/dtree.hpp>
#include <utils/types.hpp>

#include <algorithm>

namespace symprop {

class CompressedDTree {
public:
  CompressedDTree(const DTree &dt, const order_t depth);

  void kronecker_product(const std::vector<real_t> &U, size_t R);
  void kronecker_product_opt1(const std::vector<real_t> &U, size_t R);
  void kronecker_product_opt2(const std::vector<real_t> &U, size_t R);
  void kronecker_product_opt3(const std::vector<real_t> &U, size_t R);
  void kronecker_product_opt4(const std::vector<real_t> &U, size_t R);
  void kronecker_product_opt5(const std::vector<real_t> &U, size_t R,
                              std::vector<real_t> &Y, size_t ldy);
  const auto &get_leaf_kptensors() const { return leaf_kptensors; }
  const auto &get_leaf_kptensors_array() const { return leaf_kptensors_array; }
  void print() const;
  index_t exclude_search(std::span<const dim_t> indices, order_t exclude_index);

private:
  void init_dep_list();

  template <std::ranges::view View> index_t search(View view) {
    index_t res_idx = 0;
    index_t level = 0;
    index_t level_start = 0;
    index_t level_end = level_size[level];

    for (auto i : view) {
      auto begin = level_idx[level].begin();
      auto it = std::lower_bound(begin + level_start, begin + level_end, i);
      assert(it != begin + level_end);
      assert(*it == i);

      index_t next = std::distance(begin, it);
      level_start = level_ptr[level][next];
      level_end = level_ptr[level][next + 1];
      level++;
      res_idx = next;
    }
    return res_idx;
  }

  order_t depth;
  std::vector<index_t> level_size;
  std::vector<std::vector<dim_t>> level_idx;
  std::vector<std::vector<index_t>> level_ptr;

  std::vector<std::vector<index_t>> dep_list;

  std::vector<std::pair<dim_t, real_t>> nonzero;
  std::vector<std::unique_ptr<SymtensorBase>> leaf_kptensors;
  std::unique_ptr<SymtensorArray> leaf_kptensors_array;

  std::unique_ptr<Buffer2D<dim_t>> p_parent_indices;
  std::unique_ptr<Buffer2D<dim_t>> p_child_indices;
};

} // namespace symprop

#endif // SYMPROP_COMPRESSED_DTREE_HPP