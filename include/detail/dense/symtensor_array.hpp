#ifndef SYMPROP_SYMTENSOR_ARRAY_HPP
#define SYMPROP_SYMTENSOR_ARRAY_HPP

#include <detail/dense/symtensor_base.hpp>
#include <utils/buffer2d.hpp>
#include <utils/types.hpp>

#include <cassert>
#include <vector>

namespace symprop {

class SymtensorArray {

  dim_t _dim;
  order_t _order;
  index_t _count;
  index_t _elem_size;
  Buffer2D<real_t> _data;

public:
  SymtensorArray(dim_t dim, order_t order, index_t count, bool zero_init = true)
      : _dim(dim), _order(order), _count(count),
        _elem_size(symtens_size(dim, order)), _data(count, _elem_size) {
    if (!zero_init) {
      return;
    }
    auto data_span = std::span<real_t>(_data);
#pragma omp parallel for
    for (index_t i = 0; i < count * _elem_size; i++) {
      data_span[i] = 0.0;
    }
  }

  SymtensorArray(const SymtensorArray &other) = delete;
  SymtensorArray &operator=(const SymtensorArray &other) = delete;

  SymtensorArray(SymtensorArray &&other) noexcept
      : _dim(other._dim), _order(other._order), _count(other._count),
        _elem_size(other._elem_size), _data(std::move(other._data)) {}

  SymtensorArray &operator=(SymtensorArray &&other) noexcept {
    _dim = other._dim;
    _order = other._order;
    _count = other._count;
    _elem_size = other._elem_size;
    _data = std::move(other._data);
    return *this;
  }

  void set(index_t index, const std::span<const real_t> data) {
    assert(data.size() == _elem_size);
    assert(index < _count);
    std::copy(data.begin(), data.end(), _data[index].begin());
  }

  void zero_init(index_t index) {
    assert(index < _count);
    auto data_span = std::span<real_t>(_data[index]);
    for (index_t i = 0; i < _elem_size; i++) {
      data_span[i] = 0.0;
    }
  }

  void sym_outer_prod(index_t dst_index, const SymtensorArray &src,
                      index_t src_index, const std::span<const real_t> row);

  std::span<const real_t> symtensor_data(index_t index) const {
    assert(index < _count);
    return _data[index];
  }
};

} // namespace symprop

#endif // SYMPROP_SYMTENSOR_ARRAY_HPP