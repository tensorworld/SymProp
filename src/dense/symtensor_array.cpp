#include <detail/dense/symtensor.hpp>
#include <detail/dense/symtensor_array.hpp>

namespace symprop {

void SymtensorArray::sym_outer_prod(index_t dst_index,
                                    const SymtensorArray &src,
                                    index_t src_index,
                                    const std::span<const real_t> row) {
  assert(row.size() == _dim);
  assert(src._dim == _dim);
  assert(src._order == _order - 1);
  assert(src_index < src._count);
  assert(dst_index < _count);

  auto dst_data = _data[dst_index];
  auto src_data = src._data[src_index];

  switch (_order) {
  case 2: {
    SymtensorOps<2>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 3: {
    SymtensorOps<3>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 4: {
    SymtensorOps<4>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 5: {
    SymtensorOps<5>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 6: {
    SymtensorOps<6>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 7: {
    SymtensorOps<7>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 8: {
    SymtensorOps<8>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 9: {
    SymtensorOps<9>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 10: {
    SymtensorOps<10>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 11: {
    SymtensorOps<11>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 12: {
    SymtensorOps<12>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 13: {
    SymtensorOps<13>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 14: {
    SymtensorOps<14>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 15: {
    SymtensorOps<15>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  case 16: {
    SymtensorOps<16>::SymOuterProd(dst_data, src_data, row);
    break;
  }
  default:
    throw std::runtime_error(
        "SymtensorArray::sym_outer_prod: order not supported");
  }
}

} // namespace symprop