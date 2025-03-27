#include <utils/loader.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace symprop {

SymtensLoader::SymtensLoader(const std::string &filename) : _filename(filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    throw std::runtime_error("Error opening file in SymtensLoader");
  }

  file >> _order >> _dim;
  std::vector<dim_t> nnz_index;

  while (true) {
    nnz_index.clear();
    for (order_t i = 0; i < _order; i++) {
      dim_t val;
      file >> val;
      nnz_index.push_back(val);
    }

    real_t val;
    file >> val;
    if (!file.good())
      break;
    std::sort(nnz_index.begin(), nnz_index.end());
    _unnzs.push_back(val);
    _unnz_inds.insert(_unnz_inds.end(), nnz_index.begin(), nnz_index.end());
  }

  file.close();
}

} // namespace symprop
