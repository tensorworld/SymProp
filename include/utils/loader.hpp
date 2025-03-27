#ifndef SYMPROP_LOADER_HPP
#define SYMPROP_LOADER_HPP

#include <utils/types.hpp>
#include <string>
#include <vector>

namespace symprop {

class SymtensLoader {
public:
  SymtensLoader(const std::string &filename);
  ~SymtensLoader() = default;

  const std::vector<dim_t> &unnz_inds() const { return _unnz_inds; }
  const std::vector<real_t> &unnzs() const { return _unnzs; }
  order_t order() const { return _order; }
  dim_t dim() const { return _dim; }

private:
  std::string _filename;
  std::vector<dim_t> _unnz_inds;
  std::vector<real_t> _unnzs;

  order_t _order;
  dim_t _dim;
};

} // namespace symprop

#endif // SYMPROP_LOADER_HPP