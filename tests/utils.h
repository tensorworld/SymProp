#ifndef SYMPROP_TESTS_UTILS_H
#define SYMPROP_TESTS_UTILS_H

#include <cstddef>
#include <cstdint>
#include <utils/types.hpp>
#include <vector>

namespace symprop {

std::vector<real_t> single_elem_s3ttmc(const std::vector<real_t> &u_mat,
                                       const std::vector<dim_t> &indices,
                                       size_t dim);
std::vector<real_t> single_elem_s3ttmc_naive(const std::vector<real_t> &u_mat,
                                             const std::vector<dim_t> &indices,
                                             size_t dim);

} // namespace symprop

#endif // SYMPROP_TESTS_UTILS_H