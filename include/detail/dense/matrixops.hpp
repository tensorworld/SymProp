#ifndef SYMPROP_DENSE_MATRIXOPS_HPP
#define SYMPROP_DENSE_MATRIXOPS_HPP

#include <utils/types.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace symprop {
namespace matrix {

// make sure that the real_t is a double
static_assert(std::is_same_v<real_t, double>);

void leftsvd(std::vector<real_t> &A, std::vector<real_t> &U, size_t nr,
             size_t nc, size_t rank);

void gemmtn(std::vector<real_t> &C, const std::vector<real_t> &A,
            const std::vector<real_t> &B, size_t arow, size_t acol,
            size_t bcol);

void qr(std::vector<real_t> &A, size_t nr, size_t nc);

void syevd(std::vector<real_t> &A, std::vector<real_t> &U, size_t dim, size_t rank); 

// unfolded symmetric row gemm
// A: matrix of size (I1 x symtensor_size(dim, order))
// B: matrix of size (I2 x symtensor_size(dim, order))
// C: matrix of size (I1 x I2)
void symgemm(std::vector<real_t> &C, const std::vector<real_t> &A,
             const std::vector<real_t> &B, const std::vector<real_t> &row_coeff,
             size_t I1, size_t I2);

// unfolded symmtric row syrk
// A: matrix of size (I1 x symtensor_size(dim, order))
// C: matrix of size (I1 x I1)
void symsyrk(std::vector<real_t> &C, const std::vector<real_t> &A,
             const std::vector<real_t> &row_coeff, size_t I1);



} // namespace matrix
} // namespace symprop

#endif // SYMPROP_MATRIXOPS_HPP