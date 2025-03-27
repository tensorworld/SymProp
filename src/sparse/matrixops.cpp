#include <cstdio>
#include <detail/sparse/matrixops.hpp>

#ifdef USE_MKL
#include "mkl_spblas.h"
#include <mkl.h>
#endif

namespace symprop {

void matrix::sparse_leftsvd(matrix::COO<int> &A, std::vector<real_t> &U,
                            size_t rank) {

#ifndef USE_MKL
  printf("Error: sparse SVD requires MKL.\n");
  return;
#else
  sparse_matrix_t coo_A, csr_A;
  int nnz = static_cast<int>(A.values.size());

  sparse_status_t info = mkl_sparse_d_create_coo(
      &coo_A, SPARSE_INDEX_BASE_ZERO, A.rows, A.cols, nnz, A.row_indices.data(),
      A.col_indices.data(), A.values.data());

  if (info != SPARSE_STATUS_SUCCESS) {
    printf("Failed to create COO matrix.\n");
    return;
  }

  mkl_sparse_convert_csr(coo_A, SPARSE_OPERATION_NON_TRANSPOSE, &csr_A);

  int pm[128] = {0, 6, 0, 0, 0, 0, 1, 0, 0, 0};
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr.mode = SPARSE_FILL_MODE_FULL;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  int k0 = static_cast<int>(rank);
  int k;
  double *eig = new double[rank];
  double *res = new double[rank];
  // double *right = new double[rank * A.cols];
  double *left = new double[rank * A.rows];

  char arg[1] = {'L'};
  info = mkl_sparse_d_svd(arg, arg, pm, csr_A, descr, k0, &k, eig, left,
                          nullptr, res);
  if (info != SPARSE_STATUS_SUCCESS || k != k0) {
    printf("Failed to compute SVD.\n");
    return;
  }
  // mkl_sparse_d_svd()
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < k0; j++) {
      U[i * rank + j] = left[j * A.rows + i];
    }
  }

  delete[] left;
  delete[] eig;
  delete[] res;
  mkl_sparse_destroy(csr_A);
  mkl_sparse_destroy(coo_A);
#endif        
}

} // namespace symprop
