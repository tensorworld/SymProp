#include <detail/dense/matrixops.hpp>
#include <utils/function_timer.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

using namespace symprop;

void matrix::leftsvd(std::vector<real_t> &A, std::vector<real_t> &U, size_t nr,
                     size_t nc, size_t rank) {
  START_FUNCTION_TIMER();

  size_t min_dim = std::min(nr, nc);
  double *s = new double[min_dim];
  double *superb = new double[min_dim - 1];
  int nrows = static_cast<int>(nr);
  int ncols = static_cast<int>(nc);

  int info;
  info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'O', 'N', nrows, ncols, A.data(),
                        ncols, s, nullptr, 1, nullptr, 1, superb);
  if (info > 0) {
    printf("SVD failed to converge.\n");
  }

  for (size_t i = 0; i < nr; i++) {
    for (size_t j = 0; j < rank; j++) {
      U[i * rank + j] = A[i * ncols + j];
    }
  }

  delete[] s;
  delete[] superb;
}

void matrix::gemmtn(std::vector<real_t> &C, const std::vector<real_t> &A,
                    const std::vector<real_t> &B, size_t arow, size_t acol,
                    size_t bcol) {
  START_FUNCTION_TIMER();

  int m = static_cast<int>(acol);
  int n = static_cast<int>(bcol);
  int k = static_cast<int>(arow);

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, A.data(),
              m, B.data(), n, 0.0, C.data(), n);
}

void matrix::qr(std::vector<real_t> &A, size_t nr, size_t nc) {
  START_FUNCTION_TIMER();
  int nrows = static_cast<int>(nr);
  int ncols = static_cast<int>(nc);

  int lda = ncols;
  int info;
  int min_dim = nrows < ncols ? nrows : ncols;
  double *tau = new double[min_dim];

  info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, nrows, ncols, A.data(), lda, tau);
  if (info > 0) {
    printf("QR decomposition failed to converge.\n");
    return;
  }
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, nrows, ncols, min_dim, A.data(), lda, tau);
  delete[] tau;
}

void matrix::syevd(std::vector<real_t> &A, std::vector<real_t> &U, size_t dim,
                   size_t rank) {
  START_FUNCTION_TIMER();
  int ncols = static_cast<int>(dim);
  int lda = ncols;
  int info;
  double *w = new double[dim];

  info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', ncols, A.data(), lda, w);
  if (info > 0) {
    printf("Eigendecomposition failed to converge.\n");
    return;
  }

  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < rank; j++) {
      // size_t index = dim - j - 1;
      size_t index = dim - rank + j;
      U[i * rank + j] = A[i * dim + index];
    }
  }

  delete[] w;
}

void matrix::symgemm(std::vector<real_t> &C, const std::vector<real_t> &A,
                     const std::vector<real_t> &B,
                     const std::vector<real_t> &row_coeff, size_t I1,
                     size_t I2) {
  START_FUNCTION_TIMER();
  assert(I1 * row_coeff.size() == A.size());
  assert(I2 * row_coeff.size() == B.size());

  std::vector<real_t> B_copy = B;
  for (size_t i = 0; i < I2; i++) {
    for (size_t j = 0; j < row_coeff.size(); j++) {
      B_copy[i * row_coeff.size() + j] *= row_coeff[j];
    }
  }

  int nrows1 = static_cast<int>(I1);
  int nrows2 = static_cast<int>(I2);
  int ncols1 = static_cast<int>(row_coeff.size());

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nrows1, nrows2, ncols1,
              1.0, A.data(), ncols1, B_copy.data(), ncols1, 0.0, C.data(),
              nrows2);
}

void matrix::symsyrk(std::vector<real_t> &C, const std::vector<real_t> &A,
                     const std::vector<real_t> &row_coeff, size_t I1) {
  START_FUNCTION_TIMER();
  assert(I1 * row_coeff.size() == A.size());

  std::vector<real_t> A_copy = A;

  for (size_t i = 0; i < I1; i++) {
    for (size_t j = 0; j < row_coeff.size(); j++) {
      A_copy[i * row_coeff.size() + j] *= std::sqrt(row_coeff[j]);
    }
  }

  int nrows = static_cast<int>(I1);
  int ncols = static_cast<int>(row_coeff.size());

  assert(nrows * nrows == (int)C.size());

  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, nrows, ncols, 1.0,
              A_copy.data(), ncols, 0.0, C.data(), nrows);
}