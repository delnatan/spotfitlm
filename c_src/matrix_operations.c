/* #include "lapack.h" */
#include "matrix_operations.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-12

void matrix_inverse_from_cholesky(double *L, int n, double *result) {
  /* Computes matrix inverse from cholesky factor L
     (L is an lower triangular matrix obtained from the routine below
     cholesky_decomposition())
     Note: computes the inverse in-place. On output, L will contain the matrix
     inverse.
   */
  int i, j, k;
  double sum;
  // assume L is ok
  for (j = 0; j < n; j++) {
    // Diagonal elements
    L[j * n + j] = 1.0 / L[j * n + j];

    for (i = j + 1; i < n; i++) {
      sum = 0.0;
      for (k = j; k < i; k++) {
        sum -= L[i * n + k] * L[k * n + j];
      }
      L[i * n + j] = sum / L[i * n + i];
    }
  }

  // compute matrix inverse A^-1 = ( L*L' )^{-1}
  // L^{-t} * L^{-1}
  // since L is now the lower-triangular inverse
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) {
      result[i * n + j] = 0.0;
      // only sum the non-zero terms
      for (k = (i > j ? i : j); k < n; k++) {
        result[i * n + j] += L[k * n + i] * L[k * n + j];
      }
      // fill upper-triangular part with lower-triangular part
      result[j * n + i] = result[i * n + j];
    }
  }
}

int solve_Axb_cholesky(double *A, double *mu, double *b, int n, double *x) {
  /* Solves Ax = b via Cholesky decomposition
     where A is a symmetric positive-definite n-by-n matrix
     if `mu` is not NULL, then A is damped by the diagonal damping term
                                 B = A + mu * I
     where I is an identity matrix
     returns result in vector x

     Note: A is kept intact because this routines makes a copy
  */
  int i, j;
  int ill;

  // make a copy of the matrix A to keep the cholesky factor, C
  double *C = malloc(n * n * sizeof(double));

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      C[i * n + j] = A[i * n + j];
    }
  }

  if (mu) {
    // augment diagonal with constant term 'mu'
    for (i = 0; i < n; i++) {
      C[i * n + i] += *mu;
    }
  }

  // 'A' will contain the lower-triangular matrix 'L'
  ill = cholesky_decomposition(C, n);

  if (!ill) {
    // forward substitution L*y = b
    for (i = 0; i < n; i++) {
      x[i] = b[i];
      for (j = 0; j < i; j++) {
        x[i] -= C[i * n + j] * x[j];
      }
      x[i] /= C[i * n + i];
    }
    // backward substitution, L'*x = y
    for (i = n - 1; i >= 0; i--) {
      for (j = i + 1; j < n; j++) {
        x[i] -= C[j * n + i] * x[j];
      }
      x[i] /= C[i * n + i];
    }

    free(C);

    return ill;

  } else {
    // fill solution with all zeros
    for (i = 0; i < n; i++) {
      x[i] = 0.0;
    }

    free(C);
    return ill;
  }
}

int cholesky_decomposition(double *A, int n) {
  double x, r;
  int i, j, k;

  for (j = 0; j < n; j++) {
    x = A[j * n + j]; /* A_jj */
    for (k = 0; k < j; k++) {
      x -= A[j * n + k] * A[j * n + k];
    }
    if (x < 0) {
      return -1;
    }
    x = sqrt(x);

    A[j * n + j] = x; /* L_jj */
    r = 1.0 / x;

    /* i != j */
    for (i = j + 1; i < n; i++) {
      x = A[i * n + j];
      for (k = 0; k < j; k++) {
        x -= A[i * n + k] * A[j * n + k];
      }
      A[i * n + j] = x * r;
    }
  }
  return 0;
}

void print_matrix(double *A, int m, int n, int precision) {
  for (int i = 0; i < m; i++) {
    printf("|");
    for (int j = 0; j < n; j++) {
      printf("%12.*f ", precision, A[i * n + j]);
    }
    printf("|\n");
  }
}

void print_lower_triangular_matrix(double *A, int n, int precision) {
  for (int i = 0; i < n; i++) {
    printf("|");
    for (int j = 0; j < n; j++) {
      if (j <= i) {
        printf("%12.*f", precision, A[i * n + j]);
      } else {
        printf("%12.*f", precision, 0.0);
      }
    }
    printf("|\n");
  }
}
