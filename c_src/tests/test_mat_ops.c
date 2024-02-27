#include "matrix_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
On MacOS, compile this with clang:
clang -I./ test_mat_ops.c matrix_operations.c \
-o test_mat_ops \
-framework Accelerate -DACCELERATE_NEW_LAPACK

 */

int main() {
  int n = 7;
  double A[49] = {40.4083, -4.1597, 16.7376, 2.3454,  -3.0506, -0.4950, -2.2323,
                  -4.1597, 16.7376, 0.2973,  -3.8453, 1.3566,  0.0077,  0.0335,
                  16.7376, 0.2973,  68.7481, 1.6682,  -1.7615, -0.6183, -2.7881,
                  2.3454,  -3.8453, 1.6682,  75.1030, 0.8821,  -0.0068, 0.0050,
                  -3.0506, 1.3566,  -1.7615, 0.8821,  60.1282, 0.0071,  -0.0052,
                  -0.4950, 0.0077,  -0.6183, -0.0068, 0.0071,  0.0164,  0.0361,
                  -2.2323, 0.0335,  -2.7881, 0.0050,  -0.0052, 0.0361,  0.7590};
  double *evec = malloc(7 * sizeof(double));
  double *O = malloc(n * n * sizeof(double));
  double wrk;

  printf("Eigenvalue decomposition of matrix A\n");
  printf("====================================\n");
  print_matrix(A, n, n);

  /* call 'dsyev' */
  call_dsyev(A, n, evec);

  printf("====================================\n");
  printf("The resulting eigenvector matrix Q^T\n");
  print_matrix(A, n, n);

  printf("Eigenvectors :\n");
  print_matrix(evec, n, 1);
  printf("====================================\n");

  // check orthogonality Q'*Q
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      O[i * n + j] = 0;
      for (int k = 0; k < n; ++k) {
        O[i * n + j] += A[k * n + i] * A[k * n + j];
      }
    }
  }
  printf("Orthogonality test : \n");
  print_matrix(O, n, n);

  free(O);
  free(evec);
}
