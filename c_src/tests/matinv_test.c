#include "../matrix_operations.h"
#include "lapack.h"
#include <stdio.h>

#define N 5
#define LDA 5

int main() {
  int n = N;
  int lda = LDA;
  int info;

  double B[N * N] = {0.70876,  -1.80694, 0.12056,  0.64248,  -0.16320,
                     -1.80694, 8.67368,  3.63429,  -3.81736, -0.23798,
                     0.12056,  3.63429,  3.94853,  -2.59465, -0.49509,
                     0.64248,  -3.81736, -2.59465, 5.61326,  -0.59172,
                     -0.16320, -0.23798, -0.49509, -0.59172, 0.48998};
  printf("Input matrix : \n");
  print_matrix(B, n, n, 4);
  printf("\n");

  // do cholesky decomposition. Compute upper-triangular matrix
  LAPACK_dpotrf("U", &n, B, &lda, &info);
  printf("Cholesky matrix\n");
  print_matrix(B, n, n, 4);
  printf("\n");

  // invert matrix with cholesky matrix
  LAPACK_dpotri("U", &n, B, &lda, &info);
  printf("Matrix inverse \n");
  print_matrix(B, n, n, 4);
  printf("\n\n");

  printf("info = %d", info);
}
