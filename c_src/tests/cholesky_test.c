#include "../matrix_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int n = 5;
  int ill;
  double A[] = {446.14720, 1.15780,   -0.24820,   -0.00110, 0.00300,
                1.15780,   451.54570, 25.62790,   0.00230,  -0.00720,
                -0.24820,  25.62790,  2141.22850, 5.67210,  29.55400,
                -0.00110,  0.00230,   5.67210,    0.02820,  0.06990,
                0.00300,   -0.00720,  29.55400,   0.06990,  1.49190};

  printf("\nInput matrix\n");
  print_matrix(A, n, n, 5);

  ill = cholesky_decomposition(A, n);
  printf("\nCholesky factor\n");
  print_lower_triangular_matrix(A, n, 5);

  return ill;
}
