#include "matrix_operations.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <time.h>

/*
On MacOS, compile this with clang:
clang -I./ cholesky_solve_test.c matrix_operations.c \
-o cholesky_solve_test \
-framework Accelerate -DACCELERATE_NEW_LAPACK


 */
void print_matrix(double *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    printf("|");
    for (int j = 0; j < n; j++) {
      printf("%7.4f ", A[i * n + j]);
    }
    printf("|\n");
  }
}

void print_lower_triangular(double *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    printf("|");
    for (int j = 0; j < n; j++) {
      if (j <= i) {
        printf("%7.4f ", A[i * n + j]);
      } else {
        printf("%7.4f ", 0.0);
      }
    }
    printf("|\n");
  }
}

int main() {

  // for timing
  clock_t start, end;
  double compute_time;

  int n = 7;   // size of matrix
  int lda = n; // leading dimension
  int info;
  char uplo = 'U';

  double A[49] = {40.4083, -4.1597, 16.7376, 2.3454,  -3.0506, -0.4950, -2.2323,
                  -4.1597, 16.7376, 0.2973,  -3.8453, 1.3566,  0.0077,  0.0335,
                  16.7376, 0.2973,  68.7481, 1.6682,  -1.7615, -0.6183, -2.7881,
                  2.3454,  -3.8453, 1.6682,  75.1030, 0.8821,  -0.0068, 0.0050,
                  -3.0506, 1.3566,  -1.7615, 0.8821,  60.1282, 0.0071,  -0.0052,
                  -0.4950, 0.0077,  -0.6183, -0.0068, 0.0071,  0.0164,  0.0361,
                  -2.2323, 0.0335,  -2.7881, 0.0050,  -0.0052, 0.0361,  0.7590};
  double b[7] = {0.5707, 0.8331, 0.1899, 0.3121, 0.0210, 0.5516, 0.0274};
  double x[7];

  start = clock();
  // solve for x
  solve_Axb_cholesky(A, b, n, x);
  end = clock();

  compute_time = ((double)(end - start)) / CLOCKS_PER_SEC;

  for (int i = 0; i < 7; i++) {
    printf("%7.4f ", x[i]);
  }
  printf("\n");
  printf("Solving a %d x %d system took %.6f milliseconds\n", n, n,
         compute_time * 1000);

  return 0;
}
