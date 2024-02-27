#include "stdio.h"
#include "time.h"
#include <lapack.h>

void print_matrix(double *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    printf("|");
    for (int j = 0; j < n; j++) {
      printf("%8.3f ", A[i * n + j]);
    }
    printf("|\n");
  }
}

#define N 7
#define LDA 7

int main() {
  char JOBZ = 'V';
  char UPLO = 'L';
  int lwork = -1;
  int n = N;
  int lda = LDA;
  double wrkopt;
  double work[N * N] = {0};
  double eigvals[N] = {0};
  int info;

  double A[N * N] = {2.3426, 1.8979, 1.9242, 1.6442, 0.8490, 2.2657, 1.5380,
                     1.8979, 2.3117, 2.1959, 1.8782, 1.0409, 1.7236, 1.5002,
                     1.9242, 2.1959, 3.4117, 2.0735, 1.2699, 2.1691, 1.9324,
                     1.6442, 1.8782, 2.0735, 1.8188, 0.8576, 1.2362, 1.5775,
                     0.8490, 1.0409, 1.2699, 0.8576, 0.5776, 0.8785, 0.7525,
                     2.2657, 1.7236, 2.1691, 1.2362, 0.8785, 3.0764, 1.1641,
                     1.5380, 1.5002, 1.9324, 1.5775, 0.7525, 1.1641, 1.5794};
  /* note that A will be the eigenvectors, Q^T because of the column-major
   ordering of LAPACK*/

  clock_t start, end;
  double compute_time;

  start = clock();
  // optimize workspace
  LAPACK_dsyev("V", "U", &n, A, &lda, eigvals, &wrkopt, &lwork, &info);
  lwork = (int)wrkopt;
  // solve problem
  LAPACK_dsyev("V", "U", &n, A, &lda, eigvals, work, &lwork, &info);
  end = clock();
  compute_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Problem took %.2f ms\n", compute_time * 1000.0);
  printf("LWORK = %d\n", lwork);
  printf("INFO = %d\n", info);
  print_matrix(eigvals, N, 1);
  print_matrix(A, N, N);
}
