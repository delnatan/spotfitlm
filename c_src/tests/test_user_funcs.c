/*
Compile this file with:
clang -I../ ../user_funcs.c ../matrix_operations.c test_user_funcs.c \
-o test_user_funcs -framework Accelerate -DACCELERATE_NEW_LAPACK
 */

#include "../matrix_operations.h"
#include "../objective_funcs.h"
#include "../user_funcs.h"
#include <stdio.h>
#include <stdlib.h>

int main() {

  int boxsize = 13;
  int n = boxsize * boxsize;
  int m = 5;

  // parameter for symmetric gaussian
  double p[5] = {0.0, 0.0, 1.5, 50.0, 100.0};
  coord_data coords = meshgrid2d(boxsize);

  // call model function
  double *f = malloc(n * sizeof(double));
  symmetric_gaussian(p, m, n, &coords, f);
  print_matrix(f, boxsize, boxsize);

  // call jacobian function
  double *jac = malloc(n * m * sizeof(double));
  symmetric_gaussian_deriv(p, m, n, &coords, jac);
  print_matrix(jac, n, m);

  free_coord_data(&coords);
  free(f);
  free(jac);
}
