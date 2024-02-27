/*
Compile this file with:
clang -I../ ../user_funcs.c ../glm_core.c ../matrix_operations.c \
../objective_funcs.c func_fit_test.c -o func_fit_test \
-framework Accelerate -DACCELERATE_NEW_LAPACK
 */

#include "../glm_core.h"
#include "../matrix_operations.h"
#include "../objective_funcs.h"
#include "../user_funcs.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

double *read_binary_image(const char *filename, int nrow, int ncol) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    perror("Error opening file.");
    return NULL;
  }

  // allocate array
  uint16_t *array = (uint16_t *)malloc(nrow * ncol * sizeof(uint16_t));
  double *darray = (double *)malloc(nrow * ncol * sizeof(double));

  if (array == NULL) {
    perror("Error allocating memory");
    fclose(file);
    return NULL;
  }

  if (fread(array, sizeof(uint16_t), nrow * ncol, file) != nrow * ncol) {
    perror("Error reading file");
    free(array);
    fclose(file);
    return NULL;
  }

  fclose(file);

  // convert to double
  for (int i = 0; i < nrow * ncol; i++) {
    darray[i] = (double)array[i];
  }

  free(array);

  return darray;
}

int main() {

  int nrow = 13;
  int ncol = 13;

  int n = nrow * nrow;
  int m = 5;

  coord_data coords = meshgrid2d(13);
  double *data = read_binary_image("mol01_13x13.bin", nrow, ncol);
  double *covar = malloc(m * m * sizeof(double));

  double p[5] = {0.0, 0.0, 1.0, 10.0, 10.0};

  OptimizerResult opt;

  opt = dglm_der(&poisson_nll, &poisson_nll_grad, &poisson_nll_appx_hess,
                 &symmetric_gaussian, &symmetric_gaussian_deriv,
                 &symmetric_gaussian_poisson_nll_hess, p, data, m, n, 100,
                 covar, &coords, 1);

  printf("covariance matrix := \n");
  print_matrix(covar, m, m, 5);

  printf("iter = %d\n", opt.num_iterations);
  printf("-logL = %.4f\n", opt.objective_value);
  printf("|g'g| = %.4e\n", opt.scaled_grad_norm);
  printf("ret = %d\n", opt.return_code);

  free_coord_data(&coords);
  free(data);
  free(covar);
}
