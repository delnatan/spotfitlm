/*
Note, compile with:
clang -I../ ../user_funcs.c ../objective_funcs.c ../matrix_operations.c
test_opt_funcs.c \ -o test_opt_funcs -framework Accelerate
-DACCELERATE_NEW_LAPACK
*/

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
  int n = nrow * ncol;
  int m = 5;

  coord_data coords = meshgrid2d(13);
  double *image = read_binary_image("mol01_13x13.bin", nrow, ncol);

  printf("Image data read.\n");
  print_matrix(image, nrow, ncol, 0);
  printf("=== END of image data ===\n");

  double *f = malloc(nrow * ncol * sizeof(double));
  double *jac = malloc(n * m * sizeof(double));
  double *grad = malloc(m * sizeof(double));
  double *hess = malloc(m * m * sizeof(double));
  // parameters
  double p[5] = {0.0, 0.0, 1.5, 50.0, 100.0};

  symmetric_gaussian(p, m, n, &coords, f);
  symmetric_gaussian_deriv(p, m, n, &coords, jac);

  // evaluate objective function / negative log-likelihood
  double nll = poisson_nll(f, image, n);

  // evaluate gradient
  poisson_nll_grad(f, image, jac, m, n, grad);

  printf("At parameters : \n");
  for (int i = 0; i < m; i++) {
    printf("p[%d] = %.4f\n", i, p[i]);
  }
  printf("Log-likelihood = %.4f\n", nll);
  printf("Gradient \n");
  print_matrix(grad, 1, m, 4);

  // Evaluate the approximate Hessian
  poisson_nll_appx_hess(f, image, jac, m, n, hess);

  printf("\n");
  printf("Approximate Hessian:\n");
  print_matrix(hess, m, m, 4);
  printf("---- \nend program \n\n");

  // Evaluate the full Hessian
  printf("\n");
  printf("Full Hessian:\n");
  symmetric_gaussian_poisson_nll_hess(f, image, jac, p, &coords, m, n, hess);
  print_matrix(hess, m, m, 4);

  free_coord_data(&coords);
  free(image);
  free(f);
  free(jac);
  free(grad);
  free(hess);
}
