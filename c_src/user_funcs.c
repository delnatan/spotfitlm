#include "user_funcs.h"
#include <math.h>
#include <stdlib.h>

// note: remember to call `free_coord_data()`!
coord_data meshgrid2d(int boxsize) {
  int s = boxsize / 2;
  int coord_size = boxsize * boxsize;

  coord_data data;
  data.x = malloc(sizeof(double) * coord_size);
  data.y = malloc(sizeof(double) * coord_size);
  data.num_coords = coord_size;

  for (int i = 0; i < boxsize; i++) {
    for (int j = 0; j < boxsize; j++) {
      data.x[i * boxsize + j] = j - s;
      data.y[i * boxsize + j] = i - s;
    }
  }

  return data;
}

void free_coord_data(coord_data *data) {
  free(data->x);
  free(data->y);
}

void symmetric_gaussian(double *p, int m, int n, void *adata, double *f) {
  /*
    parameters, p
    m, number of parameters
    n, number of data points
    adata, coord_data
    f, output array

    5 parameters
   */
  double xc, yc, sigma, A, bg;

  // cast the generic pointer to `coord_data` pointer
  coord_data *coord = (coord_data *)adata;

  xc = p[0];
  yc = p[1];
  sigma = p[2];
  A = p[3];
  bg = p[4];

  double x_diff, y_diff, x_diff2, y_diff2, phi;

  for (int i = 0; i < n; i++) {
    x_diff = coord->x[i] - xc;
    y_diff = coord->y[i] - yc;
    x_diff2 = x_diff * x_diff;
    y_diff2 = y_diff * y_diff;
    phi = x_diff2 / (2 * sigma * sigma) + y_diff2 / (2 * sigma * sigma);
    f[i] = A * exp(-phi) + bg;
  }
}

void symmetric_gaussian_deriv(double *p, int m, int n, void *adata,
                              double *jac) {
  /*
  parameters, p
  m, number of parameters
  n, number of data points
  adata, coord_data
  jac, output jacobian n-by-m
 */
  double xc, yc, sigma, A;

  // cast the generic pointer to `coord_data` pointer
  coord_data *coord = (coord_data *)adata;

  xc = p[0];
  yc = p[1];
  sigma = p[2];
  A = p[3];

  double x_diff, y_diff, x_diff2, y_diff2, phi;

  for (int i = 0; i < n; i++) {
    x_diff = coord->x[i] - xc;
    y_diff = coord->y[i] - yc;
    x_diff2 = x_diff * x_diff;
    y_diff2 = y_diff * y_diff;
    phi = x_diff2 / (2 * sigma * sigma) + y_diff2 / (2 * sigma * sigma);

    // partial w.r.t xc
    jac[i * m] = A * x_diff * exp(-phi) / (sigma * sigma);
    // partial w.r.t yc
    jac[i * m + 1] = A * y_diff * exp(-phi) / (sigma * sigma);
    // partial w.r.t sigma
    jac[i * m + 2] =
        A * (x_diff2 + y_diff2) * exp(-phi) / (sigma * sigma * sigma);
    // partial w.r.t A
    jac[i * m + 3] = exp(-phi);
    // partial w.r.t bg
    jac[i * m + 4] = 1.0;
  }
}

void rotated_gaussian(double *p, int m, int n, void *adata, double *f) {
  /*
    Rotate Gaussian (2D) from wikipedia
    https://en.wikipedia.org/wiki/Gaussian_function

    f(x,y) = A * exp(
    -(a * (x - xc)^2 + 2b * (x - xc) * (y - yc) + c * (y - yc)^2))

    a = cos^2 (theta) / (2 * sigma**2) + sin^2 (theta)/(2* sigma**2);
    b = -sin(2*theta)

    7 parameters
   */
  double xc, yc, sigma_x, sigma_y, theta, A, bg;
  double xarg, yarg;
  double wrk;

  coord_data *coord = (coord_data *)adata;

  xc = p[0];
  yc = p[1];
  sigma_x = p[2];
  sigma_y = p[3];
  theta = p[4];
  A = p[5];
  bg = p[6];

  double a, b, c;
  double cost = cos(theta);
  double sint = sin(theta);
  double cost2 = cost * cost;
  double sint2 = sint * sint;

  a = cost2 / (2 * sigma_x * sigma_x) + sint2 / (2 * sigma_y * sigma_y);
  b = -sin(2 * theta) / (4 * sigma_x * sigma_x) +
      sin(2 * theta) / (4 * sigma_y * sigma_y);
  c = sint2 / (2 * sigma_x * sigma_x) + cost2 / (2 * sigma_y * sigma_y);

  for (int i = 0; i < n; i++) {
    xarg = coord->x[i] - xc;
    yarg = coord->y[i] - yc;
    wrk = a * xarg * xarg + 2 * b * xarg * yarg + c * yarg * yarg;
    f[i] = A * exp(-wrk) + bg;
  }
}

void rotated_gaussian_deriv(double *p, int m, int n, void *adata, double *jac) {
  double xc, yc, sigma_x, sigma_y, theta, A, bg;
  double xarg, yarg;
  double wrk;

  coord_data *coord = (coord_data *)adata;

  xc = p[0];
  yc = p[1];
  sigma_x = p[2];
  sigma_y = p[3];
  theta = p[4];
  A = p[5];
  bg = p[6];

  double a, b, c;
  double cost = cos(theta);
  double sint = sin(theta);
  double cost2 = cost * cost;
  double sint2 = sint * sint;

  a = cost2 / (2 * sigma_x * sigma_x) + sint2 / (2 * sigma_y * sigma_y);
  b = -sin(2 * theta) / (4 * sigma_x * sigma_x) +
      sin(2 * theta) / (4 * sigma_y * sigma_y);
  c = sint2 / (2 * sigma_x * sigma_x) + cost2 / (2 * sigma_y * sigma_y);

  for (int i = 0; i < n; i++) {
    xarg = coord->x[i] - xc;
    yarg = coord->y[i] - yc;
  }
}

void symmetric_gaussian_poisson_nll_hess(double *f, double *obs, double *jac,
                                         double *p, int m, int n, void *adata,
                                         double *hess) {

  double val, diag;

  int i, j, k;

  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      hess[i * m + j] = 0.0;
    }
  }

  // first compute the first-order terms
  // compute upper triangular part of J' * [w] * J
  for (i = 0; i < m; i++) {
    for (j = i; j < m; j++) {
      val = 0.0;
      // sum along the data dimension index, k
      for (k = 0; k < n; k++) {
        // diagonal elements
        diag = obs[k] / (f[k] * f[k]);
        val += jac[k * m + i] * diag * jac[k * m + j];
      }
      hess[i * m + j] = val;
    }
  }

  /* compute the second order terms
              2
  dS         d S
  --- . -------------
  df     dp_i * dp_j
   */

  // cast the generic pointer to `coord_data` pointer
  coord_data *coord = (coord_data *)adata;

  // parse the parameter vector
  double xc, yc, sigma, A, bg;
  xc = p[0];
  yc = p[1];
  sigma = p[2];
  A = p[3];
  bg = p[4];

  double s2 = sigma * sigma;
  double s3 = s2 * sigma;
  double s4 = s3 * sigma;
  double s5 = s4 * sigma;
  double s6 = s5 * sigma;

  double der1;

  double x_diff, y_diff, x_diff2, y_diff2, phi;

  for (i = 0; i < n; i++) {
    der1 = 1 - obs[i] / f[i]; // first-order term
    x_diff = coord->x[i] - xc;
    y_diff = coord->y[i] - yc;
    x_diff2 = x_diff * x_diff;
    y_diff2 = y_diff * y_diff;
    phi = x_diff2 / (2 * s2) + y_diff2 / (2 * s2);

    // hess[0,0]
    hess[0] += der1 * A * (-s2 + x_diff2) * exp(-phi) / s4;
    hess[1] += der1 * A * x_diff * y_diff * exp(-phi) / s4;
    hess[2] +=
        der1 * A * x_diff * (-2 * s2 + x_diff2 + y_diff2) * exp(-phi) / s5;
    hess[3] += der1 * x_diff * exp(-phi) / s2;
    hess[4] += 0.0;
    // hess[1,1]
    hess[6] += der1 * A * (-s2 + y_diff2) * exp(-phi) / s4;
    hess[7] +=
        der1 * A * y_diff2 * (-2 * s2 + x_diff2 + y_diff2) * exp(-phi) / s5;
    hess[8] += der1 * y_diff * exp(-phi) / s2;
    hess[9] += 0.0;
    // hess[2,2]
    hess[12] += der1 * A * (x_diff2 + y_diff2) * (-3 * s2 + x_diff2 + y_diff2) *
                exp(-phi) / s6;
    hess[13] += der1 * (x_diff2 + y_diff2) * exp(-phi) / s3;
    hess[14] += 0.0;
  }

  // mirror the upper triangular part to the lower triangular part
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      hess[i * m + j] = hess[j * m + i];
    }
  }
}
