#include "user_funcs.h"
#include <float.h>
#include <math.h>

#define EPS 1e-7

/*
  Negative log-likelihood of normalized Poisson distribution

  -logP(obs | f) = sum_i^N f[i] - obs[i] - obs[i] * log(f[i] / obs[i]);

 */
double poisson_nll(double *f, double *obs, int n) {
  double val = 0.0;

  for (int i = 0; i < n; i++) {
    double sf = (f[i] > 0) ? f[i] : EPS;
    double sd = (obs[i] > 0) ? obs[i] : EPS;
    val += sf - sd - sd * log(sf / sd);
  }

  return val;
}

/*
  gradient of poisson negative log-likelihood

  dS   dS   df     t  /     D \
  -- = -- * --  = J . | 1 - - |
  dp   df   dp        \     f /


  m-by-n x n-by-1 := m-by-1

  f, input
  obs, input
  jac, n-by-m input
  grad, output, vector of length `m`

 */
void poisson_nll_grad(double *f, double *obs, double *jac, int m, int n,
                      double *grad) {

  int i, j;

  for (i = 0; i < m; i++) {
    grad[i] = 0.0;
  }

  // do J' * dS/df
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      grad[i] += jac[j * m + i] * (1.0 - obs[j] / f[j]);
    }
  }
}

/*
 approximate hessian of poisson negative log-likelihood

 d^2 S   d^2 S    df_i    df_i   dS    d^2 f_k
 ----- = ------ . ----- . ---- + -- . ---------
 dp^2    df_i^2   dp_j    dp_k   df   dp_j dp_k

 The first term is just J' * [w] * J
 where [w] is a diagonal matrix whose elements are:

  2
 d S     D
 --- := ---
   2      2
 df      f


 The second term is ignored in the approximation
 output, m-by-m matrix

 */
void poisson_nll_appx_hess(double *f, double *obs, double *jac, int m, int n,
                           double *hess) {

  double val, diag;

  int i, j, k;

  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      hess[i * m + j] = 0.0;
    }
  }

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

  // mirror the upper triangular part to the lower triangular part
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      hess[i * m + j] = hess[j * m + i];
    }
  }
}
