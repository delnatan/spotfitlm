#include "glm_core.h"
#include "matrix_operations.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
  Damped Gauss-Newton method with 'analytical' gradient and hessian
  @param, func(), objective function
  @param, fgrad(), gradient function
  @param, fappxhess(), approximate hessian function
  @param, fjac(), jacobian of model function
  @param, fhess(), full hessian, if NULL, approximate is used instead
  @param, p, vector of parameters
  @param, obs, vector of observations/data
  @param, m, number of parameters
  @param, n, number of measurements
  @param, itmax, maximum number of iterations
  @param, covar, covariance matrix output
  @param, adata, additional data for the function
  @param, verbose, flag for printing iteration info. >0 is True, 0 is False.

  Algorithm details:
  TO DO

  Return codes:
    0   successful termination |gDg| < GTOL.
        D is the inverse of the diagonal of H to scale the gradient
    -1  maximum iteration reached
    -2  Hessian is not positive definite

*/

OptimizerResult
dglm_der(double (*func)(double *f, double *obs, int n),
         void (*funcgrad)(double *f, double *obs, double *jac, int m, int n,
                          double *grad),
         void (*funcappxhess)(double *p, double *obs, double *jac, int m, int n,
                              double *hess),
         void (*model)(double *p, int m, int n, void *adata, double *f),
         void (*fjac)(double *p, int m, int n, void *adata, double *jac),
         void (*fhess)(double *f, double *obs, double *jac, double *p, int m,
                       int n, void *adata,
                       double *hess), // can be NULL
         double *p,                   // parameters
         double *obs,                 // measurement
         int m,                       // number of parameters
         int n,                       // number of measurements
         int itmax,                   // maximum number of iterations
         double *covar,               // covariance matrix (out)
         void *adata, // additional arguments for function `func`
         int verbose  // flag for printing; 0 = silent, >1 verbose
) {

  OptimizerResult result;
  double mu = 1E-4; // damping parameter
  int i, j, it;
  int singular;
  int ret;
  int nfeval = 0;
  int mu_iter = 0;
  int mu_iter_max = 30;
  double s, s_trial, dpmag;
  double actual_reduction, predicted_reduction, ap_ratio;

  // preallocate workspace
  double *f = malloc(n * sizeof(double));
  double *jac = malloc(n * m * sizeof(double)); // model jacobian
  double *g = malloc(m * sizeof(double));       // gradient vector
  double *H = malloc(m * m * sizeof(double));   // hessian matrix
  double *dp = malloc(m * sizeof(double));      // parameter change
  double *p_trial = malloc(m * sizeof(double)); // trial parameters

  // default flag is to indicate max iter. has been reached
  // this flag is changed on alternate termination condition
  ret = -1;

  // begin iterations
  for (it = 0; it < itmax; it++) {
    // compute model, `f` contains model
    (*model)(p, m, n, adata, f);
    // compute jacobian, `jac` contains jacobian
    (*fjac)(p, m, n, adata, jac);
    // compute objective function, score s
    s = (*func)(f, obs, n);
    nfeval += 1;

    // compute gradient, stored in 'g'
    (*funcgrad)(f, obs, jac, m, n, g);
    // compute approximate hessian (for Gauss-Newton update)
    (*funcappxhess)(f, obs, jac, m, n, H);

    // compute 'length' of dp
    // |dp| = g'.diag(H).g
    dpmag = 0;
    for (i = 0; i < m; ++i) {
      dpmag += g[i] * g[i] / H[i * m + i];
    }

    if (dpmag < GTOL) {
      ret = 0;
      break;
    }
    // initialize solve loop
    singular = 1;
    mu_iter = 0;
    ap_ratio = 1.0;

    /* printf("Iteration %d, |Dp| = %.4E, s=%10.4E, mu=%10.4E, nfeval=%d\n", it,
     */
    /* dpmag, s, mu, nfeval); */

    // take a step & keep going until either conditions are met
    while ((singular || ap_ratio < 0.25) && (it < itmax) &&
           (mu_iter < mu_iter_max)) {
      mu_iter += 1;
      // solve for parameter change with damping
      // note that this solves H * dp = g
      singular = solve_Axb_cholesky(H, &mu, g, m, dp);
      // update trial parameters
      // dp = -H^-1 * g, so we do p = p - dp;
      for (i = 0; i < m; i++) {
        p_trial[i] = p[i] - dp[i];
      }

      // compute objective at new parameters
      (*model)(p_trial, m, n, adata, f);
      s_trial = (*func)(f, obs, n);

      // re-use the 'singular' flag to check for obj. function value decrease
      // actual reduction is f(x) - f(x+h)
      actual_reduction = s - s_trial;
      // if f(x) < f(x+h), then we we actually 'increased' f and >0
      singular = actual_reduction < 0;

      // compute predicted reduction
      predicted_reduction = 0.0;

      // predicted reduction is dp'*g + 0.5 * dp' * H * dp;
      for (i = 0; i < m; i++) {
        predicted_reduction += dp[i] * g[i];
      }

      for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
          predicted_reduction += 0.5 * dp[i] * H[i * m + j] * dp[j];
        }
      }

      ap_ratio = actual_reduction / predicted_reduction;

      // if no decrease, increase damping and repeat step
      if (singular || ap_ratio < 0.25) {
        mu *= REJECT;
      } else {
        // otherwise, decrease damping for next iteration
        mu /= ACCEPT;
      }

      // clip values to prevent division by small/large number
      mu = CLIP(mu, MU_LB, MU_UB);

      /* printf("\t 'mu' iteration %2d, mu = %10.6E, rho = %7.4f\n", mu_iter,
       * mu, */
      /* ap_ratio); */
    }

    // update parameters
    for (i = 0; i < m; i++) {
      p[i] -= dp[i];
    }
  }
  // calculate covariance matrix
  if (covar) {
    // calculate analytical hessian
    if (fhess) {
      (*fhess)(f, obs, jac, p, m, n, adata, H);
      // otherwise, use the approximation
    } else {
      (*funcappxhess)(f, obs, jac, m, n, H);
    }
    // calculate covariance matrix from inverse of Hessian
    singular = cholesky_decomposition(H, m);
    // H is now L
    if (singular >= 0) {
      matrix_inverse_from_cholesky(H, m, covar);
      ret = 0; // successful iteration
    } else {
      ret = -2; // Hessian is not positive definite
    }
  }

  // assign result
  result.num_iterations = it;
  result.objective_value = s;
  result.scaled_grad_norm = dpmag;
  result.return_code = ret;

  // memory clean-up
  free(f);
  free(jac);
  free(g);
  free(H);
  free(dp);
  free(p_trial);

  return result;
}
