#include <float.h>
// convenient macros
#define CLIP(x, lower, upper) (fmin(upper, fmax(x, lower)))

// algorithm parameters
#define EPS 1E-12
#define MU_LB 1E-8
#define MU_UB 1E8
#define ACCEPT 3.0
#define REJECT 2.0
#define GTOL 1e-5

/*
  Damped Gauss-Newton optimizer for function non-linear function `func`

  @param func Function to optimize (minimize). Typically a negative log-
  likelihood function.
  @param fgrad Function that computes the gradient to the objective function
  @param fhess Function that computes the hessian to the objective function
  @param model Model function
  @param modeljac Jacobian of the model function

*/

typedef struct {
  int num_iterations;
  double objective_value;
  double scaled_grad_norm;
  int return_code;
} OptimizerResult;

OptimizerResult
dglm_der(double (*func)(double *f, double *obs, int n),
         void (*funcgrad)(double *f, double *obs, double *jac, int m, int n,
                          double *grad),
         void (*funcappxhess)(double *f, double *obs, double *jac, int m, int n,
                              double *hess),
         void (*model)(double *p, int m, int n, void *adata, double *f),
         void (*fjac)(double *p, int m, int n, void *adata, double *jac),
         void (*fhess)(double *f, double *obs, double *jac, double *p, int m,
                       int n, void *adata, double *hess),
         double *p, double *obs, int m, int n, int itmax, double *covar,
         void *adata, int verbose);
