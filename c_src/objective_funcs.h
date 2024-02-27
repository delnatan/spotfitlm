
double poisson_nll(double *f, double *obs, int n);
void poisson_nll_grad(double *f, double *obs, double *jac, int m, int n,
                      double *grad);
void poisson_nll_appx_hess(double *f, double *obs, double *jac, int m, int n,
                           double *hess);
