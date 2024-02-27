#ifndef USER_FUNCS_H
#define USER_FUNCS_H

// declaration of `coord_data` structure
typedef struct {
  double *x;
  double *y;
  int num_coords;
} coord_data;

// function prototypes
coord_data meshgrid2d(int boxsize);

void free_coord_data(coord_data *data);

void symmetric_gaussian(double *p, int m, int n, void *adata, double *f);
void symmetric_gaussian_deriv(double *p, int m, int n, void *adata,
                              double *jac);
void symmetric_gaussian_poisson_nll_hess(double *f, double *obs, double *jac,
                                         double *p, int m, int n, void *adata,
                                         double *hess);

#endif // USER_FUNCS_H
