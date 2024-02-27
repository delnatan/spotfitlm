void matrix_inverse_from_cholesky(double *L, int n, double *result);
int solve_Axb_cholesky(double *A, double *mu, double *b, int n, double *x);
int cholesky_decomposition(double *A, int n);
/* int call_dsyev(double *B, int n, double *evec); */
/* int matinv(double *B, int n); */
void print_matrix(double *A, int m, int n, int truncate);
void print_lower_triangular_matrix(double *A, int n, int truncate);
