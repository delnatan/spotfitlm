#ifndef GFIT_H
#define GFIT_H

void fit_symmetric_gaussian(double *image, int *ylocs, int *xlocs,
                            int img_height, int img_width, double sigma_init,
                            int nlocs, int boxsize, int itermax,
                            double *results);

#endif // GFIT_H
