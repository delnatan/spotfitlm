#ifndef GFIT_H
#define GFIT_H

#ifdef _WIN32
#define SPOTFITLM_API __declspec(dllexport)
#else
#define SPOTFITLM_API
#endif

SPOTFITLM_API void fit_symmetric_gaussian(double *image, int *ylocs, int *xlocs,
                            int img_height, int img_width, double sigma_init,
                            int nlocs, int boxsize, int itermax,
                            double *results);

#endif // GFIT_H
