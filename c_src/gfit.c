#include "gfit.h"
#include "glm_core.h"
#include "objective_funcs.h"
#include "user_funcs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void fit_symmetric_gaussian(double *image, int *ylocs, int *xlocs,
                            int img_height, int img_width, double sigma_init,
                            int nlocs, int boxsize, int itermax,
                            double *results) {
  /*
    Perform Gaussian fitting to localize to single-molecule intensities
    using MLE.
   */
  double roi_min_val, roi_max_val, pixval;

  double sigma0 = sigma_init;

  int s = boxsize / 2;

  // `roi` is the data (subregion)
  double *roi = malloc(boxsize * boxsize * sizeof(double));

  int npars = 5;
  int nresults = 12;
  int ndata = boxsize * boxsize;

  // preallocate covariance matrix
  double *covar = malloc(npars * npars * sizeof(double));

  // define box coordinates
  coord_data coords = meshgrid2d(boxsize);

  // loop through each maxima
  for (int n = 0; n < nlocs; n++) {
    // extract ROI from image
    int yc = ylocs[n];
    int xc = xlocs[n];

    // copy pixels of ROI into 'roi'
    for (int i = -s; i <= s; i++) {
      for (int j = -s; j <= s; j++) {
        pixval = image[(yc + i) * img_width + (xc + j)];
        roi[(i + s) * boxsize + (j + s)] = pixval;
      }
    }

    // compute initial estimates
    // Amplitude and background is initially max(roi) and min(roi)
    roi_min_val = roi[0];
    roi_max_val = roi[0];

    for (int i = 0; i < boxsize; i++) {
      for (int j = 0; j < boxsize; j++) {
        pixval = roi[i * boxsize + j];
        if (pixval < roi_min_val)
          roi_min_val = pixval;
        if (pixval > roi_max_val)
          roi_max_val = pixval;
      }
    }

    // double weight = 0.0;
    // double weighted_sum_x = 0.0;
    // double weighted_sum_y = 0.0;

    // // compute first moment to estimate x,y position
    // for (int i = 0; i < coords.num_coords; i++) {
    //   weight += (roi[i] - roi_min_val);
    //   weighted_sum_x += coords.x[i] * (roi[i] - roi_min_val);
    //   weighted_sum_y += coords.y[i] * (roi[i] - roi_min_val);
    // }
    // double xc0 = weighted_sum_x / weight;
    // double yc0 = weighted_sum_y / weight;

    // at the moment start with the same parameters for all
    double pars[5] = {0.0, 0.0, sigma0, roi_max_val - roi_min_val, roi_min_val};

    OptimizerResult optres;

    optres = dglm_der(&poisson_nll, &poisson_nll_grad, &poisson_nll_appx_hess,
                      &symmetric_gaussian, &symmetric_gaussian_deriv,
                      &symmetric_gaussian_poisson_nll_hess, pars, roi, npars,
                      ndata, itermax, covar, &coords, 0);

    // populate results
    results[n * nresults] = pars[3];                           // Amplitude
    results[n * nresults + 1] = pars[4];                       // background
    results[n * nresults + 2] = (double)xc;                    // xc_init
    results[n * nresults + 3] = (double)yc;                    // yc_init
    results[n * nresults + 4] = ((double)xc) + pars[0];        // xc
    results[n * nresults + 5] = ((double)yc) + pars[1];        // yc
    results[n * nresults + 6] = sqrt(covar[0]);                // var_xc
    results[n * nresults + 7] = sqrt(covar[npars * 1 + 1]);    // var_yc
    results[n * nresults + 8] = pars[2];                       // sigma
    results[n * nresults + 9] = (double)optres.num_iterations; // niter
    results[n * nresults + 10] = optres.objective_value;       // norm2 of error
    results[n * nresults + 11] = (double)optres.return_code;   // return code
  }

  free(roi);
  free(covar);
  free_coord_data(&coords);
}
