"""
Utility functions for GEM analysis

PointSourceDetector2D is a translation from the paper:
Advances in analysis of low signal-to-noise images link dynamin and AP2.
by Francois Aguet, Costin Antonescu, Marcel Mettlen, Sandra Schmid and
Gaudenz Danuser.

I translated the code from MATLAB from:
https://github.com/DanuserLab/u-track3D/blob/master/software/pointSourceDetection.m

"""

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.stats as stats
from tqdm.auto import tqdm

from .fitters import fit_symmetric_gaussian_mle


class PointSourceDetector2D:
    def __init__(self, sigma, truncate=4):
        self.sigma = sigma
        self.truncate = int(truncate)

        w = int(np.ceil(self.truncate * self.sigma))
        x = np.arange(-w, w + 1)
        y, x = np.meshgrid(x, x, indexing="ij")

        # compute gaussian kernel
        arg = x**2 / (2.0 * self.sigma**2) + y**2 / (2.0 * self.sigma**2)
        self._g = np.exp(-arg)
        self.kernel_size = self._g.size

        # and box/sum filter
        self._u = np.ones(self._g.shape)
        self._gsum = self._g.sum()
        self._g2sum = (self._g * self._g).sum()

        J = np.vstack([self._g.ravel(), np.ones(self.kernel_size)]).T
        self.C = np.linalg.inv(J.T @ J)

    def detect_significant_pixels(self, image, alpha=0.05):
        if np.issubdtype(image.dtype, np.floating):
            f = image
        else:
            # ensure image is in 'float' format
            f = image.astype(float)

        # compute quantities
        fg = ndi.convolve(f, self._g)
        fu = ndi.convolve(f, self._u)
        fu2 = ndi.convolve(f * f, self._u)
        n = float(self.kernel_size)

        A_est = (fg - self._gsum * fu / n) / (self._g2sum - self._gsum**2 / n)
        c_est = (fu - self._gsum * A_est) / n

        f_c = fu2 - 2 * c_est * fu + n * c_est**2
        RSS = (
            A_est**2 * self._g2sum
            - 2 * A_est * (fg - c_est * self._gsum)
            + f_c
        )

        RSS[RSS < 0] = 0.0

        sigma_e2 = RSS / (n - 3)

        sigma_A = np.sqrt(sigma_e2 * self.C[0, 0])
        sigma_res = np.sqrt(RSS / (n - 1))
        # ppf is the inverse CDF
        kLevel = stats.norm.ppf(1 - alpha / 2.0, 0, 1)
        SE_sigma_c = sigma_res / np.sqrt(2 * (n - 1)) * kLevel
        df2 = (
            (n - 1)
            * (sigma_A**2 + SE_sigma_c**2) ** 2
            / (sigma_A**4 + SE_sigma_c**4)
        )

        # this is the degree of freedom
        scomb = np.sqrt((sigma_A**2 + SE_sigma_c**2) / n)
        # we are comparing two 'means': amplitude vs. k * noise
        T = (A_est - sigma_res * kLevel) / scomb

        # one sided, two-sample t-test to see if amplitude is higher than
        # noise threshold
        # originally called `stats.t.cdf(-T, df2)`
        pval = stats.t.sf(T, df2)

        return pval < alpha

    def detect_spots(self, image, significance=0.05):
        if np.issubdtype(image.dtype, np.floating):
            f = image
        else:
            f = image.astype(float)

        img_log = -ndi.gaussian_laplace(f, self.sigma)
        domsize = int(2 * np.ceil(self.sigma) + 1)
        domain = np.ones((domsize, domsize))
        ordmax_img = ndi.maximum_filter(img_log, footprint=domain)
        locmax_log = img_log == ordmax_img
        # remove border pixels
        locmax_log[:domsize, :] = 0
        locmax_log[-domsize:, :] = 0
        locmax_log[:, :domsize] = 0
        locmax_log[:, -domsize:] = 0
        mask = self.detect_significant_pixels(f, alpha=significance)
        yxlocs = np.where(mask * locmax_log)
        return yxlocs


def find_spots_in_timelapse(
    image,
    sigma=1.5,
    significance=0.05,
    boxsize=9,
    itermax=50,
    mask=None,
    progress=True,
):
    Nt, Ny, Nx = image.shape
    df_list = []

    d = PointSourceDetector2D(sigma=sigma)

    for t in tqdm(range(Nt), disable=not progress):
        img = image[t]
        yxlocs = d.detect_spots(img, significance=significance)
        if mask is not None:
            valid_spots = mask[yxlocs] > 0
            yxlocs = (yxlocs[0][valid_spots], yxlocs[1][valid_spots])
        _df = fit_symmetric_gaussian_mle(img, yxlocs, boxsize, sigma, itermax)
        _df["frame"] = t
        # compute the combined localization error
        _df["xy_std"] = (_df["x_std"] ** 2 + _df["y_std"] ** 2) ** 0.5
        df_list.append(_df)

    return pd.concat(df_list, axis=0, ignore_index=True)
