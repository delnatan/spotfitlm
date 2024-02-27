import numpy as np
import pandas as pd

from .libloader import load_library

__lib__ = load_library()


def fit_symmetric_gaussian_mle(
    image: np.ndarray,
    yxlocs: tuple,
    boxsize: int,
    sigma0: float,
    itermax: int,
) -> pd.DataFrame:
    """
    Perform maximum likelihood estimation (MLE) fitting of a Gaussian function
    on an image.

    The function calls a C library function that performs the MLE fitting, and
    returns the results in a pandas DataFrame.

    Parameters
    ----------
    image : np.ndarray
        The image data as a 2D numpy array.
    yxlocs : tuple of np.ndarray
        A tuple of two 1D numpy arrays containing the y and x coordinates of the
    peaks to fit, respectively.
    boxsize : int
        The size of the box to extract around each peak for fitting.
    itermax : int
        The maximum number of iterations for the MLE fitting algorithm.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the MLE fitting results. Each row corresponds to
        one peak, and the columns are:
        'A' - Amplitude of the Gaussian.
        'bg' - Background offset.
        'x' and 'y' - Coordinates of the Gaussian's peak.
        'sigma' - Standard deviations of the Gaussian.
        'niter' - The number of iterations performed by the fitting algorithm.
        'norm2_error' - The squared error norm of the fit.
    """

    image = np.ascontiguousarray(image.astype(np.double))
    ylocs = np.ascontiguousarray(yxlocs[0].astype(np.intc))
    xlocs = np.ascontiguousarray(yxlocs[1].astype(np.intc))
    npeaks = len(ylocs)

    # preallocate results
    fitres = np.empty((npeaks, 12), dtype=np.double)

    # call C library
    __lib__.fit_symmetric_gaussian(
        image,
        ylocs,
        xlocs,
        image.shape[0],
        image.shape[1],
        sigma0,
        npeaks,
        boxsize,
        itermax,
        fitres,
    )

    dfres = pd.DataFrame(
        fitres,
        columns=[
            "A",
            "bg",
            "x0",
            "y0",
            "x",
            "y",
            "x_std",
            "y_std",
            "sigma",
            "niter",
            "neglogL",
            "retcode",
        ],
    )
    dfres["x0"] = dfres["x0"].astype(int)
    dfres["y0"] = dfres["y0"].astype(int)
    dfres["niter"] = dfres["niter"].astype(int)
    dfres["retcode"] = dfres["retcode"].astype(int)

    # throw away molecules that are out of bounds
    s = boxsize // 2
    within_bounds = (
        (dfres["x"] < image.shape[1] - s)
        & (dfres["y"] < image.shape[0] - s)
        & (dfres["x"] > s)
        & (dfres["y"] > s)
    )
    dfres = dfres.loc[within_bounds]

    return dfres
