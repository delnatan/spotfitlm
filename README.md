# spotfitlm

A small Python library for doing robust spot detection in 2D by (MLE) Gaussian fitting. The fitting is done through a small C library implementing a damped Gauss-Newton optimization algorithm (like Levenberg-Marquardt) algorithm. Error estimates (covariance) matrix of the fit parameters are computed from the (inverse of) full Hessian matrix. Currently, only a symmetric Gaussian fit is implemented.

For spot detection, this package uses the algorithm from Danuser Lab's U-track MATLAB software. Specifically, the code from:
[https://github.com/DanuserLab/u-track3D/blob/9279b3784de64d29bb06c3693e99f2e5c064288e/software/pointSourceDetection.m#L111]

for doing the hypothesis testing on *significant* Gaussian peaks above background noise.

# Installation

This package is not yet released at PyPI so it can't be installed by a simple `pip install`, so you'll need to use `git` to install it to your Python environment:

1) Clone this repository
```bash
git clone https://github.com/delnatan/spotfitlm.git
```

2) and go into the directory `spotfitlm` and run:
```bash
pip install -e .
```

