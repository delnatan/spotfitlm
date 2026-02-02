# spotfitlm

A Python library for robust spot detection in 2D images using MLE Gaussian fitting. The core fitting algorithm is implemented in C using a Levenberg-Marquardt optimizer with Poisson noise model. Parameter uncertainties are computed from the full Hessian matrix.

See [DETAILS.md](DETAILS.md) for algorithm documentation.

## Features

- Maximum likelihood estimation with Poisson noise model
- Sub-pixel localization of point sources
- Covariance-based uncertainty estimates
- Spot detection using statistical hypothesis testing

## Installation

```bash
pip install spotfitlm
```

Pre-built wheels are available for:
- **Python**: 3.9 - 3.13
- **Platforms**: Windows (x64), macOS (Intel & Apple Silicon), Linux (x64, arm64)

## Usage

```python
import numpy as np
from spotfitlm import find_spots_in_timelapse

# Load your image (2D + time numpy array)
image = ...
# and mask (2D binary numpy array) (optional)
mask = ... 

# Detect and fit spots
results = find_spots_in_timelapse(
    image,
    mask,
    sigma=1.5,      # expected PSF width
    boxsize=9,      # fitting ROI size
    alpha=0.05,     # significance level for detection
    use_filter=True,
    min_sigma=0.8,
    max_sigma=2.4,
    min_amplitude=5.0,
    max_amplitude=800,
)

# Results is a DataFrame with columns:
# amplitude, background, x, y, x_err, y_err, sigma, ...
```

## References

- Laurence, T.A. & Chromy, B.A. (2010). Efficient maximum likelihood estimator fitting of histograms. *Nature Methods*, 7(5), 338-339.
- Aguet, F. et al. (2013). Advances in analysis of low signal-to-noise images link dynamin and AP2 to the functions of an endocytic checkpoint. *Developmental Cell*, 26(3), 279-291.
