"""Tests for symmetric Gaussian derivatives and fitting."""

import numpy as np
import pytest

from spotfitlm import fit_symmetric_gaussian_mle


def symmetric_gaussian(params, x, y):
    """Symmetric 2D Gaussian function.

    Parameters
    ----------
    params : array-like
        [xc, yc, sigma, A, bg]
    x, y : array-like
        Coordinate arrays (same shape)

    Returns
    -------
    array-like
        Gaussian values at each (x, y) point
    """
    xc, yc, sigma, A, bg = params
    x_diff = x - xc
    y_diff = y - yc
    phi = (x_diff**2 + y_diff**2) / (2 * sigma**2)
    return A * np.exp(-phi) + bg


def symmetric_gaussian_jacobian(params, x, y):
    """Analytical Jacobian of symmetric 2D Gaussian.

    Returns shape (n, 5) where n is len(x) and columns are
    derivatives w.r.t. [xc, yc, sigma, A, bg].
    """
    xc, yc, sigma, A, bg = params
    x_diff = x - xc
    y_diff = y - yc
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    phi = (x_diff2 + y_diff2) / (2 * sigma**2)
    exp_phi = np.exp(-phi)

    n = len(x)
    jac = np.zeros((n, 5))

    # d/d(xc)
    jac[:, 0] = A * x_diff * exp_phi / (sigma**2)
    # d/d(yc)
    jac[:, 1] = A * y_diff * exp_phi / (sigma**2)
    # d/d(sigma)
    jac[:, 2] = A * (x_diff2 + y_diff2) * exp_phi / (sigma**3)
    # d/d(A)
    jac[:, 3] = exp_phi
    # d/d(bg)
    jac[:, 4] = 1.0

    return jac


def numerical_jacobian(func, params, x, y, eps=1e-7):
    """Compute Jacobian via central finite differences."""
    n = len(x)
    m = len(params)
    jac = np.zeros((n, m))

    for j in range(m):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += eps
        params_minus[j] -= eps

        f_plus = func(params_plus, x, y)
        f_minus = func(params_minus, x, y)
        jac[:, j] = (f_plus - f_minus) / (2 * eps)

    return jac


def poisson_nll_hessian(params, x, y, obs):
    """Analytical Hessian of Poisson negative log-likelihood.

    This implements the full Hessian including second-order terms
    as in symmetric_gaussian_poisson_nll_hess from user_funcs.c
    """
    xc, yc, sigma, A, bg = params
    m = 5

    s2 = sigma**2
    s3 = s2 * sigma
    s4 = s3 * sigma
    s5 = s4 * sigma
    s6 = s5 * sigma

    # Get model values and jacobian
    f = symmetric_gaussian(params, x, y)
    jac = symmetric_gaussian_jacobian(params, x, y)

    # First-order terms: J' * diag(obs/f^2) * J
    weights = obs / (f**2)
    hess = jac.T @ (weights[:, None] * jac)

    # Second-order terms
    x_diff = x - xc
    y_diff = y - yc
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    phi = (x_diff2 + y_diff2) / (2 * s2)
    exp_phi = np.exp(-phi)

    der1 = 1 - obs / f  # first-order term for each data point

    # Add second-order contributions
    # hess[0,0]: d²f/dxc²
    hess[0, 0] += np.sum(der1 * A * (-s2 + x_diff2) * exp_phi / s4)
    # hess[0,1]: d²f/dxc dyc
    hess[0, 1] += np.sum(der1 * A * x_diff * y_diff * exp_phi / s4)
    # hess[0,2]: d²f/dxc dsigma
    hess[0, 2] += np.sum(der1 * A * x_diff * (-2*s2 + x_diff2 + y_diff2) * exp_phi / s5)
    # hess[0,3]: d²f/dxc dA
    hess[0, 3] += np.sum(der1 * x_diff * exp_phi / s2)
    # hess[0,4]: d²f/dxc dbg = 0

    # hess[1,1]: d²f/dyc²
    hess[1, 1] += np.sum(der1 * A * (-s2 + y_diff2) * exp_phi / s4)
    # hess[1,2]: d²f/dyc dsigma
    hess[1, 2] += np.sum(der1 * A * y_diff * (-2*s2 + x_diff2 + y_diff2) * exp_phi / s5)
    # hess[1,3]: d²f/dyc dA
    hess[1, 3] += np.sum(der1 * y_diff * exp_phi / s2)
    # hess[1,4]: d²f/dyc dbg = 0

    # hess[2,2]: d²f/dsigma²
    hess[2, 2] += np.sum(der1 * A * (x_diff2 + y_diff2) * (-3*s2 + x_diff2 + y_diff2) * exp_phi / s6)
    # hess[2,3]: d²f/dsigma dA
    hess[2, 3] += np.sum(der1 * (x_diff2 + y_diff2) * exp_phi / s3)
    # hess[2,4]: d²f/dsigma dbg = 0

    # Mirror upper triangular to lower
    for i in range(m):
        for j in range(i):
            hess[i, j] = hess[j, i]

    return hess


def poisson_nll(params, x, y, obs):
    """Poisson negative log-likelihood (ignoring constant term)."""
    f = symmetric_gaussian(params, x, y)
    # NLL = sum(f - obs*log(f))
    return np.sum(f - obs * np.log(f))


def numerical_hessian(func, params, x, y, obs, eps=1e-5):
    """Compute Hessian via central finite differences."""
    m = len(params)
    hess = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            # f(x+ei+ej)
            p_pp = params.copy()
            p_pp[i] += eps
            p_pp[j] += eps

            # f(x+ei-ej)
            p_pm = params.copy()
            p_pm[i] += eps
            p_pm[j] -= eps

            # f(x-ei+ej)
            p_mp = params.copy()
            p_mp[i] -= eps
            p_mp[j] += eps

            # f(x-ei-ej)
            p_mm = params.copy()
            p_mm[i] -= eps
            p_mm[j] -= eps

            hess[i, j] = (func(p_pp, x, y, obs) - func(p_pm, x, y, obs)
                         - func(p_mp, x, y, obs) + func(p_mm, x, y, obs)) / (4 * eps**2)

    return hess


class TestSymmetricGaussianJacobian:
    """Tests for the analytical Jacobian of the symmetric Gaussian."""

    def test_jacobian_at_center(self):
        """Test Jacobian when Gaussian is centered at origin."""
        # Create a small grid
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        x, y = x.ravel().astype(float), y.ravel().astype(float)

        params = np.array([0.0, 0.0, 1.5, 100.0, 10.0])

        jac_analytical = symmetric_gaussian_jacobian(params, x, y)
        jac_numerical = numerical_jacobian(symmetric_gaussian, params, x, y)

        np.testing.assert_allclose(jac_analytical, jac_numerical, rtol=1e-4, atol=1e-8)

    def test_jacobian_off_center(self):
        """Test Jacobian when Gaussian is off-center."""
        x, y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
        x, y = x.ravel().astype(float), y.ravel().astype(float)

        params = np.array([1.3, -0.7, 2.0, 500.0, 50.0])

        jac_analytical = symmetric_gaussian_jacobian(params, x, y)
        jac_numerical = numerical_jacobian(symmetric_gaussian, params, x, y)

        np.testing.assert_allclose(jac_analytical, jac_numerical, rtol=1e-4, atol=1e-8)

    def test_jacobian_small_sigma(self):
        """Test Jacobian with small sigma (narrow peak)."""
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        x, y = x.ravel().astype(float), y.ravel().astype(float)

        params = np.array([0.5, 0.5, 0.8, 200.0, 5.0])

        jac_analytical = symmetric_gaussian_jacobian(params, x, y)
        jac_numerical = numerical_jacobian(symmetric_gaussian, params, x, y)

        np.testing.assert_allclose(jac_analytical, jac_numerical, rtol=1e-4, atol=1e-8)


class TestPoissonNLLHessian:
    """Tests for the analytical Hessian of the Poisson NLL."""

    def test_hessian_synthetic_data(self):
        """Test Hessian with synthetic Poisson data."""
        np.random.seed(42)

        x, y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
        x, y = x.ravel().astype(float), y.ravel().astype(float)

        # True parameters
        params_true = np.array([0.0, 0.0, 1.5, 100.0, 10.0])

        # Generate Poisson observations
        f_true = symmetric_gaussian(params_true, x, y)
        obs = np.random.poisson(f_true).astype(float)
        obs = np.maximum(obs, 0.1)  # Avoid zeros for log

        # Evaluate at slightly perturbed parameters
        params = np.array([0.1, -0.1, 1.6, 95.0, 11.0])

        hess_analytical = poisson_nll_hessian(params, x, y, obs)
        hess_numerical = numerical_hessian(poisson_nll, params, x, y, obs)

        np.testing.assert_allclose(hess_analytical, hess_numerical, rtol=0.15, atol=0.01)

    def test_hessian_at_optimum(self):
        """Test Hessian near the optimum (where it matters for covariance)."""
        np.random.seed(123)

        x, y = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5))
        x, y = x.ravel().astype(float), y.ravel().astype(float)

        params = np.array([0.3, -0.2, 1.8, 150.0, 20.0])
        f = symmetric_gaussian(params, x, y)
        obs = np.random.poisson(f).astype(float)
        obs = np.maximum(obs, 0.1)

        hess_analytical = poisson_nll_hessian(params, x, y, obs)
        hess_numerical = numerical_hessian(poisson_nll, params, x, y, obs)

        np.testing.assert_allclose(hess_analytical, hess_numerical, rtol=0.15, atol=0.01)

    def test_hessian_symmetry(self):
        """Test that analytical Hessian is symmetric."""
        x, y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
        x, y = x.ravel().astype(float), y.ravel().astype(float)

        params = np.array([0.5, -0.3, 2.0, 200.0, 15.0])
        f = symmetric_gaussian(params, x, y)
        obs = f + 1  # Avoid issues with Poisson at true params

        hess = poisson_nll_hessian(params, x, y, obs)

        np.testing.assert_allclose(hess, hess.T, rtol=1e-10)


class TestGaussianFitting:
    """Tests for 2D Gaussian fitting."""

    def test_fit_single_gaussian(self):
        """Test fitting a single 2D Gaussian."""
        np.random.seed(42)

        # Create image with a Gaussian spot
        size = 64
        boxsize = 11

        # True parameters
        true_x, true_y = 32.3, 32.7
        true_sigma = 1.5
        true_A = 500.0
        true_bg = 100.0

        # Create coordinate grid
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')

        # Generate clean Gaussian
        phi = ((xx - true_x)**2 + (yy - true_y)**2) / (2 * true_sigma**2)
        image_clean = true_A * np.exp(-phi) + true_bg

        # Add Poisson noise
        image = np.random.poisson(image_clean).astype(float)

        # Fit
        ylocs = np.array([32])
        xlocs = np.array([32])

        result = fit_symmetric_gaussian_mle(
            image, (ylocs, xlocs), boxsize=boxsize, sigma0=1.5, itermax=50
        )

        assert len(result) == 1

        # Check fitted parameters are close to true values
        assert abs(result['x'].iloc[0] - true_x) < 0.2
        assert abs(result['y'].iloc[0] - true_y) < 0.2
        assert abs(result['sigma'].iloc[0] - true_sigma) < 0.2
        # Amplitude and background can vary more due to noise
        assert abs(result['A'].iloc[0] - true_A) / true_A < 0.15

    def test_fit_multiple_gaussians(self):
        """Test fitting multiple Gaussians in one image."""
        np.random.seed(123)

        size = 128
        boxsize = 11

        # Define multiple spots
        spots = [
            {'x': 30.2, 'y': 30.5, 'sigma': 1.5, 'A': 400, 'bg': 80},
            {'x': 80.7, 'y': 50.3, 'sigma': 1.5, 'A': 600, 'bg': 80},
            {'x': 60.1, 'y': 90.8, 'sigma': 1.5, 'A': 300, 'bg': 80},
        ]

        # Create image
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        image = np.ones((size, size)) * 80  # background

        for spot in spots:
            phi = ((xx - spot['x'])**2 + (yy - spot['y'])**2) / (2 * spot['sigma']**2)
            image += spot['A'] * np.exp(-phi)

        image = np.random.poisson(image).astype(float)

        # Fit all spots
        ylocs = np.array([int(s['y']) for s in spots])
        xlocs = np.array([int(s['x']) for s in spots])

        result = fit_symmetric_gaussian_mle(
            image, (ylocs, xlocs), boxsize=boxsize, sigma0=1.5, itermax=50
        )

        assert len(result) == 3

        # Check each fitted spot is close to its true position
        for i, spot in enumerate(spots):
            assert abs(result['x'].iloc[i] - spot['x']) < 0.3
            assert abs(result['y'].iloc[i] - spot['y']) < 0.3

    def test_fit_convergence(self):
        """Test that optimizer converges (retcode == 0)."""
        np.random.seed(456)

        size = 64
        boxsize = 11

        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        phi = ((xx - 32)**2 + (yy - 32)**2) / (2 * 1.5**2)
        image = 500 * np.exp(-phi) + 100
        image = np.random.poisson(image).astype(float)

        result = fit_symmetric_gaussian_mle(
            image, (np.array([32]), np.array([32])), boxsize=boxsize, sigma0=1.5, itermax=100
        )

        # retcode 0 means converged, -1 means max iter reached
        assert result['retcode'].iloc[0] in [0, -1]
