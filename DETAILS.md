# Algorithm Details

This document describes the Levenberg-Marquardt optimization algorithm used for 2D Gaussian fitting in `spotfitlm`.

## Model Function

The symmetric 2D Gaussian model is parameterized by $\theta = (x_c, y_c, \sigma, A, b)$:

$$f(x, y; \theta) = A \exp\left(-\frac{(x - x_c)^2 + (y - y_c)^2}{2\sigma^2}\right) + b$$

where:
- $(x_c, y_c)$ is the center position
- $\sigma$ is the Gaussian width (standard deviation)
- $A$ is the amplitude
- $b$ is the background offset

## Objective Function

The fitting minimizes the Poisson negative log-likelihood (NLL):

$$S(\theta) = \sum_{i=1}^{n} \left[ f_i - d_i - d_i \ln\left(\frac{f_i}{d_i}\right) \right]$$

where $d_i$ is the observed pixel intensity and $f_i = f(x_i, y_i; \theta)$ is the model prediction. This is equivalent to the MLE method described by Laurence & Chromy (Nature Methods, 2010).

## Levenberg-Marquardt Algorithm

The optimizer uses a damped Gauss-Newton method with trust-region-like step control.

### Iteration Steps

At each iteration $k$:

1. **Evaluate model and derivatives**
   - Model values: $f_i(\theta^{(k)})$
   - Jacobian: $J_{ij} = \partial f_i / \partial \theta_j$

2. **Compute gradient**
   $$g = J^T \left(1 - \frac{d}{f}\right)$$

3. **Compute approximate Hessian** (Gauss-Newton approximation)
   $$H \approx J^T W J, \quad W_{ii} = \frac{d_i}{f_i^2}$$

4. **Solve damped linear system**
   $$(H + \mu I) \, \delta\theta = g$$

   using Cholesky decomposition, where $\mu > 0$ is the damping parameter.

5. **Compute gain ratio**
   $$\rho = \frac{S(\theta^{(k)}) - S(\theta^{(k)} - \delta\theta)}{g^T \delta\theta - \frac{1}{2} \delta\theta^T H \, \delta\theta}$$

   The numerator is the actual reduction; the denominator is the predicted reduction from the quadratic model.

6. **Update damping parameter**
   - If $\rho < 0.25$ or step increases objective: $\mu \leftarrow 2\mu$ (reject step, retry)
   - Otherwise: $\mu \leftarrow \mu / 3$ (accept step)

   The damping is clipped to $[10^{-8}, 10^{8}]$.

7. **Update parameters**
   $$\theta^{(k+1)} = \theta^{(k)} - \delta\theta$$

### Convergence Criterion

Iteration terminates when the scaled gradient norm falls below tolerance:

$$\sum_{j=1}^{m} \frac{g_j^2}{H_{jj}} < \epsilon$$

where $\epsilon = 10^{-5}$ by default. This scaling accounts for different parameter magnitudes.

## Covariance Estimation

Upon convergence, parameter uncertainties are estimated from the inverse Hessian:

$$\text{Cov}(\hat{\theta}) = H^{-1}$$

The code computes the full analytical Hessian (not just the Gauss-Newton approximation) for more accurate uncertainty estimates. The matrix inverse is computed via Cholesky factorization.

## Return Codes

| Code | Meaning |
|------|---------|
| 0 | Successful convergence |
| -1 | Maximum iterations reached |
| -2 | Hessian not positive definite (covariance unreliable) |

## References

- Laurence, T.A. & Chromy, B.A. (2010). Efficient maximum likelihood estimator fitting of histograms. *Nature Methods*, 7(5), 338-339.
- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization*. Springer. Chapter 10.
