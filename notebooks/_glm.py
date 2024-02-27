import jax
import jax.numpy as jnp


def f(p, gy, gx):
    xc, yc, sigma, A, bg = p
    argx = ((gx - xc) / sigma) ** 2 / 2.0
    argy = ((gy - yc) / sigma) ** 2 / 2.0
    return A * jnp.exp(-(argx + argy)) + bg

def negloglik(p, data, gy, gx):
    model = f(p, gy, gx)
    ratio = jnp.maximum(model, 1e-8) / jnp.maximum(data, 1e-8)
    return jnp.sum(model - data - data*jnp.log(ratio))

fgrad = jax.jit(jax.grad(negloglik, argnums=0))
fhess = jax.jit(jax.hessian(negloglik, argnums=0))
funcjac = jax.jacfwd(f, argnums=0)