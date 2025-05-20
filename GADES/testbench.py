import numpy as np
from scipy.differentiate import hessian as scp_hess
import jax.numpy as jnp
from jax import grad, hessian

from utils import muller_brown_potential_base

f = muller_brown_potential_base
g = grad(f)
H = hessian(f)

h_test = lambda x: scp_hess(f, x)

X1 = jnp.asarray([0.5, 0.5])
f1 = f(X1)
g1 = g(X1)
H1 = H(X1)
H1p = h_test(np.asarray(X1))
print(H1)
print(H1p)