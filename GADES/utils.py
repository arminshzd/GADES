import numpy as np
from jax import vmap, jit, grad, hessian, random, lax
import jax.numpy as jnp
from scipy.optimize import approx_fprime

#@jit
def muller_brown_potential_base(x):
    """
    2D Muller-Brown potential.

    Args:
        x (jax.ndarray): (2, ) vector of x0 and x1

    Returns:
        float: Muller-Brown potential value at `x`
    """
    A = jnp.array([-200, -100, -170, 15])
    a = jnp.array([-1, -1, -6.5, 0.7])
    b = jnp.array([0, 0, 11, 0.6])
    c = jnp.array([-10, -10, -6.5, 0.7])
    x0 = jnp.array([1, 0, -0.5, -1])
    y0 = jnp.array([0, 0.5, 1.5, 1])

    z = jnp.sum(
        A * jnp.exp(
            a * (x[0] - x0) ** 2 +
            b * (x[0] - x0) * (x[1] - y0) +
            c * (x[1] - y0) ** 2
        )
    )
    return z

@jit
def muller_brown_potential(X):
    """
    `vmap` version of the Muller-Brown potential.

    Args:
        X (jax.ndarray): (N, 2) input x0 and x1 values

    Returns:
        jax.ndarray: (N, 1) Muller-Brown potential values
    """
    return vmap(muller_brown_potential_base, in_axes=(0))(X)

@jit
def muller_brown_force_base(x):
    """ Muller-Brown forces at `x` calculated using AD.

    Args:
        x (jax.ndarray): (2, ) position

    Returns:
        jax.ndarray: (2, ) forces vector [-dU/dx0, -dU/dx1]
    """
    return -grad(muller_brown_potential_base)(x)

@jit
def muller_brown_force(X):
    """ `vmap` version of Muller-Brown forces at `X` calculated using AD.

    Args:
        X (jax.ndarray): (N, 2) position

    Returns:
        jax.ndarray: (N, 2) forces vectors [-dU/dx0, -dU/dx1]
    """
    return vmap(muller_brown_force_base, in_axes=(0))(X)

@jit
def muller_brown_hess_base(x):
    """ Muller-Brown Hessian at `x` calculated using AD.

    Args:
        x (jax.ndarray): (2, ) position

    Returns:
        jax.ndarray: (2, 2) Hessian matrix [[ddU/ddx0, ddU/dx0dx1], [ddU/dx1dx0, ddU/ddx1]]
    """
    return hessian(muller_brown_potential_base)(x)

@jit
def muller_brown_hess(X):
    """ `vmap` version of Muller-Brown Hessian at `x` calculated using AD.

    Args:
        x (jax.ndarray): (N, 2) position

    Returns:
        jax.ndarray: (N, 2, 2) Hessian matrix [[ddU/ddx0, ddU/dx0dx1], [ddU/dx1dx0, ddU/ddx1]]
    """
    return vmap(muller_brown_hess_base, in_axes=(0))(X)

@jit
def muller_brown_gad_force_base(position, kappa=0.9):
  """ GADES forces for the Muller-Brown potential at `position` calculated using AD. Calculates the total forces, then finds the most-negative eigenvalue and the corresponding eigenvector of the Hessian and returns negative `kappa` times the force projected in the eigenvector direction as the biasing force.

    Args:
        position (jax.ndarray): (2, ) position
        kappa (float): GAD intensity parameter. Determines how much of the GAD force is used for biasing. `kappa=1` is 100% and `kappa=0` is none.

    Returns:
        jax.ndarray: (2, ) GAD bias forces vector
    """

  # unbiased forces
  forces_u = muller_brown_force_base(position)

  # biased forces (softened by kappa)
  h = muller_brown_hess_base(position)
  w, v = jnp.linalg.eigh(h)
  n = v[:,0]
  n = n/jnp.sqrt(jnp.dot(n,n))
  forces_b = -jnp.dot(forces_u,n)*n
  forces_b *= kappa

  return forces_b

### Plotting helper functions

@jit
def muller_brown_potential_plot(x, y):
    return vmap(lambda yi: vmap(lambda xi: muller_brown_potential_base([xi, yi]))(x))(y)

@jit
def muller_brown_force_plot(x, y):
    return vmap(lambda yi: vmap(lambda xi: muller_brown_force_base(jnp.asarray([xi, yi])))(x))(y)

@jit
def muller_brown_hessian_plot(x, y):
    # `jax.numpy` supports broadcasting, so this works naturally over grids
    return vmap(lambda yi: vmap(lambda xi: muller_brown_hess_base(jnp.asarray([xi, yi])))(x))(y)

### 

@jit
def null_force(X):
    """
    Helper function for return Null forces. Used for unbiased runs.

    Args:
        X (jax.ndarray): (d, ) array of position

    Returns:
        (d, ): Forces vector of all zeros
    """
    return jnp.zeros_like(X)

@jit
def inverse_power_iteration(A, mu=0., num_iters=100, tol=1e-6):
    """
    Computes the smallest eigenvalue and eigenvector of a matrix using inverse power iteration.

    Parameters:
        A (jax.numpy.ndarray): The square matrix (n x n) for which to find the eigenvalue/vector pair with eval closest to mu.
        mu (float): Shift value applied to A to condition search on eigenvalue/vector pair with eval closest to mu.
        num_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance for the eigenvector.

    Returns:
        eigenvalue (float): Smallest eigenvalue of the matrix.
        eigenvector (jax.numpy.ndarray): Corresponding eigenvector (normalized).
    """
    n = A.shape[0]

    # Initialize a random vector as initial guess for evec
    b_k = random.normal(random.PRNGKey(0), shape=(n,))
    b_k = b_k / jnp.linalg.norm(b_k)  # Normalize initial vector

    # Applying shift to matrix A -> A - mu*I
    A_shift = A - mu*jnp.identity(n)

    # Body function implementing inverse power iteration
    def body_fun(state):
        b_k, prev_b_k, iteration = state
        # Solve (A - mu*I)x = b for x as prescribed by inverse power iteration
        b_k_new = jnp.linalg.solve(A_shift, b_k)
        b_k_new = b_k_new / jnp.linalg.norm(b_k_new)  # Normalize the vector
        return b_k_new, b_k, iteration + 1

    # Conditional function defining convergence
    def cond_fun(state):
        b_k, prev_b_k, iteration = state
        not_converged = jnp.linalg.norm(b_k - prev_b_k) > tol
        not_max_iter = iteration < num_iters
        return not_converged & not_max_iter

    # Initialize state with (current vector, previous vector, iteration count)
    initial_state = (b_k, jnp.zeros_like(b_k), 0)

    # Iterate using lax.while_loop
    final_state = lax.while_loop(cond_fun, body_fun, initial_state)

    # Extract the final vector
    b_k = final_state[0]

    # Compute the smallest eigenvalue using the Rayleigh quotient
    eigenvalue = jnp.dot(b_k, jnp.dot(A, b_k)) / jnp.dot(b_k, b_k)

    return eigenvalue, b_k

def baoab_langevin_integrator(positions, velocities, forces_u, forces_b, mass, gamma, dt, kBT, force_function_u, force_function_b, n_steps=1):
    """
    BAOAB Langevin integrator based on Leimkuhler and Matthews (2013).
    https://dx.doi.org/10.1093/amrx/abs010

    Parameters:
        positions (jax.ndarray): Initial positions (shape: [D, ], where D is dimensionality).
        velocities (jax.ndarray): Initial velocities (shape: [D, ]).
        forces_u (jax.ndarray): Initial unbiased forces (shape: [D, ]).
        forces_b (jax.ndarray): Initial biased forces (shape: [D, ]).
        mass (float): Mass of the particles (scalar).
        gamma (float): Friction coefficient. (scalar).
        dt (float): Time step. (scalar).
        n_steps (int): Number of simulation steps. (scalar).
        kBT (float): Thermal energy (\(k_B T\)). (scalar).
        force_function_u (callable): Function to compute unbiased forces given positions (returns forces of shape [D, ]).
        force_function_b (callable): Function to compute biased forces given positions (returns forces of shape [D, ]).

    Returns:
        positions (jax.ndarray): New positions (shape: [D, ]).
        velocities (jax.ndarray): New velocities (shape: [D, ]).
        forces_u (jax.ndarray): Unbiased forces at new position (shape: [D, ]).
        forces_b (jax.ndarray): Biased forces at new position (shape: [D, ]).
    """
    dim = positions.shape[0]

    # Precompute constants
    c1 = jnp.exp(-gamma * dt)
    c3 = jnp.sqrt(kBT * (1 - c1**2))
    inv_mass = jnp.reciprocal(mass)
    inv_mass_sqrt = jnp.reciprocal(jnp.sqrt(mass))

    for step in range(n_steps):

        # Step B (First half-step momentum update)
        forces = forces_u + forces_b
        velocities += 0.5 * dt * inv_mass * forces

        # Step A (Half-step position update)
        positions += 0.5 * dt * velocities

        # Step O (Thermostat and randomization)
        random_force = np.random.normal(size=(dim,))
        velocities = c1 * velocities + c3 * inv_mass_sqrt * random_force

        # Step A (Second half-step position update)
        positions += 0.5 * dt * velocities

        # Step B (Second half-step momentum update)
        forces_u = force_function_u(positions)
        forces_b = force_function_b(positions)
        forces = forces_u + forces_b
        velocities += 0.5 * dt * inv_mass * forces

    return positions, velocities, forces_u, forces_b

def get_hessian_fdiff(func, x0, epsilon=1e-6):
    """
    Compute the Hessian matrix of a scalar function using finite differences.

    Parameters:
        func (callable): The scalar function f(x) whose Hessian is to be computed.
        x0 (ndarray): The point at which the Hessian is evaluated.
        epsilon (float): Small step size for finite difference approximation.

    Returns:
        ndarray: The Hessian matrix of f at x0.
    """
    n = len(x0)
    hessian_matrix = np.zeros((n, n))
    f1 = approx_fprime(x0, func, epsilon)  # Gradient at x0

    for i in range(n):
        x_i = x0.copy()
        x_i[i] += epsilon  # Perturb along dimension i
        f2 = approx_fprime(x_i, func, epsilon)  # Gradient after perturbation
        hessian_matrix[:, i] = (f2 - f1) / epsilon  # Second derivative approximation

    return hessian_matrix

