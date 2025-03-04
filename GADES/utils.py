import numpy as np
from jax import vmap, jit, grad, hessian, random, lax
import jax.numpy as jnp

@jit
def muller_brown_potential_base(x):
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
    return vmap(muller_brown_potential_base, in_axes=(0))(X)

@jit
def muller_brown_force_base(x):
    return -grad(muller_brown_potential_base)(x)

@jit
def muller_brown_force(X):
    return vmap(muller_brown_force_base, in_axes=(0))(X)

@jit
def muller_brown_hess_base(x):
    return hessian(muller_brown_potential_base)(x)

@jit
def muller_brown_hess(X):
    return vmap(muller_brown_hess_base, in_axes=(0))(X)

@jit
def muller_brown_gad_force_base(position, kappa=0.9):

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

@jit
def null_force(X):
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
        positions (np.ndarray): Initial positions (shape: [D, ], where D is dimensionality).
        velocities (np.ndarray): Initial velocities (shape: [D, ]).
        forces_u (np.ndarray): Initial unbiased forces (shape: [D, ]).
        forces_b (np.ndarray): Initial biased forces (shape: [D, ]).
        mass (float): Mass of the particles (scalar).
        gamma (float): Friction coefficient. (scalar).
        dt (float): Time step. (scalar).
        n_steps (int): Number of simulation steps. (scalar).
        kBT (float): Thermal energy (\(k_B T\)). (scalar).
        force_function_u (callable): Function to compute unbiased forces given positions (returns forces of shape [D, ]).
        force_function_b (callable): Function to compute biased forces given positions (returns forces of shape [D, ]).

    Returns:
        positions (np.ndarray): New positions (shape: [D, ]).
        velocities (np.ndarray): New velocities (shape: [D, ]).
        forces_u (np.ndarray): Unbiased forces at new position (shape: [D, ]).
        forces_b (np.ndarray): Biased forces at new position (shape: [D, ]).
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