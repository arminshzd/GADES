import numpy as np
from jax import vmap, jit, grad, hessian, random, lax
import jax.numpy as jnp
from scipy.optimize import approx_fprime
from joblib import Parallel, delayed
import openmm


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


def central_diff_ij(func, x0, i, j, epsilon):
    x_ijp = x0.copy()
    x_ijm = x0.copy()
    x_ipj = x0.copy()
    x_imj = x0.copy()

    x_ijp[i] += epsilon
    x_ijp[j] += epsilon

    x_ijm[i] += epsilon
    x_ijm[j] -= epsilon

    x_ipj[i] -= epsilon
    x_ipj[j] += epsilon

    x_imj[i] -= epsilon
    x_imj[j] -= epsilon

    f_ijp = func(x_ijp)
    f_ijm = func(x_ijm)
    f_ipj = func(x_ipj)
    f_imj = func(x_imj)

    hess_ij = (f_ijp - f_ijm - f_ipj + f_imj) / (4 * epsilon**2)
    return (i, j, hess_ij)

def get_hessian_cdiff_parallel(func, x0, epsilon=1e-5, n_jobs=-1):
    n = len(x0)
    tasks = [(i, j) for i in range(n) for j in range(i, n)]

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(central_diff_ij)(func, x0, i, j, epsilon) for i, j in tasks
    )

    hessian = np.zeros((n, n))
    for i, j, val in results:
        hessian[i, j] = val
        hessian[j, i] = val  # enforce symmetry

    return hessian

def _get_openMM_forces(context, positions):
    context.setPositions(positions)
    state = context.getState(getForces=True, groups={0}) # the `groups` keyword makes sure we're only capturing the forces from the original pmf and not the biased one.
    forces = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
    return -forces.flatten()

def compute_hessian_force_fd_block_parallel(system, positions, atom_indices, epsilon=1e-4, n_jobs=-1, platform_name='CPU'):
    """
    Compute square Hessian block (3M x 3M) for selected atoms only.

    Parameters:
    - atom_indices: list of atom indices (0-based).
    - system: OpenMM System.
    - positions: OpenMM Quantity array (nanometers).
    - epsilon: finite difference step.
    - n_jobs: number of parallel workers.
    - platform_name: 'CPU' or 'CUDA'.

    Returns:
    - Symmetric Hessian block (3M x 3M).
    """
    n_atoms = len(positions)
    positions_array = np.asarray(positions.value_in_unit(openmm.unit.nanometer))

    # Map atom indices to coordinate indices
    if atom_indices is None:
        atom_indices = np.arange(0, n_atoms)

    coord_indices = []
    for idx in atom_indices:
        coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m_dof = len(coord_indices)

    def compute_block_column(j):
        integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(system, integrator, platform)

        positions = positions_array * openmm.unit.nanometer
        context.setPositions(positions)

        # Reference forces (full, but we'll slice)
        f0 = _get_openMM_forces(context, positions)[coord_indices]

        # Perturb along coordinate j
        perturbed_pos = positions_array.flatten()
        perturbed_pos[j] += epsilon
        perturbed_pos = perturbed_pos.reshape((-1, 3)) * openmm.unit.nanometer

        f_perturbed = _get_openMM_forces(context, perturbed_pos)[coord_indices]

        df = (f_perturbed - f0) / epsilon

        del context
        del integrator

        return j, df

    # Parallel over selected j columns only
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_block_column)(j) for j in coord_indices
    )

    # Assemble square Hessian block
    hessian_block = np.zeros((m_dof, m_dof))
    for col_idx, (j, df) in enumerate(results):
        hessian_block[:, col_idx] = df

    # Symmetrize block
    hessian_block = 0.5 * (hessian_block + hessian_block.T)

    return hessian_block

def compute_hessian_force_fd_block_serial(system, positions, atom_indices, epsilon=1e-4, platform_name='CPU'):
    """
    Compute square Hessian block (3M x 3M) for selected atoms (serial, non-parallel).

    Parameters:
    - system: OpenMM System.
    - positions: OpenMM Quantity array (nanometers).
    - atom_indices: list of atom indices (0-based) to compute block for.
    - epsilon: finite difference step.
    - platform_name: 'CPU' or 'CUDA'.

    Returns:
    - Symmetric Hessian block (3M x 3M).
    """
    n_atoms = len(positions)
    positions_array = positions.value_in_unit(openmm.unit.nanometer)
    positions_array = np.array(positions_array)  # Convert Vec3 list to numpy

    # Map atom indices to coordinate indices
    if atom_indices is None:
        atom_indices = np.arange(0, n_atoms)
    
    coord_indices = []
    for idx in atom_indices:
        coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m_dof = len(coord_indices)

    # Prepare Hessian block
    hessian_block = np.zeros((m_dof, m_dof))

    # Create context (reuse for all columns)
    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)

    # Reference forces on selected coordinates
    positions_nm = positions_array * openmm.unit.nanometer
    f0 = _get_openMM_forces(context, positions_nm)[coord_indices]

    # Loop over selected perturbations
    for col_idx, j in enumerate(coord_indices):
        perturbed_pos = positions_array.flatten()
        perturbed_pos[j] += epsilon
        perturbed_pos = perturbed_pos.reshape((-1, 3)) * openmm.unit.nanometer

        f_perturbed = _get_openMM_forces(context, perturbed_pos)[coord_indices]

        df = (f_perturbed - f0) / epsilon
        hessian_block[:, col_idx] = df

    # Symmetrize
    hessian_block = 0.5 * (hessian_block + hessian_block.T)

    # Cleanup
    del context
    del integrator

    return hessian_block

def compute_hessian_force_fd_richardson(system, positions, atom_indices, epsilon=1e-4, platform_name='CPU'):
    """
    Compute Hessian block using Richardson extrapolation with 3 step sizes: eps, eps/2, eps/4.

    Parameters:
    - system: OpenMM System.
    - positions: OpenMM Quantity array (nanometers).
    - atom_indices: list of atom indices (0-based) to compute block for.
    - epsilon: base finite difference step.
    - platform_name: 'CPU' or 'CUDA'.

    Returns:
    - Symmetric Hessian block (3M x 3M) with third-order accuracy.
    """
    n_atoms = len(positions)
    positions_array = positions.value_in_unit(openmm.unit.nanometer)
    positions_array = np.array(positions_array)

    if atom_indices is None:
        atom_indices = np.arange(0, n_atoms)

    coord_indices = []
    for idx in atom_indices:
        coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m_dof = len(coord_indices)

    hessian_block = np.zeros((m_dof, m_dof))

    # Create context
    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)

    # Baseline force
    f0 = _get_openMM_forces(context, positions_array * openmm.unit.nanometer)[coord_indices]

    for col_idx, j in enumerate(coord_indices):
        f_eps_list = []
        for factor in [1.0, 0.5, 0.25]:
            perturbed_pos = positions_array.copy().flatten()
            perturbed_pos[j] += factor * epsilon
            perturbed_pos = perturbed_pos.reshape((-1, 3)) * openmm.unit.nanometer
            f = _get_openMM_forces(context, perturbed_pos)[coord_indices]
            f_eps_list.append(f)

        # Finite differences
        D0 = (f_eps_list[0] - f0) / epsilon
        D1 = (f_eps_list[1] - f0) / (epsilon / 2)
        D2 = (f_eps_list[2] - f0) / (epsilon / 4)

        # Richardson extrapolation to O(epsilon^3)
        R1_0 = 2 * D1 - D0
        R1_1 = 2 * D2 - D1
        R2_0 = (4 * R1_1 - R1_0) / 3

        hessian_block[:, col_idx] = R2_0

    hessian_block = 0.5 * (hessian_block + hessian_block.T)

    del context, integrator
    return hessian_block

def clamp_force_magnitudes(forces_flat, max_force):
    """
    Clamp the magnitudes of 3D force vectors represented in a flattened array.

    Parameters:
    - forces_flat (np.ndarray): Flattened array of shape (3 * N,), where each consecutive
                                triplet is a 3D force vector.
    - max_force (float): Maximum allowed magnitude for each force vector.

    Returns:
    - np.ndarray: Flattened array with clamped force vectors, same shape as input.
    """
    forces = forces_flat.reshape(-1, 3)
    magnitudes = np.linalg.norm(forces, axis=1)
    scale = np.minimum(1, np.where(magnitudes != 0, max_force / magnitudes, 1))
    forces_clamped = forces * scale[:, np.newaxis]
    return forces_clamped.flatten()