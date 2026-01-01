"""
Core utility functions for GADES.

This module provides essential utilities for GADES computations:
- Hessian matrix computation via finite differences
- Force magnitude clamping
"""

from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import approx_fprime
from joblib import Parallel, delayed
import openmm

if TYPE_CHECKING:
    from .backend import Backend


def get_hessian_fdiff(func: Callable, x0: np.ndarray, epsilon: Optional[float] = 1e-6) -> np.ndarray:
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


def central_diff_ij(
    func: Callable, x0: np.ndarray, i: int, j: int, epsilon: float
) -> Tuple[int, int, float]:
    """Compute Hessian element H[i,j] using central differences."""
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


def get_hessian_cdiff_parallel(func: Callable, x0: np.ndarray,
                               epsilon: Optional[float] = 1e-5,
                               n_jobs: int = -1) -> np.ndarray:
    """
    Compute the Hessian matrix using central differences in parallel.

    Parameters:
        func (callable): The scalar function f(x) whose Hessian is to be computed.
        x0 (ndarray): The point at which the Hessian is evaluated.
        epsilon (float): Small step size for finite difference approximation.
        n_jobs (int): Number of parallel workers (-1 for all cores).

    Returns:
        ndarray: The symmetric Hessian matrix of f at x0.
    """
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


def _get_openMM_forces(context: openmm.Context,
                       positions: openmm.unit.Quantity) -> np.ndarray:
    """
    Compute the original (unbiased) forces from an OpenMM context (internal use only).

    This function updates the context with the provided positions, then retrieves
    forces from force group `0` only. Group `0` is assumed to correspond to the
    system's original potential (e.g., the PMF) without additional bias terms.
    The forces are converted to units of kJ/mol/nm and flattened into a 1D array.

    Args:
        context (openmm.Context):
            The OpenMM context containing the current system and integrator state.
        positions (openmm.unit.Quantity):
            Atomic positions, shaped `(N, 3)` with distance units compatible with OpenMM.

    Returns:
        np.ndarray:
            Flattened force vector of shape `(3 * N,)`, in units of kJ/mol/nm.

    Notes:
        - By restricting to `groups={0}`, the returned forces exclude any
          externally applied bias forces (e.g., from GADES).
    """
    context.setPositions(positions)
    # the `groups` keyword makes sure we're only capturing the forces from the
    # original pmf and not the biased one.
    state = context.getState(getForces=True, groups={0})
    forces = state.getForces(asNumpy=True).value_in_unit(
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
    return -forces.flatten()


def _get_forces(backend: "Backend", positions: np.ndarray) -> np.ndarray:
    """Get forces from a backend at given positions."""
    return backend.get_forces(positions)


def compute_hessian_force_fd_block_parallel(
    backend: "Backend",
    atom_indices: Sequence[int],
    epsilon: Optional[float] = 1e-4,
    n_jobs: Optional[int] = -1,
    platform_name: Optional[str] = 'CPU',
) -> np.ndarray:
    """
    Compute the Hessian block for a subset of atoms via finite-difference forces.

    This function builds the Hessian matrix (second derivatives of the potential
    energy with respect to Cartesian coordinates) for a selected set of atoms.
    The calculation perturbs each coordinate by a small displacement and computes
    the corresponding force differences in parallel. This is the parallel version
    of `compute_hessian_force_fd_block_serial`. The performance gain of the parallel
    version is minimal for systems with <10000 biased particles. Because of `joblib`
    overhead, this method is in fact __slower__ than the serial version for small
    systems.

    Args:
        backend:
            The backend object providing system state and force calculations.
        atom_indices (Sequence[int] or None):
            Indices of atoms to include in the Hessian block. If None, all atoms
            are included.
        epsilon (float, optional):
            Finite-difference displacement step size (in nanometers).
            Default is `1e-4`.
        n_jobs (int, optional):
            Number of parallel workers for finite-difference force evaluations.
            Default is `-1` (use all available cores).
        platform_name (str, optional):
            OpenMM platform to use for evaluations (e.g., `"CPU"`, `"CUDA"`).
            Default is `"CPU"`.

    Returns:
        np.ndarray:
            A symmetric Hessian block matrix of shape `(3M, 3M)`, where `M` is
            the number of atoms in `atom_indices`. Units are kJ/(mol·nm²).

    Notes:
        - The Hessian is computed column by column using finite-difference forces:
          ```
          H_ij = d²V / (dx_i dx_j)
          ```
        - Parallelization uses `joblib.Parallel` with the `'threading'` backend.
          The `'loky'` (multiprocessing) backend cannot be used because OpenMM/ASE
          backends contain non-picklable objects (contexts, file handles, etc.).
        - Due to Python's GIL, threading provides limited speedup for CPU-bound
          workloads. For most systems, `compute_hessian_force_fd_block_serial` or
          `compute_hessian_force_fd_richardson` are recommended instead.
        - The final matrix is symmetrized to mitigate finite-difference noise.

    Examples:
        >>> hess_block = compute_hessian_force_fd_block_parallel(
        ...     backend, atom_indices=[0, 1, 2], epsilon=1e-4, n_jobs=4
        ... )
        >>> hess_block.shape
        (9, 9)
    """
    positions_array = backend.get_positions()
    n_atoms = len(positions_array)

    # Map atom indices to coordinate indices
    if atom_indices is None:
        atom_indices = np.arange(0, n_atoms)

    coord_indices = []
    for idx in atom_indices:
        coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m_dof = len(coord_indices)

    def compute_block_column(j):
        # Reference forces (unbiased, flattened)
        # positions_array is captured from outer scope
        f0 = backend.get_forces(positions_array)[coord_indices]

        # Perturb along coordinate j
        perturbed_pos = positions_array.flatten().copy()
        perturbed_pos[j] += epsilon
        perturbed_pos = perturbed_pos.reshape((-1, 3))

        f_perturbed = backend.get_forces(perturbed_pos)[coord_indices]

        df = (f_perturbed - f0) / epsilon

        return j, df

    # Parallel over selected j columns only
    # Use 'threading' backend because OpenMM/ASE backends are not picklable
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_block_column)(j) for j in coord_indices
    )

    # Assemble square Hessian block
    hessian_block = np.zeros((m_dof, m_dof))
    for col_idx, (j, df) in enumerate(results):
        hessian_block[:, col_idx] = df

    # Symmetrize block
    hessian_block = 0.5 * (hessian_block + hessian_block.T)

    return hessian_block


def compute_hessian_force_fd_block_serial(
    backend: "Backend",
    atom_indices: Sequence[int],
    epsilon: Optional[float] = 1e-4,
    platform_name: Optional[str] = 'CPU',
) -> np.ndarray:
    """
    Compute the Hessian block for a subset of atoms via finite-difference forces (serial version).

    This function constructs the Hessian matrix (second derivatives of the potential
    energy with respect to Cartesian coordinates) for a selected set of atoms. The
    calculation perturbs each coordinate one at a time and computes the corresponding
    force differences, without parallelization. Use this version for system with
    <10000 biased atoms.

    Args:
        backend:
            The backend object providing system state and force calculations.
        atom_indices (Sequence[int] or None):
            Indices of atoms to include in the Hessian block. If None, all atoms
            are included.
        epsilon (float, optional):
            Finite-difference displacement step size (in nanometers).
            Default is `1e-4`.
        platform_name (str, optional):
            OpenMM platform to use for evaluations (e.g., `"CPU"`, `"CUDA"`).
            Default is `"CPU"`.

    Returns:
        np.ndarray:
            A symmetric Hessian block matrix of shape `(3M, 3M)`, where `M` is
            the number of atoms in `atom_indices`. Units are kJ/(mol·nm²).

    Notes:
        - The Hessian is computed column by column using finite-difference forces:
          ```
          H_ij = d²V / (dx_i dx_j)
          ```
        - This serial implementation is simpler but slower than the parallel
          version (`compute_hessian_force_fd_block_parallel`) for large systems.
        - The final matrix is symmetrized to mitigate finite-difference noise.

    Examples:
        >>> hess_block = compute_hessian_force_fd_block_serial(
        ...     backend, atom_indices=[0, 1], epsilon=1e-4
        ... )
        >>> hess_block.shape
        (6, 6)
    """
    positions_array = backend.get_positions()
    n_atoms = len(positions_array)

    # Map atom indices to coordinate indices
    if atom_indices is None:
        atom_indices = np.arange(0, n_atoms)

    coord_indices = []
    for idx in atom_indices:
        coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m_dof = len(coord_indices)

    # Prepare Hessian block
    hessian_block = np.zeros((m_dof, m_dof))

    # Reference forces on selected coordinates
    f0 = backend.get_forces(positions_array)
    f0 = f0[coord_indices]

    # Loop over selected perturbations
    for col_idx, j in enumerate(coord_indices):
        perturbed_pos = positions_array.flatten()
        perturbed_pos[j] += epsilon
        perturbed_pos = perturbed_pos.reshape((-1, 3))

        f_perturbed = backend.get_forces(perturbed_pos)
        f_perturbed = f_perturbed[coord_indices]

        df = (f_perturbed - f0) / epsilon
        hessian_block[:, col_idx] = df

    # Symmetrize
    hessian_block = 0.5 * (hessian_block + hessian_block.T)

    return hessian_block


def compute_hessian_force_fd_richardson(
    backend: "Backend",
    atom_indices: Sequence[int],
    step_size: Optional[float] = 1e-4,
    platform_name: Optional[str] = 'CPU',
    factors: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Compute the Hessian block for a subset of atoms using Richardson-extrapolated
    finite differences.

    This method estimates second derivatives of the potential energy by
    recursively applying Richardson extrapolation to finite-difference
    force calculations at multiple step sizes. This improves accuracy
    compared to a single-step finite-difference scheme. This is the go-to method
    for calculating numerical Hessian for GADES. Using the Richardson extrapolation
    drastically reduces the depency of accuracy on step size and prevents numerical
    error.

    Args:
        backend:
            The backend object providing system state and force calculations.
        atom_indices (Sequence[int] or None):
            Indices of atoms to include in the Hessian block. If None, all atoms
            are included.
        step_size (float, optional):
            Base finite-difference displacement step size (in nanometers).
            Default is `1e-4`.
        platform_name (str, optional):
            OpenMM platform to use for evaluations (e.g., `"CPU"`, `"CUDA"`).
            Default is `"CPU"`.
        factors (Sequence[float], optional):
            Decreasing list of scaling factors for step sizes, applied to `epsilon`.
            Must be strictly decreasing (e.g., `[1.0, 0.5, 0.25]`).
            Default is `[1.0, 0.5, 0.25]`.

    Returns:
        np.ndarray:
            A symmetric Hessian block matrix of shape `(3M, 3M)`, where `M` is
            the number of atoms in `atom_indices`. Units are kJ/(mol·nm²).

    Notes:
        - The Hessian is computed column by column. For each perturbed coordinate,
          force differences are evaluated at multiple step sizes and combined via
          Richardson extrapolation:
          ```
          R(k, i) = (r * R(k-1, i+1) - R(k-1, i)) / (r - 1)
          ```
          where `r = h_i / h_{i+k}` is the ratio of step sizes.
        - Using more factors generally improves accuracy, but increases cost.
        - The final Hessian is symmetrized to reduce numerical noise.

    Examples:
        >>> hess_block = compute_hessian_force_fd_richardson(
        ...     backend, atom_indices=[0, 1],
        ...     step_size=1e-4, factors=[1.0, 0.5, 0.25]
        ... )
        >>> hess_block.shape
        (6, 6)
    """
    if factors is None:
        factors = [1.0, 0.5, 0.25]  # Default: up to third order

    positions_array = backend.get_positions()
    n_atoms = len(positions_array)

    if atom_indices is None:
        atom_indices = np.arange(0, n_atoms)

    coord_indices = []
    for idx in atom_indices:
        coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m_dof = len(coord_indices)

    hessian_block = np.zeros((m_dof, m_dof))

    # Reference (baseline) forces on selected coordinates
    f0 = backend.get_forces(positions_array)
    f0 = f0[coord_indices]

    for col_idx, j in enumerate(coord_indices):
        # First, compute all finite-difference derivatives
        D = []
        for factor in factors:
            perturbed_pos = positions_array.copy().flatten()
            perturbed_pos[j] += factor * step_size
            perturbed_pos = perturbed_pos.reshape((-1, 3))

            f = backend.get_forces(perturbed_pos)
            f = f[coord_indices]

            d = (f - f0) / (factor * step_size)
            D.append(d)

        # Build Richardson tableau
        R = [D]
        for k in range(1, len(factors)):
            prev = R[-1]
            new = []
            for i in range(len(prev) - 1):
                r = (factors[i] / factors[i + k]) ** 1  # first-order FD
                Rij = (r * prev[i + 1] - prev[i]) / (r - 1)
                new.append(Rij)
            R.append(new)

        # Take the most extrapolated value
        hessian_block[:, col_idx] = R[-1][0]

    # Symmetrize
    hessian_block = 0.5 * (hessian_block + hessian_block.T)

    return hessian_block


def clamp_force_magnitudes(forces_flat: np.ndarray, max_force: float) -> np.ndarray:
    """
    Clamp the magnitudes of 3D force vectors in a flattened array.

    This function rescales each 3D force vector so that the magnitude of the bias
    force on each particle does not exceed `max_force`. The input is a flattened
    array where each consecutive triplet of values corresponds to one `(fx, fy, fz)` vector.

    Args:
        forces_flat (np.ndarray):
            Flattened array of shape `(3 * N,)`, where `N` is the number of
            force vectors. Each consecutive triplet represents a 3D force.
        max_force (float):
            Maximum allowed magnitude for each force vector. Forces with
            smaller magnitudes are unchanged.

    Returns:
        np.ndarray:
            Flattened array of the same shape as `forces_flat`, where each
            3D force vector has magnitude <= `max_force`.

    Notes:
        - Zero-length vectors remain unchanged.
        - The scaling is applied independently to each force vector.

    Examples:
        >>> import numpy as np
        >>> forces = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 10.0])  # two vectors
        >>> clamped = clamp_force_magnitudes(forces, max_force=5.0)
        >>> clamped
        array([3., 4., 0., 0., 0., 5.])
    """
    forces = forces_flat.reshape(-1, 3)
    magnitudes = np.linalg.norm(forces, axis=1)
    scale = np.minimum(1, np.where(magnitudes != 0, max_force / magnitudes, 1))
    forces_clamped = forces * scale[:, np.newaxis]
    return forces_clamped.flatten()
