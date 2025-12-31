"""
Lanczos algorithm for eigenvalue computation.

This module provides NumPy-based implementations of the Lanczos algorithm
for computing eigenvalues and eigenvectors of symmetric matrices.
"""

from typing import Dict, Optional, Tuple

import numpy as np


# Default options for Lanczos iterations
options: Dict[str, int] = {
    "N_ITER": 10,
}


def lanczos(
    A: np.ndarray, n_iter: Optional[int] = None, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lanczos method for calculating eigenvalues and eigenvectors of a symmetric matrix.

    The Lanczos algorithm builds an orthonormal basis for the Krylov subspace and
    produces a tridiagonal matrix whose eigenvalues approximate those of the original
    matrix. This is particularly efficient for finding extreme eigenvalues of large
    sparse matrices.

    Args:
        A (np.ndarray): (N, N) symmetric matrix to compute eigenvalues/eigenvectors for.
        n_iter (int, optional): Number of Lanczos iterations. Defaults to options["N_ITER"].
            More iterations give better approximations but increase computation time.
        seed (int, optional): Random seed for reproducibility of the initial vector.

    Returns:
        tuple: (eigenvalues, eigenvectors) where:
            - eigenvalues: (n_iter,) array of approximate eigenvalues, sorted ascending
            - eigenvectors: (N, n_iter) array of corresponding eigenvectors as columns

    Example:
        >>> A = np.array([[4, 1], [1, 3]])
        >>> eigvals, eigvecs = lanczos(A, n_iter=2)
        >>> # Compare with np.linalg.eigh(A)
    """
    if n_iter is None:
        n_iter = options["N_ITER"]

    n = A.shape[0]
    n_iter = min(n_iter, n)  # Can't have more iterations than matrix dimension

    V = np.zeros((n, n_iter))
    T = np.zeros((n_iter, n_iter))

    # Initialize random unit vector
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    # First iteration
    w = A @ v
    alpha = np.dot(v, w)
    w = w - alpha * v

    V[:, 0] = v
    T[0, 0] = alpha

    # Lanczos iterations
    for j in range(1, n_iter):
        beta = np.linalg.norm(w)

        if beta < 1e-12:
            # Invariant subspace found, restart with new random vector
            v_next = rng.standard_normal(n)
            # Orthogonalize against all previous vectors
            for k in range(j):
                v_next = v_next - np.dot(V[:, k], v_next) * V[:, k]
            v_next = v_next / np.linalg.norm(v_next)
        else:
            v_next = w / beta

        w = A @ v_next - beta * v
        alpha = np.dot(v_next, w)
        w = w - alpha * v_next

        # Re-orthogonalization for numerical stability
        for k in range(j + 1):
            w = w - np.dot(V[:, k], w) * V[:, k]

        V[:, j] = v_next
        T[j, j] = alpha
        T[j, j - 1] = beta
        T[j - 1, j] = beta

        v = v_next

    # Compute eigenvalues and eigenvectors of tridiagonal matrix T
    eigvals, eigvecs_T = np.linalg.eigh(T)

    # Transform eigenvectors back to original space
    eigvecs = V @ eigvecs_T

    # Normalize eigenvectors
    norms = np.linalg.norm(eigvecs, axis=0)
    eigvecs = eigvecs / norms

    return eigvals, eigvecs


def lanczos_smallest(
    A: np.ndarray, n_iter: Optional[int] = None, seed: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Find the smallest eigenvalue and corresponding eigenvector using Lanczos.

    This is a convenience function that returns only the smallest (most negative)
    eigenvalue and its eigenvector, which is useful for finding the softest mode
    in molecular dynamics applications.

    Args:
        A: ``(N, N)`` symmetric matrix.
        n_iter: Number of Lanczos iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (eigenvalue, eigenvector) where eigenvalue is a float (the smallest
        eigenvalue) and eigenvector is a ``(N,)`` array (the corresponding normalized
        eigenvector).
    """
    eigvals, eigvecs = lanczos(A, n_iter=n_iter, seed=seed)
    idx = np.argmin(eigvals)
    return float(eigvals[idx]), eigvecs[:, idx]


def lanczos_shift_invert(
    A: np.ndarray,
    sigma: float,
    n_iter: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lanczos method with shift-and-invert to find eigenvalues near a target value.

    This variant finds eigenvalues of A closest to `sigma` by applying Lanczos
    to (A - sigma*I)^(-1). The eigenvalues of this shifted-inverted matrix that
    are largest in magnitude correspond to eigenvalues of A closest to sigma.

    Args:
        A (np.ndarray): (N, N) symmetric matrix.
        sigma (float): Target shift value. Eigenvalues of A near this value
            will be found most accurately.
        n_iter (int, optional): Number of Lanczos iterations.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (eigenvalues, eigenvectors) of A, sorted by distance from sigma.

    Note:
        This requires solving linear systems with (A - sigma*I), which can be
        expensive for large matrices. Best suited for finding interior eigenvalues.
    """
    if n_iter is None:
        n_iter = options["N_ITER"]

    n = A.shape[0]
    n_iter = min(n_iter, n)

    # Compute LU factorization of (A - sigma*I) for efficient solves
    shifted = A - sigma * np.eye(n)

    try:
        # Try to use LU decomposition for efficiency
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(shifted)

        def solve(b):
            return lu_solve((lu, piv), b)
    except ImportError:
        # Fall back to direct solve
        def solve(b):
            return np.linalg.solve(shifted, b)

    V = np.zeros((n, n_iter))
    T = np.zeros((n_iter, n_iter))

    # Initialize random unit vector
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    # First iteration with (A - sigma*I)^(-1)
    w = solve(v)
    alpha = np.dot(v, w)
    w = w - alpha * v

    V[:, 0] = v
    T[0, 0] = alpha

    # Lanczos iterations
    for j in range(1, n_iter):
        beta = np.linalg.norm(w)

        if beta < 1e-12:
            v_next = rng.standard_normal(n)
            for k in range(j):
                v_next = v_next - np.dot(V[:, k], v_next) * V[:, k]
            v_next = v_next / np.linalg.norm(v_next)
        else:
            v_next = w / beta

        w = solve(v_next) - beta * v
        alpha = np.dot(v_next, w)
        w = w - alpha * v_next

        # Re-orthogonalization
        for k in range(j + 1):
            w = w - np.dot(V[:, k], w) * V[:, k]

        V[:, j] = v_next
        T[j, j] = alpha
        T[j, j - 1] = beta
        T[j - 1, j] = beta

        v = v_next

    # Compute eigenvalues of T and transform back
    eigvals_T, eigvecs_T = np.linalg.eigh(T)

    # Transform eigenvalues back: lambda_A = 1/lambda_T + sigma
    eigvals = 1.0 / eigvals_T + sigma

    # Transform eigenvectors back to original space
    eigvecs = V @ eigvecs_T
    norms = np.linalg.norm(eigvecs, axis=0)
    eigvecs = eigvecs / norms

    # Sort by distance from sigma
    order = np.argsort(np.abs(eigvals - sigma))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return eigvals, eigvecs
