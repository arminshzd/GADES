"""
Bofill Hessian update algorithm.

This module provides the Bofill update formula for approximating the Hessian
matrix based on position and gradient changes, without requiring explicit
Hessian computation at each step.

Reference:
    Bofill, J. M. (1994). Updated Hessian matrix and the restricted step method
    for locating transition structures. J. Comput. Chem., 15(1), 1-11.
    https://doi.org/10.1002/jcc.540150102

    Anglada, J. M., & Bofill, J. M. (1998). On the restricted step method
    coupled with the augmented Hessian for the search of stationary points.
    Int. J. Quantum Chem., 62(2), 153-165.
    https://doi.org/10.1002/qua.10709
"""

import numpy as np


def _get_spectral_abs(A: np.ndarray) -> np.ndarray:
    """
    Compute the spectral absolute value of a matrix.

    Given a matrix A, computes |A| where the eigenvalues have the same magnitude
    as those of A, but are all positive. This is done by eigendecomposition,
    taking absolute values of eigenvalues, and reconstructing.

    Args:
        A (np.ndarray): (N, N) square matrix.

    Returns:
        np.ndarray: (N, N) matrix |A| with all positive eigenvalues.

    Note:
        For symmetric matrices, this is equivalent to replacing negative
        eigenvalues with their absolute values while preserving eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Take absolute value of eigenvalues
    abs_eigenvalues = np.abs(eigenvalues)

    # Reconstruct matrix with absolute eigenvalues
    abs_A = eigenvectors @ np.diag(abs_eigenvalues) @ np.linalg.inv(eigenvectors)

    # Return real part (imaginary parts should be negligible for real symmetric input)
    if np.allclose(abs_A.imag, 0):
        return abs_A.real
    else:
        # If significant imaginary parts exist, return zeros as a fallback
        return np.zeros_like(A, dtype=float)


def get_bofill_H(
    pos_new: np.ndarray,
    pos_old: np.ndarray,
    grad_new: np.ndarray,
    grad_old: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Bofill's Hessian update algorithm (TS-BFGS method).

    Updates the Hessian matrix based on the change in positions and gradients
    between two consecutive steps. This is useful for approximating the Hessian
    without expensive explicit computation.

    The Bofill update combines the symmetric rank-1 (SR1) and Powell-symmetric-
    Broyden (PSB) updates in a way that is particularly effective for transition
    state searches.

    Args:
        pos_new (np.ndarray): Positions at time t+dt, shape (3N,) or (N, 3).
        pos_old (np.ndarray): Positions at time t, shape (3N,) or (N, 3).
        grad_new (np.ndarray): Gradients at time t+dt, shape (3N,) or (N, 3).
        grad_old (np.ndarray): Gradients at time t, shape (3N,) or (N, 3).
        H (np.ndarray): Hessian matrix at time t, shape (3N, 3N).

    Returns:
        np.ndarray: Updated Hessian matrix at time t+dt, shape (3N, 3N).

    Example:
        >>> # Simple 2D quadratic: f(x,y) = x^2 + 2*y^2
        >>> H_old = np.array([[2.0, 0.0], [0.0, 4.0]])
        >>> pos_old = np.array([1.0, 1.0])
        >>> pos_new = np.array([0.9, 0.8])
        >>> grad_old = np.array([2.0, 4.0])  # gradient at pos_old
        >>> grad_new = np.array([1.8, 3.2])  # gradient at pos_new
        >>> H_new = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_old)

    Note:
        - Positions and gradients can be passed as either flat (3N,) arrays
          or (N, 3) arrays; they will be flattened internally.
        - The update assumes that the step size is not too large; very large
          steps may lead to poor Hessian approximations.
        - For best results, the initial Hessian H should be a reasonable
          approximation (e.g., from a previous explicit calculation).
    """
    # Flatten inputs to column vectors
    d_vec = (pos_new - pos_old).reshape(-1, 1)  # 3N x 1
    y_vec = (grad_new - grad_old).reshape(-1, 1)  # 3N x 1

    # Compute spectral absolute value of H
    abs_H = np.real(_get_spectral_abs(H))  # 3N x 3N

    # Compute the update matrix E = A + B - C
    # Following the TS-BFGS formulation from the reference

    # A1 = y - H*d (the secant condition residual)
    A1 = y_vec - (H @ d_vec)  # 3N x 1

    # Compute scalar products
    A21 = (y_vec.T @ d_vec).item()  # y^T * d
    A22 = (d_vec.T @ abs_H @ d_vec).item()  # d^T * |H| * d

    # A2 = A21*y + A22*(|H|*d)
    A2 = A21 * y_vec + A22 * (abs_H @ d_vec)  # 3N x 1

    # A3 = A21^2 + A22^2 (normalization factor)
    A3 = A21**2 + A22**2

    # Avoid division by zero
    if A3 < 1e-20:
        return H.copy()

    # Compute update terms
    A = (1.0 / A3) * (A1 @ A2.T)  # 3N x 3N
    B = (1.0 / A3) * (A2 @ A1.T)  # 3N x 3N

    # C term
    C1 = (y_vec.T @ d_vec).item() - (d_vec.T @ H @ d_vec).item()
    C = (C1 / A3**2) * (A2 @ A2.T)  # 3N x 3N

    E = A + B - C

    return H + E


def get_sr1_H(
    pos_new: np.ndarray,
    pos_old: np.ndarray,
    grad_new: np.ndarray,
    grad_old: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Symmetric Rank-1 (SR1) Hessian update.

    A simpler alternative to the Bofill update. The SR1 update can capture
    negative curvature, making it useful for transition state searches.

    Args:
        pos_new: Positions at time t+dt.
        pos_old: Positions at time t.
        grad_new: Gradients at time t+dt.
        grad_old: Gradients at time t.
        H: Hessian matrix at time t.

    Returns:
        Updated Hessian matrix.

    Note:
        The SR1 update is skipped if the denominator is too small,
        returning the unchanged Hessian.
    """
    d_vec = (pos_new - pos_old).reshape(-1, 1)
    y_vec = (grad_new - grad_old).reshape(-1, 1)

    # SR1 update: H_new = H + (y - H*d)(y - H*d)^T / (y - H*d)^T * d
    residual = y_vec - H @ d_vec

    denominator = (residual.T @ d_vec).item()

    # Skip update if denominator is too small (numerical stability)
    if abs(denominator) < 1e-12:
        return H.copy()

    return H + (residual @ residual.T) / denominator


def get_bfgs_H(
    pos_new: np.ndarray,
    pos_old: np.ndarray,
    grad_new: np.ndarray,
    grad_old: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) Hessian update.

    The standard BFGS update maintains positive definiteness of the Hessian,
    making it suitable for minimization but not for transition state searches
    where negative eigenvalues are expected.

    Args:
        pos_new: Positions at time t+dt.
        pos_old: Positions at time t.
        grad_new: Gradients at time t+dt.
        grad_old: Gradients at time t.
        H: Hessian matrix at time t.

    Returns:
        Updated Hessian matrix.

    Note:
        For transition state searches, use ``get_bofill_H`` or ``get_sr1_H`` instead,
        as BFGS enforces positive definiteness.
    """
    d_vec = (pos_new - pos_old).reshape(-1, 1)
    y_vec = (grad_new - grad_old).reshape(-1, 1)

    rho = (y_vec.T @ d_vec).item()

    # Skip update if curvature condition not satisfied
    if rho < 1e-12:
        return H.copy()

    rho = 1.0 / rho
    n = H.shape[0]
    I = np.eye(n)

    V = I - rho * (d_vec @ y_vec.T)
    H_new = V.T @ H @ V + rho * (y_vec @ y_vec.T)

    return H_new
