"""
Test potentials for GADES.

This module provides analytical potential energy surfaces for testing and
demonstration purposes. These are implemented in pure NumPy for simplicity.
"""

import numpy as np
from typing import Union


def muller_brown_potential(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Muller-Brown potential energy surface.

    The Muller-Brown potential is a 2D surface with three minima and two
    saddle points, commonly used for testing rare event sampling methods.

    Args:
        x: Position array of shape (2,) for single point or (N, 2) for batch.

    Returns:
        Potential energy value(s). Scalar for single point, (N,) array for batch.

    References:
        Muller, K., & Brown, L. D. (1979). Theoretica chimica acta, 53(1), 75-93.
    """
    x = np.atleast_2d(x)

    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    x_coord = x[:, 0:1]  # (N, 1)
    y_coord = x[:, 1:2]  # (N, 1)

    exponent = (a * (x_coord - x0) ** 2 +
                b * (x_coord - x0) * (y_coord - y0) +
                c * (y_coord - y0) ** 2)

    z = np.sum(A * np.exp(exponent), axis=1)

    return z[0] if z.shape[0] == 1 else z


def muller_brown_force(x: np.ndarray) -> np.ndarray:
    """
    Muller-Brown forces (negative gradient of potential).

    Args:
        x: Position array of shape (2,) for single point or (N, 2) for batch.

    Returns:
        Force array of shape (2,) for single point or (N, 2) for batch.
    """
    x = np.atleast_2d(x)

    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    x_coord = x[:, 0:1]  # (N, 1)
    y_coord = x[:, 1:2]  # (N, 1)

    exponent = (a * (x_coord - x0) ** 2 +
                b * (x_coord - x0) * (y_coord - y0) +
                c * (y_coord - y0) ** 2)

    exp_terms = A * np.exp(exponent)  # (N, 4)

    # dV/dx = sum_i A_i * exp(...) * (2*a_i*(x-x0_i) + b_i*(y-y0_i))
    dV_dx = np.sum(exp_terms * (2 * a * (x_coord - x0) + b * (y_coord - y0)), axis=1)

    # dV/dy = sum_i A_i * exp(...) * (b_i*(x-x0_i) + 2*c_i*(y-y0_i))
    dV_dy = np.sum(exp_terms * (b * (x_coord - x0) + 2 * c * (y_coord - y0)), axis=1)

    forces = -np.column_stack([dV_dx, dV_dy])

    return forces[0] if forces.shape[0] == 1 else forces


def muller_brown_hess(x: np.ndarray) -> np.ndarray:
    """
    Muller-Brown Hessian matrix (second derivatives of potential).

    Args:
        x: Position array of shape (2,) for single point or (N, 2) for batch.

    Returns:
        Hessian matrix of shape (2, 2) for single point or (N, 2, 2) for batch.
    """
    x = np.atleast_2d(x)
    N = x.shape[0]

    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    x_coord = x[:, 0:1]  # (N, 1)
    y_coord = x[:, 1:2]  # (N, 1)

    dx = x_coord - x0  # (N, 4)
    dy = y_coord - y0  # (N, 4)

    exponent = a * dx ** 2 + b * dx * dy + c * dy ** 2
    exp_terms = A * np.exp(exponent)  # (N, 4)

    # First derivatives of exponent
    dexp_dx = 2 * a * dx + b * dy  # (N, 4)
    dexp_dy = b * dx + 2 * c * dy  # (N, 4)

    # Second derivatives: d²V/dx² = sum_i A_i * exp(...) * (dexp_dx_i² + 2*a_i)
    d2V_dx2 = np.sum(exp_terms * (dexp_dx ** 2 + 2 * a), axis=1)

    # d²V/dy² = sum_i A_i * exp(...) * (dexp_dy_i² + 2*c_i)
    d2V_dy2 = np.sum(exp_terms * (dexp_dy ** 2 + 2 * c), axis=1)

    # d²V/dxdy = sum_i A_i * exp(...) * (dexp_dx_i * dexp_dy_i + b_i)
    d2V_dxdy = np.sum(exp_terms * (dexp_dx * dexp_dy + b), axis=1)

    hessians = np.zeros((N, 2, 2))
    hessians[:, 0, 0] = d2V_dx2
    hessians[:, 0, 1] = d2V_dxdy
    hessians[:, 1, 0] = d2V_dxdy
    hessians[:, 1, 1] = d2V_dy2

    return hessians[0] if N == 1 else hessians


# Convenience aliases matching old API (for backward compatibility)
def muller_brown_potential_base(x: np.ndarray) -> float:
    """Single-point Muller-Brown potential (backward compatibility)."""
    return float(muller_brown_potential(x))


def muller_brown_force_base(x: np.ndarray) -> np.ndarray:
    """Single-point Muller-Brown force (backward compatibility)."""
    return muller_brown_force(x)


def muller_brown_hess_base(x: np.ndarray) -> np.ndarray:
    """Single-point Muller-Brown Hessian (backward compatibility)."""
    return muller_brown_hess(x)
