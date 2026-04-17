"""
Hessian-Vector Product (HVP) computation via finite differences.

This module provides matrix-free methods for computing Hessian-vector products,
enabling GADES to scale to large molecular systems without forming the full
Hessian matrix explicitly.

The key insight is that for any vector v:
    H @ v ≈ (F(x - εv) - F(x + εv)) / (2ε)

where F is the force (negative gradient of energy) and ε is a small displacement.
This requires only 2 force evaluations per HVP, regardless of system size.
"""

from typing import Callable

import numpy as np


def finite_difference_hvp(
    force_func: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    vector: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute Hessian-vector product using central finite differences.

    Given a force function F(x) = -∇E(x), computes H @ v where H is the Hessian
    of the energy E. Uses central differences for O(ε²) accuracy.

    Args:
        force_func: Callable that takes positions (N, 3) or (M,) and returns
            forces with the same shape. Forces should be the negative gradient
            of energy (F = -∇E).
        positions: Current atomic positions, shape (N, 3) or flattened (3N,).
        vector: Direction vector for HVP, same shape as positions or (3N,).
        epsilon: Finite difference step size. Default 1e-5 balances truncation
            and round-off errors for typical molecular systems.

    Returns:
        Hessian-vector product H @ v, shape (3N,) flattened.

    Example:
        >>> from GADES.potentials import muller_brown_force
        >>> pos = np.array([0.0, 0.5])
        >>> v = np.array([1.0, 0.0])
        >>> hvp = finite_difference_hvp(muller_brown_force, pos, v)

    Notes:
        - The force function should return F = -∇E (standard MD convention)
        - For the Hessian H = ∇²E, we have: H @ v = -∇F @ v
        - Central difference: H @ v ≈ (F(x-εv) - F(x+εv)) / (2ε)
    """
    # Flatten inputs
    pos_flat = positions.reshape(-1)
    v_flat = vector.reshape(-1)

    if pos_flat.shape != v_flat.shape:
        raise ValueError(
            f"Position shape {pos_flat.shape} != vector shape {v_flat.shape}"
        )

    # Normalize direction for numerical stability
    v_norm = np.linalg.norm(v_flat)
    if v_norm < 1e-14:
        # Zero vector, return zero HVP
        return np.zeros_like(pos_flat)

    v_unit = v_flat / v_norm

    # Compute displaced positions
    pos_plus = pos_flat + epsilon * v_unit
    pos_minus = pos_flat - epsilon * v_unit

    # Get forces at displaced positions
    # Reshape to original shape for force function if needed
    original_shape = positions.shape
    force_plus = force_func(pos_plus.reshape(original_shape)).reshape(-1)
    force_minus = force_func(pos_minus.reshape(original_shape)).reshape(-1)

    # Central difference: H @ v = (F(x-εv) - F(x+εv)) / (2ε)
    # This comes from: H = ∇²E = -∇F, so H @ v = -dF/dx @ v
    hvp_unit = (force_minus - force_plus) / (2 * epsilon)

    # Scale back by original vector norm
    return hvp_unit * v_norm


def finite_difference_hvp_richardson(
    force_func: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    vector: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Compute Hessian-vector product using Richardson extrapolation.

    Uses two step sizes (ε and ε/2) and Richardson extrapolation to achieve
    O(ε⁴) accuracy, eliminating the leading O(ε²) error term.

    Args:
        force_func: Callable that takes positions and returns forces.
        positions: Current atomic positions, shape (N, 3) or (3N,).
        vector: Direction vector for HVP.
        epsilon: Base finite difference step size. Can be larger than
            simple central difference since Richardson cancels leading errors.

    Returns:
        Hessian-vector product H @ v, shape (3N,).

    Notes:
        Richardson extrapolation formula:
            HVP_accurate = (4 * HVP(ε/2) - HVP(ε)) / 3

        This requires 4 force evaluations instead of 2, but provides
        significantly better accuracy for the same base step size.
    """
    # Compute HVP at two step sizes
    hvp_h = finite_difference_hvp(force_func, positions, vector, epsilon)
    hvp_h2 = finite_difference_hvp(force_func, positions, vector, epsilon / 2)

    # Richardson extrapolation: cancels O(ε²) error term
    # For central difference, error is O(ε²), so:
    # HVP(h) = HVP_true + c*h² + O(h⁴)
    # HVP(h/2) = HVP_true + c*(h/2)² + O(h⁴) = HVP_true + c*h²/4 + O(h⁴)
    # => HVP_true = (4*HVP(h/2) - HVP(h)) / 3
    return (4 * hvp_h2 - hvp_h) / 3


def finite_difference_hvp_forward(
    force_func: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    vector: np.ndarray,
    force_at_positions: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute Hessian-vector product using forward finite differences.

    This variant reuses a pre-computed force at the current position,
    requiring only 1 additional force evaluation instead of 2. However,
    it has only O(ε) accuracy compared to O(ε²) for central differences.

    Args:
        force_func: Callable that takes positions and returns forces.
        positions: Current atomic positions.
        vector: Direction vector for HVP.
        force_at_positions: Pre-computed forces at current positions.
        epsilon: Finite difference step size.

    Returns:
        Hessian-vector product H @ v.

    Notes:
        Forward difference: H @ v ≈ (F(x) - F(x + εv)) / ε
        Less accurate than central difference but saves one force evaluation.
        Useful when force at current position is already available.
    """
    pos_flat = positions.reshape(-1)
    v_flat = vector.reshape(-1)
    f0_flat = force_at_positions.reshape(-1)

    v_norm = np.linalg.norm(v_flat)
    if v_norm < 1e-14:
        return np.zeros_like(pos_flat)

    v_unit = v_flat / v_norm

    pos_plus = pos_flat + epsilon * v_unit
    force_plus = force_func(pos_plus.reshape(positions.shape)).reshape(-1)

    # Forward difference
    hvp_unit = (f0_flat - force_plus) / epsilon

    return hvp_unit * v_norm
