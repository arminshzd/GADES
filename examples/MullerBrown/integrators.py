"""
Integrators for Muller-Brown examples.

This module provides integrators for running molecular dynamics simulations
on the Muller-Brown potential surface.
"""

import numpy as np
from typing import Callable, Tuple


def baoab_langevin_integrator(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces_u: np.ndarray,
    forces_b: np.ndarray,
    mass: float,
    gamma: float,
    dt: float,
    kBT: float,
    force_function_u: Callable,
    force_function_b: Callable,
    n_steps: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    BAOAB Langevin integrator based on Leimkuhler and Matthews (2013).

    This integrator uses the BAOAB splitting scheme which provides excellent
    configurational sampling properties for Langevin dynamics.

    Args:
        positions: Initial positions (shape: [D,], where D is dimensionality).
        velocities: Initial velocities (shape: [D,]).
        forces_u: Initial unbiased forces (shape: [D,]).
        forces_b: Initial biased forces (shape: [D,]).
        mass: Mass of the particles (scalar).
        gamma: Friction coefficient (scalar).
        dt: Time step (scalar).
        kBT: Thermal energy (k_B * T) (scalar).
        force_function_u: Function to compute unbiased forces given positions.
        force_function_b: Function to compute biased forces given positions.
        n_steps: Number of simulation steps (scalar).

    Returns:
        Tuple of (positions, velocities, forces_u, forces_b) after n_steps.

    References:
        Leimkuhler, B., & Matthews, C. (2013). Applied Mathematics Research
        eXpress, 2013(1), 34-56. https://dx.doi.org/10.1093/amrx/abs010

    Example:
        >>> from GADES.potentials import muller_brown_force
        >>> pos = np.array([-0.5, 1.5])
        >>> vel = np.array([0.0, 0.0])
        >>> forces_u = muller_brown_force(pos)
        >>> forces_b = np.zeros(2)
        >>> pos, vel, f_u, f_b = baoab_langevin_integrator(
        ...     pos, vel, forces_u, forces_b,
        ...     mass=1.0, gamma=1.0, dt=0.001, kBT=1.0,
        ...     force_function_u=muller_brown_force,
        ...     force_function_b=lambda x: np.zeros(2),
        ...     n_steps=100
        ... )
    """
    dim = positions.shape[0]

    # Precompute constants
    c1 = np.exp(-gamma * dt)
    c3 = np.sqrt(kBT * (1 - c1**2))
    inv_mass = 1.0 / mass
    inv_mass_sqrt = 1.0 / np.sqrt(mass)

    # Make copies to avoid modifying input
    positions = positions.copy()
    velocities = velocities.copy()
    forces_u = forces_u.copy()
    forces_b = forces_b.copy()

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
