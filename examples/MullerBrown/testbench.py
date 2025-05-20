import sys
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# add GADES to the path
GADES_path = os.path.abspath(os.path.join(os.getcwd(), "GADES"))
sys.path.append(GADES_path)
from utils import muller_brown_force_base as get_F
from utils import muller_brown_hess_base as get_H
from bofill import get_bofill_H

def muller_brown_gad_force_base(positions, positions_old, forces_u, forces_u_old, step, kappa=0.9):
    global last_h, queue_fb
    """ GADES forces for the Muller-Brown potential at `position` calculated using AD. Calculates the total forces, then finds the most-negative eigenvalue and the corresponding eigenvector of the Hessian and returns negative `kappa` times the force projected in the eigenvector direction as the biasing force.

    Args:
        position (jax.ndarray): (2, ) position
        kappa (float): GAD intensity parameter. Determines how much of the GAD force is used for biasing. `kappa=1` is 100% and `kappa=0` is none.

    Returns:
        jax.ndarray: (2, ) GAD bias forces vector
    """

    # unbiased forces

    if (step % 100) == 0:
        # biased forces (softened by kappa)
        h = get_H(positions)
    else:
        h = get_bofill_H(positions, positions_old, -forces_u, -forces_u_old, last_h)
    w, v = jnp.linalg.eigh(h)
    n = v[:,0]
    n = n/jnp.sqrt(jnp.dot(n,n))
    forces_b = -jnp.dot(forces_u,n)*n
    
    # update the queues
    last_h = h
    queue_fb = jnp.roll(queue_fb, -1, axis=0)
    queue_fb = queue_fb.at[-1].set(forces_b)
    
    # calculate the average of the last 10 biasing events
    forces_b_avg = queue_fb.mean(axis=0)
    forces_b_avg *= kappa
    
    return forces_b_avg

def baoab_langevin_integrator(positions, velocities, forces_u, forces_b, mass, gamma, dt, kBT, force_function_u, force_function_b, step):
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
    positions_old = positions.copy()
    forces_u_old = forces_u.copy()

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
    forces_b = force_function_b(positions, positions_old, forces_u, forces_u_old, step)
    forces = forces_u + forces_b
    velocities += 0.5 * dt * inv_mass * forces

    return positions, velocities, forces_u, forces_b

# Parameters
x = jnp.array([0.7,0])
v = jnp.zeros_like(x)
f = get_F(x)
queue_fb = np.full((10, x.shape[0]), 0.0)
last_h = None
fb = muller_brown_gad_force_base(x, None, f, None, 0, kappa=0.6)
mass = 1.0
gamma = 1.0
dt = 0.01
n_steps = 20
kBT = 2.0  # Thermal energy

# prepare storage arrays
traj_p = np.full((n_steps+1, x.shape[0]), np.nan)
traj_v = np.full((n_steps+1, x.shape[0]), np.nan)
traj_fu = np.full((n_steps+1, x.shape[0]), np.nan)
traj_fb = np.full((n_steps+1, x.shape[0]), np.nan)

traj_p[0,:] = x
traj_v[0,:] = v
traj_fu[0,:] = f
traj_fb[0,:] = fb

# Run the integrator
for i in range(1,n_steps+1):

  x, v, f, fb = baoab_langevin_integrator(
    x, v, f, fb, mass, gamma, dt, kBT, get_F, muller_brown_gad_force_base, i
  )

  traj_p[i,:] = x
  traj_v[i,:] = v
  traj_fu[i,:] = f
  traj_fb[i,:] = fb

# Convert trajectory to numpy array for analysis
traj_p = np.array(traj_p)
traj_v = np.array(traj_v)
traj_fu = np.array(traj_fu)
traj_fb = np.array(traj_fb)