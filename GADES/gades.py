import numpy as np
from openmm import CustomExternalForce

from utils import clamp_force_magnitudes as fclamp

def getGADESBiasForce():
    """
    Function to create the custom force for GADES

    Returns:
        CustomExternalForce: OpenMM custom force class generated for GADES biasing
    """
    force = CustomExternalForce("fx*x+fy*y+fz*z")
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    return force


class GADESForceUpdater(object):
    def __init__(self, biased_force, bias_atom_indices, hess_func, clamp_magnitude, kappa, interval):
        """
        Class to update biased forces in a molecular simulation using Gentlest Ascent Dynamics (GAD).
        
        This updater periodically recalculates the biasing force based on the softest mode
        (smallest eigenvalue direction) from the system's Hessian matrix and applies it to
        specified atoms.

        Parameters
        ----------
        biased_force : openmm.CustomExternalForce or similar
            The OpenMM force object where the biased forces will be applied.
        bias_atom_indices : array-like
            Indices of the atoms to which the biased forces are applied.
        hess_func : callable
            Function that computes the Hessian matrix given the system, positions, 
            selected atom indices, a displacement tolerance, and platform.
        clamp_magnitude : float
            Maximum allowed magnitude for each biased force component.
        kappa : float
            Scaling factor for the biased force.
        interval : int
            Number of simulation steps between force updates.
        """
        self.biased_force = biased_force
        self.bias_atom_indices = bias_atom_indices
        self.hess_func = hess_func
        self.clamp_magnitude = clamp_magnitude
        self.interval = interval
        self.kappa = kappa
        self.hess_step_size = 1e-6

    def set_kappa(self, kappa):
        """
        Update the scaling factor for the biased force.

        Parameters
        ----------
        kappa : float
            New scaling factor.
        """
        self.kappa = kappa
        return None
    
    def set_hess_step_size(self, delta):
        """
        Update the displacement step size used in the Hessian calculation.

        Parameters
        ----------
        delta : float
            New displacement step size.
        """
        self.hess_step_size = delta
        return None
    
    def _get_gad_force(self, simulation):
        """
        Compute the Gentlest Acent Dynamics (GAD) force vector and direction for a molecular system.

        Parameters
        ----------
        sim : openmm.app.Simulation
            The OpenMM simulation object containing the current system state.
        bias_atom_indices : array-like
            Indices of the atoms to which the biasing force is applied.
        hess_func : callable
            Function that computes the Hessian matrix given the system, positions, 
            selected atom indices, a displacement tolerance, and platform.
        clamp_magnitude : float
            Maximum allowed magnitude for the biased force components; forces 
            exceeding this will be clamped.
        kappa : float, optional
            Scaling factor applied to the biased force (default is 0.9).

        Returns
        -------
        forces_b : np.ndarray
            The biased force vector, reshaped to match the system's position shape 
            (typically (N_atoms, 3)).

        Notes
        -----
        This function computes the Hessian of the system, extracts the eigenvector 
        associated with the smallest eigenvalue (softest mode), and constructs a 
        biased force vector aligned with this mode. The biased force is scaled, 
        clamped, and reshaped to match the atomic positions.

        """
        state = simulation.context.getState(getPositions=True, getForces=True)
        platform = simulation.context.getPlatform().getName()
        forces_u = state.getForces(asNumpy=True)
        positions = state.getPositions(asNumpy=True)
        hess = self.hess_func(simulation.system, positions, self.bias_atom_indices, self.hess_step_size, platform)
        w, v = np.linalg.eigh(hess)
        n = v[:, w.argsort()[0]]
        n /= np.linalg.norm(n)
        ## cast n back to the full position vector
        n_new = np.zeros_like(positions)
        n_new[self.bias_atom_indices] = n.reshape(-1, 3)
        forces_b = -np.dot(n_new.flatten(), forces_u.flatten()) * n_new.flatten() * self.kappa
        # clamping biased forces so their abs value is never larger than `clamp_magnitude`
        forces_b = fclamp(forces_b, self.clamp_magnitude)
        return forces_b.reshape(positions.shape)
        
    def describeNextReport(self, simulation):
        """
        Define the interval and required data for the next report.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The OpenMM simulation object (unused here but required by interface).

        Returns
        -------
        tuple
            A tuple of (interval, pos, vel, force, energy, volume) flags.
            Only interval is used; all data flags are False.
        """
        steps = self.interval - simulation.currentStep%self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        """
        Apply the computed biased forces at the current simulation step.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The OpenMM simulation object.
        state : openmm.State
            The current simulation state (unused here but required by interface).
        """
        print(f"[step {simulation.currentStep}] Updating GAD forces...", flush=True)
        biased_forces = self._get_gad_force(simulation)
        for i in self.bias_atom_indices:
            self.biased_force.setParticleParameters(i, i, tuple(biased_forces[i]))
        self.biased_force.updateParametersInContext(simulation.context)
