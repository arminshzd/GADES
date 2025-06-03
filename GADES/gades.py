import numpy as np
from openmm import CustomExternalForce, unit, CMMotionRemover

from utils import clamp_force_magnitudes as fclamp

def getGADESBiasForce(n_particles):
    """
    Function to create the custom force for GADES

    Returns:
        CustomExternalForce: OpenMM custom force class generated for GADES biasing
    """
    force = CustomExternalForce("fx*x+fy*y+fz*z")
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    for i in range(n_particles):
        force.addParticle(i, [0.0, 0.0, 0.0])
    force.setForceGroup(1)
    return force


class GADESForceUpdater(object):
    def __init__(self, biased_force, bias_atom_indices, hess_func, clamp_magnitude, kappa, interval, stability_interval=None):
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
        if interval < 100:
            print("[GADES| WARNING] Bias update interval must be larger than 100 steps to ensure system stability. Changing the frequency to 110 steps internally...")
            self.interval = 110
        else:
            self.interval = interval
        self.kappa = kappa
        self.hess_step_size = 1e-6
        self.check_stability = False
        self.is_biasing = False
        self.s_interval = stability_interval
        
        # post bias update check
        self.next_postbias_check_step = None

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
    
    def _is_stable(self, simulation):
        dof = 0
        system = simulation.system
        state = simulation.context.getState(getEnergy=True)
        for i in range(system.getNumParticles()):
            if system.getParticleMass(i) > 0*unit.dalton:
                dof += 3
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if system.getParticleMass(p1) > 0*unit.dalton or system.getParticleMass(p2) > 0*unit.dalton:
                dof -= 1
        if any(type(system.getForce(i)) == CMMotionRemover for i in range(system.getNumForces())):
            dof -= 3
        temperature = (2*state.getKineticEnergy()/(dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)
        target_temperature = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
        if abs(temperature - target_temperature) > 50:
            return False
        return True
    
    def _get_gad_force(self, simulation):
        """
        Compute the Gentlest Ascent Dynamics (GAD) force vector and direction for a molecular system.

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
        forces_u = state.getForces(asNumpy=True)[self.bias_atom_indices, :]
        positions = state.getPositions(asNumpy=True)
        hess = self.hess_func(simulation.system, positions, self.bias_atom_indices, self.hess_step_size, platform)
        w, v = np.linalg.eigh(hess)
        n = v[:, w.argsort()[0]]
        n /= np.linalg.norm(n)
        forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
        # clamping biased forces so their abs value is never larger than `clamp_magnitude`
        forces_b = fclamp(forces_b, self.clamp_magnitude)
        return forces_b.reshape(forces_u.shape)
        
    def describeNextReport(self, simulation):
        """
        Define the interval and required data for the next report.

        Parameters
        ----------
        simulation : openmm.app.Simulation

        Returns
        -------
        tuple
            (steps until next report, pos, vel, force, energy, volume)
        """
        step = simulation.currentStep

        # Compute time to each type of report
        steps_to_check = (
            self.s_interval - step % self.s_interval
            if self.s_interval else np.inf
        )
        steps_to_bias = self.interval - step % self.interval

        if self.next_postbias_check_step is not None:
            steps_to_postbias = max(self.next_postbias_check_step - step, 0)
        else:
            steps_to_postbias = np.inf

        # Choose the next event
        steps = min(steps_to_bias, steps_to_check, steps_to_postbias)

        # Set flags *before* return
        self.is_biasing = (steps == steps_to_bias)
        is_forced_check = (steps == steps_to_postbias)
        is_regular_check = (steps == steps_to_check)

        self.check_stability = is_forced_check or is_regular_check

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
        step = simulation.currentStep

        def remove_bias():
            for idx in self.bias_atom_indices:
                self.biased_force.setParticleParameters(idx, idx, (0.0, 0.0, 0.0))

        def apply_bias():
            biased_forces = self._get_gad_force(simulation)
            for i, idx in enumerate(self.bias_atom_indices):
                self.biased_force.setParticleParameters(idx, idx, tuple(biased_forces[i]))

        if self.check_stability:
            is_stable = self._is_stable(simulation)
            if not is_stable:
                print(f"[GADES | step {step}] System is unstable: Removing bias until next cycle...", flush=True)
                remove_bias()
            elif self.is_biasing:
                print(f"[GADES | step {step}] Updating bias forces...", flush=True)
                apply_bias()
                self.next_postbias_check_step = step + 100

            self.biased_force.updateParametersInContext(simulation.context)
            self.check_stability = False
            self.is_biasing = False
            if step == self.next_postbias_check_step:
                self.next_postbias_check_step = None
            return None

        if self.is_biasing:
            print(f"[GADES | step {step}] Updating bias forces...", flush=True)
            apply_bias()
            self.biased_force.updateParametersInContext(simulation.context)
            self.is_biasing = False
            self.next_postbias_check_step = step + 100
            return None

        # If neither flag is True, do nothing
        return None