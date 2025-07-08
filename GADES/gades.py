import atexit

import numpy as np
from openmm import CustomExternalForce, unit, CMMotionRemover

from utils import clamp_force_magnitudes as fclamp

def getGADESBiasForce(n_particles: int) -> CustomExternalForce:
    """
    Create and return a CustomExternalForce for applying GADES biasing forces.

    Parameters
    ----------
    n_particles : int
        Number of particles in the system.

    Returns
    -------
    CustomExternalForce
        Configured OpenMM force object for GADES biasing.
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
    def __init__(self, biased_force, bias_atom_indices, hess_func, 
                 clamp_magnitude: float, kappa: float, interval: int, 
                 stability_interval: int = None, logfile_prefix: str = None):
        """
        Initialize the GADESForceUpdater for periodically applying Gentlest Ascent Dynamics (GAD) bias forces.

        This class identifies the softest mode of the system (lowest eigenvalue of the Hessian),
        constructs a directional force along that mode, and applies it to a selected set of atoms.
        It also performs periodic stability checks and logs key diagnostic information.

        Parameters
        ----------
        biased_force : openmm.CustomExternalForce
            The OpenMM force object to which GADES bias forces are applied (should be created using `getGADESBiasForce()`).
        
        bias_atom_indices : array-like of int
            Atom indices that will receive the bias force updates.
        
        hess_func : callable
            A function that returns the Hessian matrix given (system, positions, atom_indices, step_size, platform).
        
        clamp_magnitude : float
            Maximum allowed magnitude for each component of the bias force; used to prevent unphysical updates.
        
        kappa : float
            Scaling factor applied to the bias force along the softest mode.
        
        interval : int
            Number of simulation steps between each bias force update. If set below 100, it will be internally overridden to 110.
        
        stability_interval : int, optional
            Frequency (in steps) at which to perform stability checks (based on kinetic temperature).
            If None, only post-bias stability checks are used.
        
        logfile_prefix : str, optional
            If provided, enables logging of:
                - Softest eigenvectors:    <prefix>_evec.log
                - Corresponding eigenvalues: <prefix>_eval.log
                - Biased atom positions:   <prefix>_biased_atoms.xyz (in XYZ format)
        """
        self.biased_force = biased_force
        self.bias_atom_indices = bias_atom_indices
        self.hess_func = hess_func
        self.clamp_magnitude = clamp_magnitude
        if interval < 100:
            print("\033[1;33m[GADES| WARNING] Bias update interval must be larger than 100 steps to ensure system stability. Changing the frequency to 110 steps internally...\033[0m")
            self.interval = 110
        else:
            self.interval = interval
        self.kappa = kappa
        self.hess_step_size = 1e-5
        self.check_stability = False
        self.is_biasing = False
        self.s_interval = stability_interval
        self.eigenvec = None
        
        # post bias update check
        self.next_postbias_check_step = None

        # logging
        self.atom_symbols = None
        self.logfile_prefix = logfile_prefix
        self._evec_log = None
        self._eval_log = None
        self._xyz_log = None

        if logfile_prefix is not None:
            # register with atexit for safe handling of log files
            atexit.register(self._close_logs)
            
            self._evec_log = open(f"{logfile_prefix}_evec.log", "w")
            self._evec_log.write("# Softest-mode eigenvector at each step (one per line)\n")
            self._evec_log.write("# Columns: step, eigenvector components (flattened)\n")
            self._evec_log.flush()
            self._eval_log = open(f"{logfile_prefix}_eval.log", "w")
            self._eval_log.write("# Sorted eigenvalue spectrum at each step (one line per frame)\n")
            self._eval_log.write("# Columns: step, eigenvalues from smallest to largest\n")
            self._eval_log.flush()

            self._xyz_log  = open(f"{logfile_prefix}_biased_atoms.xyz", "w")
            self._xyz_log.write("# Trajectory of biased atoms only\n")
            self._xyz_log.write("# Each frame follows XYZ format: N_atoms, comment, atom lines\n")
            self._xyz_log.write("# Coordinates are in nanometers; atoms are labeled 'C' by default\n")
            self._xyz_log.flush()
            

    def set_kappa(self, kappa: float) -> None:
        """
        Update the scaling factor for the biased force.

        Parameters
        ----------
        kappa : float
            New scaling factor.
        """
        self.kappa = kappa
        return None
    
    def set_hess_step_size(self, delta: float) -> None:
        """
        Update the displacement step size used in the Hessian calculation.

        Parameters
        ----------
        delta : float
            New displacement step size.
        """
        self.hess_step_size = delta
        return None

    def set_eigenvec(self, vec: np.ndarray) -> None:
        """
        Provide a precomputed softest eigenvector to be used during biasing.

        Parameters
        ----------
        vec : np.ndarray
            The softest-mode eigenvector.
        """
        self.eigenvec = vec
        return None
    
    def _is_stable(self, simulation) -> bool:
        """
        Determine whether the simulation is thermodynamically stable.

        Parameters
        ----------
        simulation : openmm.app.Simulation

        Returns
        -------
        bool
            True if within temperature threshold; False otherwise.
        """
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
    
    def _ensure_atom_symbols(self, simulation) -> None:
        """
        Lazily initializes atom symbols based on the simulation topology.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation from which to extract atom symbols.

        Sets
        -----
        self.atom_symbols : list of str
            Atomic symbols corresponding to `bias_atom_indices`.
        """
        if self.atom_symbols is None:
            atom_list = list(simulation.topology.atoms())
            self.atom_symbols = [
                atom_list[i].element.symbol if atom_list[i].element is not None else "X"
                for i in self.bias_atom_indices
            ]
    
    def _get_gad_force(self, simulation) -> np.ndarray:
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
        # because the StepTuner is running before the ForceUpdater, We don't
        # need to re-calculate the softest eigenvector anymore. just use the one
        # that comes from stepTuner.
        state = simulation.context.getState(getPositions=True, getForces=True)
        if self.eigenvec is None:
            platform = simulation.context.getPlatform().getName()
            positions = state.getPositions(asNumpy=True)
            hess = self.hess_func(simulation.system, positions, self.bias_atom_indices, self.hess_step_size, platform)
            w, v = np.linalg.eigh(hess)
            w_sorted = w.argsort()
            n = v[:, w_sorted[0]]
        else:
            n = self.eigenvec
        n /= np.linalg.norm(n)
        forces_u = state.getForces(asNumpy=True)[self.bias_atom_indices, :]
        forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
        # clamping biased forces so their abs value is never larger than `clamp_magnitude`
        forces_b = fclamp(forces_b, self.clamp_magnitude)

         # Logging
        if self._evec_log is not None:
            self._evec_log.write(f"{simulation.currentStep} " + " ".join(map(str, n)) + "\n")
            self._evec_log.flush()
        if self._eval_log is not None:
            self._eval_log.write(f"{simulation.currentStep} " + " ".join(map(str, w[w_sorted])) + "\n")
            self._eval_log.flush()
        if self._xyz_log is not None:
            pos_nm = positions[self.bias_atom_indices, :].value_in_unit(unit.nanometer)
            self._xyz_log.write(f"{len(self.bias_atom_indices)}\n")
            self._xyz_log.write(f"Step {simulation.currentStep}\n")
            for symbol, coord in zip(self.atom_symbols, pos_nm):
                x, y, z = coord
                self._xyz_log.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
            self._xyz_log.flush()
        
        return forces_b.reshape(forces_u.shape)
        
    def describeNextReport(self, simulation) -> tuple:
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

        # Extract atom symbols on the first call for logging
        self._ensure_atom_symbols(simulation)

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

    def report(self, simulation, state) -> None:
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

        # Defensive fallback in case describeNextReport hasn't been called yet
        self._ensure_atom_symbols(simulation)

        def remove_bias():
            for idx in self.bias_atom_indices:
                self.biased_force.setParticleParameters(idx, idx, (0.0, 0.0, 0.0))

        def apply_bias():
            biased_forces = self._get_gad_force(simulation)
            for i, idx in enumerate(self.bias_atom_indices):
                self.biased_force.setParticleParameters(idx, idx, tuple(biased_forces[i]))
            self.eigenvec = None
            self.next_postbias_check_step = step + 100

        if self.check_stability:
            is_stable = self._is_stable(simulation)
            if not is_stable:
                print(f"\033[1;31m[GADES | step {step}] System is unstable: Removing bias until next cycle...\033[0m", flush=True)
                remove_bias()
            elif self.is_biasing:
                print(f"\033[1;32m[GADES | step {step}] Updating bias forces...\033[0m", flush=True)
                apply_bias()
                

            self.biased_force.updateParametersInContext(simulation.context)
            self.check_stability = False
            self.is_biasing = False
            if step == self.next_postbias_check_step:
                self.next_postbias_check_step = None
            return None

        if self.is_biasing:
            print(f"\033[1;32m[GADES | step {step}] Updating bias forces...\033[0m", flush=True)
            apply_bias()
            self.biased_force.updateParametersInContext(simulation.context)
            self.is_biasing = False
            return None

        # If neither flag is True, do nothing
        return None
    
    def _close_logs(self) -> None:
        """
        Method to close the log files safely. Registered with `atexit` and called by `__del__`
        """
        for attr in ("_evec_log", "_eval_log", "_xyz_log"):
            f = getattr(self, attr, None)
            if f is not None and not f.closed:
                try:
                    f.close()
                except Exception:
                    pass
    
    def __del__(self) -> None:
        """
        Clean up the log files when the object is garbage-collected
        """
        self._close_logs()

class StepSizeTuner(object):
    def __init__(self, gades_updater, 
                 tolerance: float = 0.05,
                 step_size_list: list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]):
        """
        Reporter that dynamically selects the largest step size for Hessian finite differences
        by checking cosine similarity between eigenvectors from 2- and 3-point extrapolation.

        Parameters
        ----------
        gades_updater : GADESForceUpdater
            Instance whose step size and eigenvector will be updated.
        tolerance : float, optional
            Allowed deviation from cosine similarity of 1.0.
        step_size_list : list of float, optional
            Candidate step sizes, in descending order.
        """
        self.gades_updater = gades_updater
        self.reportInterval = self.gades_updater.interval
        self.hess_func = self.gades_updater.hess_func
        self.atom_indices = self.gades_updater.bias_atom_indices
        self.tolerance = tolerance
        self.step_sizes = step_size_list
        self.eigenvec = None

    def _get_softest_mode(self, hessian: np.ndarray) -> np.ndarray:
        """
        Extract the eigenvector associated with the most negative eigenvalue.

        Parameters
        ----------
        hessian : np.ndarray

        Returns
        -------
        np.ndarray
            Softest-mode eigenvector.
        """
        evals, evecs = np.linalg.eigh(hessian)
        return evecs[:, np.argmin(evals)]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Parameters
        ----------
        v1 : np.ndarray
        v2 : np.ndarray

        Returns
        -------
        float
            Cosine similarity.
        """
        return np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))

    def _find_valid_step_size(self, system, positions, platform: str) -> float:
        """
        Find the largest step size for which the softest mode is consistent between
        2- and 3-point Richardson extrapolation (within cosine similarity tolerance).

        Parameters
        ----------
        system : openmm.System
        positions : openmm.Vec3 array or Quantity
        platform : str

        Returns
        -------
        float or None
            Valid step size or None if none found.
        """
        for eps in self.step_sizes:
            hess_2 = self.hess_func(system, positions, self.atom_indices, eps, platform, [1.0, 0.5])
            hess_3 = self.hess_func(system, positions, self.atom_indices, eps, platform, [1.0, 0.5, 0.25])

            vec2 = self._get_softest_mode(hess_2)
            vec3 = self._get_softest_mode(hess_3)

            sim = self._cosine_similarity(vec2, vec3)
            if abs(sim - 1.0) <= self.tolerance:
                self.eigenvec = vec2
                return eps
        return None

    def describeNextReport(self, simulation) -> tuple:
        """
        Return number of steps until next report and required data.

        Parameters
        ----------
        simulation : openmm.app.Simulation

        Returns
        -------
        tuple
            (steps, pos, vel, force, energy, volume)
        """
        stepsUntilNextReport = self.reportInterval - (simulation.currentStep % self.reportInterval or self.reportInterval)
        return (stepsUntilNextReport, True, False, False, False, False)
    
    def report(self, simulation, state) -> None:
        """
        Evaluate optimal step size and softest eigenvector, and update GADES updater.

        Parameters
        ----------
        simulation : openmm.app.Simulation
        state : openmm.State
        """
        platform = simulation.context.getPlatform().getName()
        positions = state.getPositions(asNumpy=True)
        system = simulation.system

        new_eps = self._find_valid_step_size(system, positions, platform)
        if new_eps is not None:
            self.gades_updater.set_hess_step_size(new_eps)
            self.gades_updater.set_eigenvec(self.eigenvec)
            print(f"\033[1;32m[HessianStepSizeReporter] Updated epsilon to {new_eps}\033[0m")
        else:
            print("\033[1;33m[HessianStepSizeReporter] No suitable epsilon found.\033[0m")
