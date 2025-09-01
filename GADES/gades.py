from typing import Sequence, Callable, Optional
import atexit

import numpy as np
import openmm
import openmm.app
from openmm import CustomExternalForce, unit, CMMotionRemover

from .utils import clamp_force_magnitudes as fclamp

def getGADESBiasForce(n_particles: int) -> CustomExternalForce:
    """
    Create a custom OpenMM force object for GADES biasing.

    This function constructs an OpenMM `CustomExternalForce` that applies
    per-particle forces in the form:

    $$F(x, y, z) = f_x * x + f_y * y + f_z * z$$

    where `fx`, `fy`, and `fz` are per-particle parameters that can be updated
    during a simulation. The force is assigned to group `1` so that it can be
    easily separated from other forces in analysis or reporting.

    Args:
        n_particles (int):
            Number of particles in the system. Each particle will be assigned
            its own `(fx, fy, fz)` parameter set.

    Returns:
        openmm.CustomExternalForce:
            A `CustomExternalForce` object configured with per-particle force
            parameters for GADES biasing.

    Raises:
        ValueError: If `n_particles` is negative.

    Examples:
        >>> from GADES import getGADESBiasForce
        >>> system = ...
        >>> force = GAD_force = getGADESBiasForce(system.getNumParticles())
        >>> system.addForce(GAD_force)
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
    def __init__(
        self,
        biased_force: CustomExternalForce,
        bias_atom_indices: Sequence[int],
        hess_func: Callable,
        clamp_magnitude: float,
        kappa: float,
        interval: int,
        stability_interval: Optional[int] = None,
        logfile_prefix: Optional[str] = None,
    ):
        r"""
        Initialize a GADESForceUpdater for applying Gentlest Ascent Dynamics (GADES) bias forces.

        The updater identifies the softest Hessian eigenmode of the system and constructs
        a directional bias force along that mode using:
        $$
        F_{\text{GADES}} = - \kappa \, (\mathbf{F}_{\text{system}} \cdot \vec{n}) \, \vec{n},
        $$

        where $\vec{n}$ is normalized eigenvector corresponding to the softest mode.
        the The bias is applied to a specified set of atoms at regular intervals,
        with optional stability checks and logging.

        Args:
            biased_force (openmm.CustomExternalForce):
                The OpenMM force object that will receive GADES bias forces.
                Must be created using `getGADESBiasForce()`.
            bias_atom_indices (Sequence[int]):
                Indices of atoms that should receive the bias force.
            hess_func (Callable):
                A user-supplied function returning the Hessian matrix for the system.
                Must accept `(system, positions, atom_indices, step_size, platform)`
                as input and return a 2D array-like Hessian. Choose one of 
                `GADES.utils.compute_hessian_force_fd_richardson`, 
                `GADES.utils.compute_hessian_force_fd_block_serial`, or
                `GADES.utils.compute_hessian_force_fd_block_parallel`. We suggest
                the Richardson variant.
            clamp_magnitude (float):
                Maximum allowed magnitude for each component of the bias force,
                used to prevent unphysical updates or exploration of irrelavant
                regions.
            kappa (float):
                Scaling factor (0 < κ < 1) applied to the bias force along the
                softest eigenmode. GADES is designed for exploration applications
                with κ=0.9. Values larger than 1 will lead to the system lingering
                in the transition regions. Values smaller than 0.9 limit the
                maximum gradient GADES is able to overcome. We suggest controling
                this with `clamp_magnitude` instead of `kappa` since `kappa` will
                damp __all__ forces while the clamp is only effective in high-
                gradient regions.
            interval (int):
                Number of simulation steps between bias force updates. Values
                less than 100 are overridden to 110 internally to ensure stability.
                Smaller values make for a more accurate bias direction, at the 
                expense of computational cost. In our experience, a value of 
                ~2000 is a good place to start.
            stability_interval (int, optional):
                Number of steps between stability checks based on kinetic
                temperature. If None, only post-bias stability checks are used.
                This check ensures that the system doesn't stray too far from 
                the set temperature by turning the bias off if the temperature
                rises >50 K of the simulation temeprature. We suggest a value of
                500 steps for most simulations.
            logfile_prefix (str, optional):
                Prefix for log files. If provided, the following files are created:
                  - `<prefix>_evec.log`: trajectory of softest-mode eigenvectors
                  - `<prefix>_eval.log`: trajectory of  sorted eigenvalue spectra
                  - `<prefix>_biased_atoms.xyz`: biased atom trajectories in XYZ format

        Raises:
            ValueError: If `interval` is not a positive integer.
            OSError: If log files cannot be created when `logfile_prefix` is set.

        Examples:
            >>> from GADES import getGADESBiasForce, GADESForceUpdater
            >>> from GADES.utils import compute_hessian_force_fd_richardson as hessian
            >>> system = ...
            >>> simulation = ...
            >>> biasing_atom_ids = ...
            >>> GAD_force = getGADESBiasForce(system.getNumParticles())
            >>> GADESupdater = GADESForceUpdater(
                                biased_force=GAD_force, 
                                bias_atom_indices=biasing_atom_ids,
                                hess_func=hessian, 
                                clamp_magnitude=2500,
                                kappa=0.9, 
                                interval=1000, 
                                stability_interval=500, 
                                logfile_prefix="GADES_log"
                                )
            >>> simulation.reporters.append(GADESupdater)
            >>> simulation.step(10000)
            
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
        Update the scaling factor κ used for the GADES bias force.

        This method is the setter of `kappa`, which determines how strongly
        the bias is applied along the softest Hessian eigenmode. The new value
        will be used in all subsequent bias updates. Note that there are no checks
        on the provided value for `kappa` for maximum flexibility. However, 
        GADES is designed for exploration applications with 0 < κ < 1. Stability
        and behavior has not been tested for κ outside this range. Values larger
        than 1 will lead to the system lingering in the transition regions. 
        Values smaller than 0.9 limit the maximum gradient GADES is able to 
        overcome. We suggest controling this with `clamp_magnitude` instead of 
        `kappa` since `kappa` will damp __all__ forces while the clamp is only 
        effective in high-gradient regions. 

        Args:
            kappa (float):
                New scaling factor κ for the bias force.

        Returns:
            None

        Examples:
            >>> GADESupdater.set_kappa(0.8)
            >>> print(updater.kappa)
            0.8
        """
        self.kappa = kappa
        return None
    
    def set_hess_step_size(self, delta: float) -> None:
        """
        Update the displacement step size used for numerical Hessian calculations.

        The Hessian is computed via finite-difference displacements of the
        atomic coordinates. This method updates the step size `delta` used in
        those displacements, which can affect both accuracy and numerical stability.
        The Richardson method-based Hessian calculators are less prone to 
        numerical errors due to small/large step sizes.

        Args:
            delta (float):
                New displacement step size (in nanometers) for Hessian evaluation.

        Returns:
            None

        Raises:
            ValueError: If `delta` is not positive.

        Examples:
            >>> GADESupdater.set_hess_step_size(1e-4)
            >>> print(updater.hess_step_size)
            0.0001
        """
        if delta <= 0:
            raise ValueError("Hessian step size `delta` cannot be zero or negative.")
        self.hess_step_size = delta
        return None
    
    def _is_stable(self, simulation: openmm.app.Simulation) -> bool:
        """
        Check whether the simulation is thermodynamically stable (internal use only).

        This method estimates the instantaneous temperature from the system's
        kinetic energy and compares it to the target temperature of the integrator.
        If the deviation exceeds 50 K, the system is considered unstable.

        Args:
            simulation (openmm.app.Simulation):
                The OpenMM Simulation object to evaluate.

        Returns:
            bool:
                True if the instantaneous temperature is within 50 K of the
                target temperature, False otherwise.

        Notes:
            - Degrees of freedom (DOF) are reduced by one for each constraint
              and by three if a `CMMotionRemover` is present.
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
    
    def _ensure_atom_symbols(self, simulation: openmm.app.Simulation) -> None:
        """
        Lazily initialize atom symbols from the simulation topology (internal use only).

        This method populates `self.atom_symbols` with element symbols
        corresponding to the atoms listed in `bias_atom_indices`. If an atom
        lacks an associated element, it is assigned the placeholder symbol "X".
        The initialization is performed only once; subsequent calls will be
        no-ops if `self.atom_symbols` is already set.

        Args:
            simulation (openmm.app.Simulation):
                The OpenMM Simulation object from which atom symbols are extracted.

        Sets:
            self.atom_symbols (list of str):
                Atomic symbols corresponding to `bias_atom_indices`.

        Returns:
            None
        """
        if self.atom_symbols is None:
            atom_list = list(simulation.topology.atoms())
            self.atom_symbols = [
                atom_list[i].element.symbol if atom_list[i].element is not None else "X"
                for i in self.bias_atom_indices
            ]
        return None
    
    def _get_gad_force(self, simulation: openmm.app.Simulation) -> np.ndarray:
        """
        Compute the Gentlest Ascent Dynamics (GAD) biasing force (internal use only).

        This method calculates the biased force vector aligned with the softest
        Hessian eigenmode of the system. The bias is scaled by `kappa`, clamped
        to prevent unphysical magnitudes, and reshaped to match the force array
        of the selected atoms. Optional logging writes eigenvectors, eigenvalues,
        and biased atom trajectories to disk.

        Args:
            simulation (openmm.app.Simulation):
                The OpenMM Simulation object containing the current system state.

        Returns:
            np.ndarray:
                The biased force array with shape `(N_bias_atoms, 3)`,
                corresponding to the atoms in `bias_atom_indices`.

        Notes:
            - The Hessian is computed using `self.hess_func`, which must accept
              `(system, positions, atom_indices, step_size, platform)` as arguments.
            - The eigenvector associated with the smallest eigenvalue (softest mode)
              is normalized and used to construct the bias direction.
            - Forces are clamped so that the force on each particle does not exceed
              `self.clamp_magnitude` in absolute value.
            - If logging is enabled (`logfile_prefix` set at initialization),
              eigenvectors, eigenvalues, and atom coordinates are written to
              `<prefix>_evec.log`, `<prefix>_eval.log`, and `<prefix>_biased_atoms.xyz`.
        """
        state = simulation.context.getState(getPositions=True, getForces=True)
        platform = simulation.context.getPlatform().getName()
        forces_u = state.getForces(asNumpy=True)[self.bias_atom_indices, :]
        positions = state.getPositions(asNumpy=True)
        hess = self.hess_func(simulation.system, positions, self.bias_atom_indices, self.hess_step_size, platform)
        w, v = np.linalg.eigh(hess)
        w_sorted = w.argsort()
        n = v[:, w_sorted[0]]
        n /= np.linalg.norm(n)
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
        
    def describeNextReport(self, simulation: openmm.app.Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """
        Define when the reporter should run next and what data it requires.

        This method is required by the OpenMM `Reporter` interface and must be
        implemented in all reporter subclasses. It determines how many steps
        until the next reporting event and specifies which data types (positions,
        velocities, forces, energies, volumes) are needed at that time.

        Args:
            simulation (openmm.app.Simulation):
                The OpenMM Simulation object providing the current step and state.

        Returns:
            tuple:
                A 6-element tuple with the following contents:
                  - steps (int): Number of steps until the next report.
                  - needsPositions (bool): Always False.
                  - needsVelocities (bool): Always False.
                  - needsForces (bool): Always False.
                  - needsEnergy (bool): Always False.
                  - needsVolume (bool): Always False.

        Notes:
            - Even though this method is not part of the intended public API for
              `GADESForceUpdater`, it must remain public (no leading underscore)
              because OpenMM requires `describeNextReport` to be defined.
            - Internally, this method:
                * Ensures atom symbols are initialized for logging.
                * Schedules the next bias update, stability check, or post-bias check.
                * Sets internal flags (`is_biasing`, `check_stability`) for use in
                  subsequent reporting steps.
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

    def report(self, simulation: openmm.app.Simulation, 
               state: openmm.State) -> None:
        """
        Apply (or remove) GADES bias forces at the current simulation step.

        This method fulfills the OpenMM `Reporter` interface requirement. It uses
        internal scheduling flags set by `describeNextReport` to decide whether to:
          1) perform a stability check and remove bias if unstable,
          2) update/apply the GADES bias forces, or
          3) do nothing this step.
        Parameter updates are pushed to the OpenMM context when modifications occur.

        Args:
            simulation (openmm.app.Simulation):
                The OpenMM Simulation object (provides context, step counter, etc.).
            state (openmm.State):
                Current simulation state. Unused here, but required by the Reporter API.

        Returns:
            None

        Side Effects:
            - Calls `_get_gad_force` to compute biased forces when `is_biasing` is True.
            - Updates `self.biased_force` per-atom parameters and pushes them with
              `updateParametersInContext(simulation.context)`.
            - Clears or sets scheduling flags:
                * `self.check_stability` → False after a stability-handling step.
                * `self.is_biasing` → False after bias has been applied.
                * `self.next_postbias_check_step` → set to `step + 100` after applying bias,
                  or cleared when the scheduled post-bias check is reached.
            - Emits informational messages to stdout about actions taken.

        Internal Helpers:
            - `remove_bias()` (internal use only):
                Reset the per-atom bias parameters to `(0.0, 0.0, 0.0)` for all
                `bias_atom_indices`, effectively disabling the bias.
            - `apply_bias()` (internal use only):
                Compute the current GADES bias via `_get_gad_force(simulation)` and set
                per-atom parameters accordingly for all `bias_atom_indices`.

        Notes:
            - This method is typically triggered by OpenMM according to the schedule
              returned by `describeNextReport`. It defensively calls
              `_ensure_atom_symbols(simulation)` in case the reporter was invoked
              before `describeNextReport`.
            - If `self.check_stability` is True, the method first evaluates stability via
              `_is_stable(simulation)`:
                * If unstable (ΔT > 50 K from target), the bias is removed for safety.
                * If stable and `self.is_biasing` is True, the bias is (re)applied and a
                  post-bias check is scheduled in 100 steps.
            - If neither `self.check_stability` nor `self.is_biasing` is set, the method
              performs no action for the current step.
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

        if self.check_stability:
            is_stable = self._is_stable(simulation)
            if not is_stable:
                print(f"\033[1;31m[GADES | step {step}] System is unstable: Removing bias until next cycle...\033[0m", flush=True)
                remove_bias()
            elif self.is_biasing:
                print(f"\033[1;32m[GADES | step {step}] Updating bias forces...\033[0m", flush=True)
                apply_bias()
                self.next_postbias_check_step = step + 100

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
            self.next_postbias_check_step = step + 100
            return None

        # If neither flag is True, do nothing
        return None
    
    def _close_logs(self) -> None:
        """
        Close all open log files associated with the updater (internal use only).

        This method ensures that any log files opened during initialization are
        properly closed. It is registered with `atexit` for automatic cleanup at
        interpreter shutdown and is also called by `__del__` as a safeguard.

        Closes:
            - `self._evec_log`: eigenvector log file (`<prefix>_evec.log`)
            - `self._eval_log`: eigenvalue log file (`<prefix>_eval.log`)
            - `self._xyz_log`: biased atom trajectory log file (`<prefix>_biased_atoms.xyz`)

        Notes:
            - Each file handle is checked for existence and open state before closing.
            - Any exceptions during file closure are suppressed to avoid interfering
              with shutdown.

        Returns:
            None
        """
        for attr in ("_evec_log", "_eval_log", "_xyz_log"):
            f = getattr(self, attr, None)
            if f is not None and not f.closed:
                try:
                    f.close()
                except Exception:
                    pass
        return None

    def __del__(self) -> None:
        """
        Clean up the log files when the object is garbage-collected

        Returns:
            None
        """
        self._close_logs()
        return None