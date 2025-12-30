import atexit
import logging
import warnings
import numpy as np
from typing import Sequence, Callable, Optional

from .utils import clamp_force_magnitudes as fclamp

# Get the GADES logger (configured in __init__.py)
logger = logging.getLogger("GADES")

class GADESBias:
    def __init__(
        self,
        backend,
        biased_force,
        bias_atom_indices: Sequence[int],
        hess_func: Callable,
        clamp_magnitude: float,
        kappa: float,
        interval: int,
        stability_interval: Optional[int] = None,
        logfile_prefix: Optional[str] = None,
    ):
        r"""
        Initialize a GADESBias for applying Gentlest Ascent Dynamics (GADES) bias forces.

        The updater identifies the softest Hessian eigenmode of the system and constructs
        a directional bias force along that mode using:
        $$
        F_{\text{GADES}} = - \kappa \, (\mathbf{F}_{\text{system}} \cdot \vec{n}) \, \vec{n},
        $$

        where $\vec{n}$ is normalized eigenvector corresponding to the softest mode.
        the The bias is applied to a specified set of atoms at regular intervals,
        with optional stability checks and logging.

        Args:
            backend:
                e.g. OpenMMBackend if using the OpenMM backend
                or ASEBackend if using the ASE backend.
                The simulation backend providing access to system state.
            biased_force:
                openmm.CustomExternalForce if using the OpenMM backend
                The OpenMM force object that will receive GADES bias forces.
                Must be created using `createGADESBiasForce()`.
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
            TypeError: If `hess_func` is not callable.
            TypeError: If `bias_atom_indices` is not a sequence.
            ValueError: If `bias_atom_indices` is empty or contains non-integer/negative values.
            ValueError: If `clamp_magnitude` is not a positive number.
            ValueError: If `interval` is not a positive integer.
            ValueError: If `stability_interval` is provided but not a positive integer.
            OSError: If log files cannot be created when `logfile_prefix` is set.

        Warns:
            UserWarning: If `kappa` is outside the recommended range (0, 1].

        Examples:
            >>> from GADES import createGADESBiasForce, GADESForceUpdater
            >>> from GADES.utils import compute_hessian_force_fd_richardson as hessian
            >>> system = ...
            >>> simulation = ...
            >>> biasing_atom_ids = ...
            >>> GAD_force = createGADESBiasForce(system.getNumParticles())
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
        # --- Input validation ---
        # Validate hess_func
        if not callable(hess_func):
            raise TypeError(f"hess_func must be callable, got {type(hess_func).__name__}")

        # Validate bias_atom_indices
        if not hasattr(bias_atom_indices, '__iter__'):
            raise TypeError(f"bias_atom_indices must be a sequence, got {type(bias_atom_indices).__name__}")
        bias_atom_indices = list(bias_atom_indices)
        if len(bias_atom_indices) == 0:
            raise ValueError("bias_atom_indices cannot be empty")
        if not all(isinstance(i, (int, np.integer)) and i >= 0 for i in bias_atom_indices):
            raise ValueError("All bias_atom_indices must be non-negative integers")

        # Validate clamp_magnitude
        if not isinstance(clamp_magnitude, (int, float, np.number)) or clamp_magnitude <= 0:
            raise ValueError(f"clamp_magnitude must be a positive number, got {clamp_magnitude}")

        # Validate interval
        if not isinstance(interval, (int, np.integer)) or interval <= 0:
            raise ValueError(f"interval must be a positive integer, got {interval}")

        # Validate stability_interval (if provided)
        if stability_interval is not None:
            if not isinstance(stability_interval, (int, np.integer)) or stability_interval <= 0:
                raise ValueError(f"stability_interval must be a positive integer, got {stability_interval}")

        # Validate kappa (warning only, for flexibility)
        if not (0 < kappa <= 1):
            warnings.warn(
                f"kappa={kappa} is outside the recommended range (0, 1]. "
                "Values > 1 may cause the system to linger in transition regions. "
                "Values <= 0 will invert or nullify the bias force.",
                UserWarning
            )

        # --- Attribute assignment ---
        self.backend = backend
        self.biased_force = biased_force
        self.bias_atom_indices = bias_atom_indices
        self.hess_func = hess_func
        self.clamp_magnitude = clamp_magnitude
        if interval < 100:
            logger.warning(
                "Bias update interval must be larger than 100 steps to ensure system stability. "
                "Changing the frequency to 110 steps internally."
            )
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
            TypeError: If `delta` is not a number.
            ValueError: If `delta` is not positive.

        Examples:
            >>> GADESupdater.set_hess_step_size(1e-4)
            >>> print(updater.hess_step_size)
            0.0001
        """
        if not isinstance(delta, (int, float, np.number)):
            raise TypeError(f"delta must be a number, got {type(delta).__name__}")
        if delta <= 0:
            raise ValueError("Hessian step size `delta` must be positive.")
        self.hess_step_size = delta
        return None
    
    def _is_stable(self) -> bool:
        """
        invokes the backend is_stable() method to determine if the system's stable.
        """
        return self.backend.is_stable()
       
    def _ensure_atom_symbols(self) -> None:
        """
        Lazily initialize atom symbols from the simulation topology (internal use only).

        This method populates `self.atom_symbols` with element symbols
        corresponding to the atoms listed in `bias_atom_indices`. If an atom
        lacks an associated element, it is assigned the placeholder symbol "X".
        The initialization is performed only once; subsequent calls will be
        no-ops if `self.atom_symbols` is already set.

        Sets:
            self.atom_symbols (list of str):
                Atomic symbols corresponding to `bias_atom_indices`.

        Returns:
            None
        """
        if self.atom_symbols is None:
            self.atom_symbols = self.backend.get_atom_symbols(self.bias_atom_indices)
    
    def get_gad_force(self) -> np.ndarray:
        """
        Compute the Gentlest Ascent Dynamics (GAD) biasing force.

        This method calculates the biased force vector aligned with the softest
        Hessian eigenmode of the system. The bias is scaled by `kappa`, clamped
        to prevent unphysical magnitudes, and reshaped to match the force array
        of the selected atoms. Optional logging writes eigenvectors, eigenvalues,
        and biased atom trajectories to disk.

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
        positions, forces = self.backend.get_current_state()
        forces_u = forces[self.bias_atom_indices, :]

        platform = "CPU"
        hess = self.hess_func(self.backend, self.bias_atom_indices, self.hess_step_size, platform,)

        w, v = np.linalg.eigh(hess)
        w_sorted = w.argsort()
        n = v[:, w_sorted[0]]
        n /= np.linalg.norm(n)
        forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
        # clamping biased forces so their abs value is never larger than `clamp_magnitude`
        forces_b = fclamp(forces_b, self.clamp_magnitude)

        # Logging
        self._logging(n, w, w_sorted, positions)

        return forces_b.reshape(forces_u.shape)

    def _logging(self, n, w, w_sorted, positions) -> None:
        self._ensure_atom_symbols()
        step = self.backend.get_currentStep()

        if self._evec_log is not None:
            self._evec_log.write(f"{step} " + " ".join(map(str, n)) + "\n")
            self._evec_log.flush()
        if self._eval_log is not None:
            self._eval_log.write(f"{step} " + " ".join(map(str, w[w_sorted])) + "\n")
            self._eval_log.flush()
        if self._xyz_log is not None:
            pos_nm = positions[self.bias_atom_indices, :]
            self._xyz_log.write(f"{len(self.bias_atom_indices)}\n")
            self._xyz_log.write(f"Step {step}\n")
            for symbol, coord in zip(self.atom_symbols, pos_nm):
                x, y, z = coord
                self._xyz_log.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
            self._xyz_log.flush()

    def remove_bias(self) -> None:
        """
        Invoke the backend remove_bias() to reset the per-atom bias parameters to `(0.0, 0.0, 0.0)`
        for all `bias_atom_indices`, effectively disabling the bias.

        """
        self.backend.remove_bias(self.biased_force, self.bias_atom_indices)

    def apply_bias(self) -> None:
        """
        Compute the current GADES bias via `get_gad_force()` and invoke the backend `apply_bias()`
        for all `bias_atom_indices`.
        """
        gad_biased_forces = self.get_gad_force()
        self.backend.apply_bias(self.biased_force, gad_biased_forces, self.bias_atom_indices)

    def applying_bias(self) -> bool:
        """
        Returns whether the bias is being applied in the current step.

        Returns:
            bool: True if the bias is being applied, False otherwise.
        """
        step = self.backend.get_currentStep()
        if step < 0:
            self.is_biasing = False
        elif step % self.interval == 0:
            self.is_biasing = True
        else:
            self.is_biasing = False
        return self.is_biasing
    
    def register_next_step(self) -> int:
        """
        Define when the bias forces should run next
        This function is used by the OpenMM Reporter interface in describeNextReport().

        Returns:
            - steps (int): Number of steps until the next report.

        """
        step = self.backend.get_currentStep()

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

        return steps

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
                    raise warnings.warn(f"Failed to close log file {attr} properly.")
        return None

    def __del__(self) -> None:
        """
        Clean up the log files when the object is garbage-collected

        Returns:
            None
        """
        self._close_logs()
        return None

'''
    OpenMM specific features
'''
from openmm import CustomExternalForce
def createGADESBiasForce(n_particles: int) -> CustomExternalForce:
    """
    Create a custom OpenMM force object used for GADES biasing.

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
        ValueError: If `n_particles` is not a non-negative integer.

    Examples:
        >>> from GADES import createGADESBiasForce
        >>> system = ...
        >>> GAD_force = createGADESBiasForce(system.getNumParticles())
        >>> system.addForce(GAD_force)
    """
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 0:
        raise ValueError(f"n_particles must be a non-negative integer, got {n_particles}")

    force = CustomExternalForce("fx*x+fy*y+fz*z")
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    for i in range(n_particles):
        force.addParticle(i, [0.0, 0.0, 0.0])
    force.setForceGroup(1)
    return force

class GADESForceUpdater(GADESBias):
    def __init__(
        self,
        backend,
        biased_force,
        bias_atom_indices: Sequence[int],
        hess_func: Callable,
        clamp_magnitude: float,
        kappa: float,
        interval: int,
        stability_interval: Optional[int] = None,
        logfile_prefix: Optional[str] = None,
    ):
        r"""
        Initialize a GADESForceUpdater for applying Gentlest Ascent Dynamics (GADES) bias forces
        as an OpenMM Reporter.

        The updater identifies the softest Hessian eigenmode of the system and constructs
        a directional bias force along that mode using:
        $$
        F_{\text{GADES}} = - \kappa \, (\mathbf{F}_{\text{system}} \cdot \vec{n}) \, \vec{n},
        $$

        where $\vec{n}$ is normalized eigenvector corresponding to the softest mode.
        The bias is applied to the specified set of atoms at regular intervals,
        with optional stability checks and logging.

        Args:
            biased_force:
                openmm.CustomExternalForce if using the OpenMM backend
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
                used to prevent unphysical updates or exploration of irrelevant
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
                rises >50 K of the simulation temperature. We suggest a value of
                500 steps for most simulations.
            logfile_prefix (str, optional):
                Prefix for log files. If provided, the following files are created:
                  - `<prefix>_evec.log`: trajectory of softest-mode eigenvectors
                  - `<prefix>_eval.log`: trajectory of  sorted eigenvalue spectra
                  - `<prefix>_biased_atoms.xyz`: biased atom trajectories in XYZ format

        Raises:
            TypeError: If `hess_func` is not callable.
            TypeError: If `bias_atom_indices` is not a sequence.
            ValueError: If `bias_atom_indices` is empty or contains non-integer/negative values.
            ValueError: If `clamp_magnitude` is not a positive number.
            ValueError: If `interval` is not a positive integer.
            ValueError: If `stability_interval` is provided but not a positive integer.
            OSError: If log files cannot be created when `logfile_prefix` is set.

        Warns:
            UserWarning: If `kappa` is outside the recommended range (0, 1].

        Examples:
            >>> from GADES import createGADESBiasForce, GADESForceUpdater
            >>> from GADES.utils import compute_hessian_force_fd_richardson as hessian
            >>> system = ...
            >>> simulation = ...
            >>> biasing_atom_ids = ...
            >>> GAD_force = createGADESBiasForce(system.getNumParticles())
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
        super().__init__(
            backend,
            biased_force,
            bias_atom_indices,
            hess_func,
            clamp_magnitude,
            kappa,
            interval,
            stability_interval,
            logfile_prefix,
        )
            
    def describeNextReport(self, simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        """
        Define when the reporter should run next and what data it requires.

        This method is required by the OpenMM `Reporter` interface and must be
        implemented in all reporter subclasses. It determines how many steps
        until the next reporting event and specifies which data types (positions,
        velocities, forces, energies, volumes) are needed at that time.

        Args:
            simulation: openmm.app.Simulation
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
        steps = self.register_next_step()
        return (steps, False, False, False, False, False)

    def report(self, simulation, state) -> None:
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
            - Calls `get_gad_force` to compute biased forces when `is_biasing` is True.
            - Updates `self.biased_force` per-atom parameters and pushes them with
              `updateParametersInContext(simulation.context)`.
            - Clears or sets scheduling flags:
                * `self.check_stability` → False after a stability-handling step.
                * `self.is_biasing` → False after bias has been applied.
                * `self.next_postbias_check_step` → set to `step + 100` after applying bias,
                  or cleared when the scheduled post-bias check is reached.
            - Emits informational messages to stdout about actions taken.

        - `remove_bias()` and `apply_bias()` used to be internal helpers, 
           now moved to the parent class GADESBias which then invoke the corresponding methods of the backend

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
        step = self.backend.get_currentStep()

        # Defensive fallback in case describeNextReport hasn't been called yet
        self._ensure_atom_symbols()

        if self.check_stability:
            is_stable = self._is_stable()
            if not is_stable:
                logger.warning(f"step {step}] System is unstable: Removing bias until next cycle...")
                self.remove_bias()
            elif self.is_biasing:
                logger.info(f"step {step}] Updating bias forces...")
                self.apply_bias()
                self.next_postbias_check_step = step + 100

            self.check_stability = False
            self.is_biasing = False
            if step == self.next_postbias_check_step:
                self.next_postbias_check_step = None
            return None

        if self.is_biasing:
            logger.info(f"step {step}] Updating bias forces...")
            self.apply_bias()
            self.is_biasing = False
            self.next_postbias_check_step = step + 100
            return None

        # If neither flag is True, do nothing
        return None
    
