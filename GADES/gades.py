import atexit
import logging
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .config import defaults
from .utils import clamp_force_magnitudes as fclamp

if TYPE_CHECKING:
    from .backend import Backend

# Get the GADES logger (configured in __init__.py)
logger = logging.getLogger("GADES")


class GADESBias:
    def __init__(
        self,
        backend: "Backend",
        biased_force: Any,
        bias_atom_indices: Sequence[int],
        hess_func: Callable,
        clamp_magnitude: float,
        kappa: float,
        interval: int,
        stability_interval: Optional[int] = None,
        logfile_prefix: Optional[str] = None,
        eigensolver: str = "numpy",
        lanczos_iterations: Optional[int] = None,
        use_bofill_update: bool = False,
        full_hessian_interval: Optional[int] = None,
        hvp_epsilon: Optional[float] = None,
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
                Must accept `(backend, atom_indices, step_size, platform)` as input
                and return a 2D array-like Hessian. Choose one of
                `GADES.utils.compute_hessian_force_fd_richardson` (recommended) or
                `GADES.utils.compute_hessian_force_fd_block_serial`.
            clamp_magnitude (float):
                Maximum allowed magnitude for each atom's bias force vector.
                Forces exceeding this magnitude are rescaled (direction preserved).
                Used to prevent unphysical updates or exploration of irrelevant
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
            eigensolver (str, optional):
                Method for computing the softest eigenmode. Options:
                  - ``'numpy'`` (default): Use ``np.linalg.eigh()`` for full eigendecomposition.
                    Accurate and recommended for small systems (< 500 atoms).
                  - ``'lanczos'``: Use Lanczos iteration to find only the smallest eigenvalue.
                    Faster for large systems but approximate. Requires explicit Hessian.
                  - ``'lanczos_hvp'``: Matrix-free Lanczos using Hessian-vector products via
                    finite differences. Scales to very large systems (1000+ atoms) by avoiding
                    explicit Hessian construction. Memory usage O(N) instead of O(N²).
            lanczos_iterations (int, optional):
                Number of Lanczos iterations when ``eigensolver='lanczos'`` or ``'lanczos_hvp'``.
                More iterations improve accuracy. Defaults to ``defaults["lanczos_iterations"]``.
            hvp_epsilon (float, optional):
                Finite difference step size for Hessian-vector products when using
                ``eigensolver='lanczos_hvp'``. Defaults to ``defaults["hvp_epsilon"]``.
            use_bofill_update (bool, optional):
                If True, use Bofill quasi-Newton updates to approximate the Hessian
                between full Hessian calculations. This reduces computational cost
                by avoiding expensive Hessian evaluations at every bias update.
                Defaults to False.
            full_hessian_interval (int, optional):
                When ``use_bofill_update=True``, specifies how often (in steps) to
                compute the full Hessian instead of using Bofill approximation.
                Defaults to ``interval * defaults["bofill_full_hessian_multiplier"]``.

        Raises:
            TypeError: If `hess_func` is not callable.
            TypeError: If `bias_atom_indices` is not a sequence.
            ValueError: If `bias_atom_indices` is empty or contains non-integer/negative values.
            ValueError: If `clamp_magnitude` is not a positive number.
            ValueError: If `interval` is not a positive integer.
            ValueError: If `stability_interval` is provided but not a positive integer.
            ValueError: If `eigensolver` is not one of 'numpy', 'lanczos', or 'lanczos_hvp'.
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

        # Bounds check against system size (skip if backend is None for testing)
        if backend is not None:
            n_atoms = len(backend.get_positions())
            max_index = max(bias_atom_indices)
            if max_index >= n_atoms:
                raise ValueError(
                    f"bias_atom_indices contains index {max_index}, but system only has "
                    f"{n_atoms} atoms (valid range: 0 to {n_atoms - 1})"
                )

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

        # Validate eigensolver
        valid_eigensolvers = ('numpy', 'lanczos', 'lanczos_hvp')
        if eigensolver not in valid_eigensolvers:
            raise ValueError(
                f"eigensolver must be one of {valid_eigensolvers}, got '{eigensolver}'"
            )

        # Validate hvp_epsilon (if provided)
        if hvp_epsilon is not None:
            if not isinstance(hvp_epsilon, (int, float, np.number)) or hvp_epsilon <= 0:
                raise ValueError(f"hvp_epsilon must be a positive number, got {hvp_epsilon}")

        # Validate lanczos_iterations (if provided)
        if lanczos_iterations is not None:
            if not isinstance(lanczos_iterations, (int, np.integer)) or lanczos_iterations <= 0:
                raise ValueError(f"lanczos_iterations must be a positive integer, got {lanczos_iterations}")

        # Validate full_hessian_interval (if provided)
        if full_hessian_interval is not None:
            if not isinstance(full_hessian_interval, (int, np.integer)) or full_hessian_interval <= 0:
                raise ValueError(f"full_hessian_interval must be a positive integer, got {full_hessian_interval}")

        # Warn if Bofill is enabled with lanczos_hvp (Bofill has no effect)
        if use_bofill_update and eigensolver == 'lanczos_hvp':
            warnings.warn(
                "use_bofill_update=True has no effect with eigensolver='lanczos_hvp'. "
                "The lanczos_hvp method computes Hessian-vector products via finite differences "
                "and does not use an explicit Hessian matrix that Bofill could update.",
                UserWarning
            )

        # --- Attribute assignment ---
        self.backend = backend
        self.biased_force = biased_force
        self.bias_atom_indices = bias_atom_indices
        self.hess_func = hess_func
        self.clamp_magnitude = clamp_magnitude
        min_interval = defaults["min_bias_update_interval"]
        if interval < min_interval - 10:
            logger.warning(
                f"Bias update interval must be larger than {min_interval - 10} steps to ensure system stability. "
                f"Changing the frequency to {min_interval} steps internally."
            )
            self.interval = min_interval
        else:
            self.interval = interval
        self.kappa = kappa
        self.hess_step_size = 1e-5
        self.check_stability = False
        self.is_biasing = False
        self.s_interval = stability_interval

        # Eigensolver settings
        self.eigensolver = eigensolver
        self.lanczos_iterations = lanczos_iterations or defaults["lanczos_iterations"]
        self.hvp_epsilon = hvp_epsilon or defaults["hvp_epsilon"]

        # Bofill update settings
        self.use_bofill_update = use_bofill_update
        if full_hessian_interval is not None:
            self.full_hessian_interval = full_hessian_interval
        else:
            self.full_hessian_interval = self.interval * defaults["bofill_full_hessian_multiplier"]

        # Bofill state (for storing previous Hessian/positions/forces)
        self._last_hess = None
        self._last_positions = None
        self._last_forces = None
        self._last_hess_step = -1  # Step at which last full Hessian was computed

        # post bias update check
        self.next_postbias_check_step = None
        self._last_biased_step = None

        # logging
        self.atom_symbols = None
        self.logfile_prefix = logfile_prefix
        self._evec_log = None
        self._eval_log = None
        self._xyz_log = None
        self._epot_log = None
        self._forces_log = None
        self._hess_log = None
        self._pos_log = None

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

            self._epot_log = open(f"{logfile_prefix}_epot.log", "w")
            self._epot_log.write("# Unbiased potential energy at each bias step\n")
            self._epot_log.write("# Columns: step, potential energy (kJ/mol for OpenMM; eV for ASE)\n")
            self._epot_log.flush()

            self._forces_log = open(f"{logfile_prefix}_forces.log", "w")
            self._forces_log.write("# Unbiased forces on biased atoms at each bias step\n")
            self._forces_log.write("# Columns: step, force components (flattened, biased atoms only)\n")
            self._forces_log.flush()

            self._hess_log = open(f"{logfile_prefix}_hess.log", "w")
            self._hess_log.write("# Hessian of the biased-atom subspace at each bias step\n")
            self._hess_log.write("# Columns: step, Hessian elements (row-major); unavailable for lanczos_hvp\n")
            self._hess_log.flush()

            self._pos_log = open(f"{logfile_prefix}_pos.log", "w")
            self._pos_log.write("# Positions of biased atoms at each bias step\n")
            self._pos_log.write("# Columns: step, position components (flattened, biased atoms only)\n")
            self._pos_log.flush()

            # Warn if eigenvalue logging is unavailable with lanczos_hvp
            if eigensolver == 'lanczos_hvp':
                logger.warning(
                    "Eigenvalue logging unavailable with eigensolver='lanczos_hvp' "
                    "(no full Hessian computed). Eigenvector and XYZ logging will still work."
                )
            
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

    def _compute_softest_mode(self, hess: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the smallest eigenvalue and corresponding eigenvector of the Hessian.

        This method dispatches to either NumPy's ``eigh`` or the Lanczos algorithm
        based on the ``eigensolver`` setting.

        Args:
            hess (np.ndarray): The Hessian matrix, shape (3N, 3N).

        Returns:
            tuple: (eigenvalue, eigenvector) where:
                - eigenvalue (float): The smallest eigenvalue (softest mode).
                - eigenvector (np.ndarray): The corresponding normalized eigenvector.
        """
        if self.eigensolver == 'lanczos':
            from .lanczos import lanczos_smallest
            eigval, eigvec = lanczos_smallest(hess, n_iter=self.lanczos_iterations)
        else:
            # Default: numpy
            w, v = np.linalg.eigh(hess)
            idx = w.argsort()[0]
            eigval, eigvec = w[idx], v[:, idx]

        # Normalize eigenvector
        eigvec = eigvec / np.linalg.norm(eigvec)
        return eigval, eigvec

    def _compute_softest_mode_hvp(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the smallest eigenvalue and eigenvector using matrix-free Lanczos with HVP.

        This method uses Hessian-vector products via finite differences, avoiding
        explicit construction of the full Hessian matrix. This scales as O(N) in
        memory instead of O(N²), making it suitable for large systems.

        Args:
            positions (np.ndarray): Full atomic positions, shape (N_atoms, 3).

        Returns:
            tuple: (eigenvalue, eigenvector) where:
                - eigenvalue (float): The smallest eigenvalue (softest mode).
                - eigenvector (np.ndarray): The corresponding normalized eigenvector,
                  shape (3 * N_bias,).

        Notes:
            The HVP is computed by displacing only the biased atoms and computing
            force differences. This requires 2 force evaluations per Lanczos iteration.
        """
        from .hvp import finite_difference_hvp
        from .lanczos import lanczos_hvp_smallest

        n_bias = len(self.bias_atom_indices)
        n_dof = 3 * n_bias

        # Extract positions of biased atoms
        bias_positions = positions[self.bias_atom_indices, :].copy()

        def force_func_biased(pos_biased_flat: np.ndarray) -> np.ndarray:
            """
            Force function for biased atoms only.

            Takes flattened positions of biased atoms, updates full positions,
            computes forces via backend, and returns forces on biased atoms.
            """
            # Reshape to (N_bias, 3)
            pos_biased = pos_biased_flat.reshape(-1, 3)

            # Create full positions with displaced biased atoms
            full_positions = positions.copy()
            full_positions[self.bias_atom_indices, :] = pos_biased

            # Compute forces via backend (returns negative gradient, flattened)
            forces_flat = self.backend.get_forces(full_positions)

            # Reshape to (N_atoms, 3) and extract biased atoms
            forces = forces_flat.reshape(-1, 3)
            forces_biased = forces[self.bias_atom_indices, :]

            return forces_biased.flatten()

        def hvp_func(v: np.ndarray) -> np.ndarray:
            """Hessian-vector product for biased atoms."""
            return finite_difference_hvp(
                force_func_biased,
                bias_positions,
                v.reshape(n_dof),
                epsilon=self.hvp_epsilon,
            )

        # Run matrix-free Lanczos
        eigval, eigvec = lanczos_hvp_smallest(
            hvp_func,
            n_dof=n_dof,
            n_iter=self.lanczos_iterations,
        )

        # Normalize eigenvector
        eigvec = eigvec / np.linalg.norm(eigvec)
        return eigval, eigvec

    def _get_hessian(self, positions: np.ndarray, forces: np.ndarray, step: int) -> np.ndarray:
        """
        Get the Hessian matrix, either by full computation or Bofill approximation.

        When ``use_bofill_update`` is enabled, this method uses the Bofill quasi-Newton
        update to approximate the Hessian between full Hessian calculations. Full Hessian
        is computed at intervals specified by ``full_hessian_interval``.

        Args:
            positions (np.ndarray): Current atomic positions.
            forces (np.ndarray): Current atomic forces.
            step (int): Current simulation step.

        Returns:
            np.ndarray: The Hessian matrix, shape (3N_bias, 3N_bias).
        """
        platform = "CPU"
        bias_positions = positions[self.bias_atom_indices, :]
        bias_forces = forces[self.bias_atom_indices, :]

        # Determine if we need to compute full Hessian
        need_full_hessian = (
            not self.use_bofill_update or
            self._last_hess is None or
            (step - self._last_hess_step) >= self.full_hessian_interval
        )

        if need_full_hessian:
            # Compute full Hessian
            hess = self.hess_func(
                self.backend, self.bias_atom_indices, self.hess_step_size, platform
            )
            # Store state for future Bofill updates
            if self.use_bofill_update:
                self._last_hess = hess.copy()
                self._last_positions = bias_positions.copy()
                self._last_forces = bias_forces.copy()
                self._last_hess_step = step
        else:
            # Use Bofill approximation
            from .bofill import get_bofill_H
            # bias_forces comes from get_forces() which returns F = -∇V (forces)
            # Bofill expects gradients (∇V), so we negate: -F = ∇V
            hess = get_bofill_H(
                pos_new=bias_positions,
                pos_old=self._last_positions,
                grad_new=-bias_forces,
                grad_old=-self._last_forces,
                H=self._last_hess
            )
            # Update stored state for next Bofill iteration
            self._last_hess = hess.copy()
            self._last_positions = bias_positions.copy()
            self._last_forces = bias_forces.copy()

        return hess

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
              `(backend, atom_indices, step_size, platform)` as arguments.
            - The eigenvector associated with the smallest eigenvalue (softest mode)
              is normalized and used to construct the bias direction.
            - Forces are clamped so that the force on each particle does not exceed
              `self.clamp_magnitude` in absolute value.
            - If logging is enabled (`logfile_prefix` set at initialization),
              eigenvectors, eigenvalues, and atom coordinates are written to
              `<prefix>_evec.log`, `<prefix>_eval.log`, and `<prefix>_biased_atoms.xyz`.
        """
        positions = self.backend.get_positions()
        # Use unbiased forces for bias calculation (true PES curvature)
        forces_unbiased = self.backend.get_forces(positions).reshape(-1, 3)
        forces_u = forces_unbiased[self.bias_atom_indices, :]
        step = self.backend.get_currentStep()

        if self.eigensolver == 'lanczos_hvp':
            # Matrix-free path: compute softest mode using HVP
            eigval, n = self._compute_softest_mode_hvp(positions)
            hess = None  # No Hessian computed
        else:
            # Matrix-based path: compute or approximate Hessian
            hess = self._get_hessian(positions, forces_unbiased, step)
            eigval, n = self._compute_softest_mode(hess)

        # Compute GAD bias force
        forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
        # clamping biased forces so their abs value is never larger than `clamp_magnitude`
        forces_b = fclamp(forces_b, self.clamp_magnitude)

        # Logging (compute full spectrum only if logging is enabled and Hessian available)
        needs_logging = any(
            f is not None for f in (
                self._evec_log, self._eval_log, self._xyz_log,
                self._epot_log, self._forces_log, self._hess_log, self._pos_log,
            )
        )
        if needs_logging:
            energy = self.backend.get_energy() if self._epot_log is not None else None
            if self._eval_log is not None and hess is not None:
                w, v = np.linalg.eigh(hess)
                w_sorted = w.argsort()
                self._logging(n, w, w_sorted, positions, forces_u, hess, energy)
            else:
                self._logging(n, None, None, positions, forces_u, hess, energy)

        return forces_b.reshape(forces_u.shape)

    def _logging(self, n, w, w_sorted, positions, forces_u=None, hess=None, energy=None) -> None:
        self._ensure_atom_symbols()
        step = self.backend.get_currentStep()

        if self._evec_log is not None:
            self._evec_log.write(f"{step} " + " ".join(map(str, n)) + "\n")
            self._evec_log.flush()
        if self._eval_log is not None and w is not None:
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
        if self._epot_log is not None and energy is not None:
            self._epot_log.write(f"{step} {energy}\n")
            self._epot_log.flush()
        if self._forces_log is not None and forces_u is not None:
            self._forces_log.write(f"{step} " + " ".join(map(str, forces_u.flatten())) + "\n")
            self._forces_log.flush()
        if self._hess_log is not None and hess is not None:
            self._hess_log.write(f"{step} " + " ".join(map(str, hess.flatten())) + "\n")
            self._hess_log.flush()
        if self._pos_log is not None:
            pos_biased = positions[self.bias_atom_indices, :]
            self._pos_log.write(f"{step} " + " ".join(map(str, pos_biased.flatten())) + "\n")
            self._pos_log.flush()

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
        self._last_biased_step = self.backend.get_currentStep()

    def applying_bias(self) -> bool:
        """
        Returns whether the bias is being applied in the current step.

        Returns:
            bool: True if the bias is being applied, False otherwise.
        """
        step = self.backend.get_currentStep()
        if step < 0:
            self.is_biasing = False
        elif step % self.interval == 0 and step != self._last_biased_step:
            self.is_biasing = True
        else:
            self.is_biasing = False
        return self.is_biasing

    def should_check_stability(self) -> bool:
        """
        Check if stability should be verified this step.

        Used by ASE backend to determine when to perform temperature-based
        stability checks. Returns True if:
          - ``stability_interval`` is set and the current step is a multiple
            of that interval, OR
          - A post-bias stability check has been scheduled and the current
            step has reached or passed that scheduled step.

        Post-bias checks are scheduled after each bias update to catch
        instabilities caused by bias application. This ensures ASE users
        get stability monitoring even when ``stability_interval=None``.

        Returns:
            bool: True if stability should be checked, False otherwise.
        """
        step = self.backend.get_currentStep()

        # Regular interval check
        if self.s_interval is not None and step > 0 and step % self.s_interval == 0:
            return True

        # Post-bias check
        if self.next_postbias_check_step is not None and step >= self.next_postbias_check_step:
            return True

        return False

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
            - `self._epot_log`: potential energy log file (`<prefix>_epot.log`)
            - `self._forces_log`: unbiased forces log file (`<prefix>_forces.log`)
            - `self._hess_log`: Hessian log file (`<prefix>_hess.log`)
            - `self._pos_log`: biased-atom positions log file (`<prefix>_pos.log`)

        Notes:
            - Each file handle is checked for existence and open state before closing.
            - Any exceptions during file closure are suppressed to avoid interfering
              with shutdown.

        Returns:
            None
        """
        for attr in ("_evec_log", "_eval_log", "_xyz_log", "_epot_log", "_forces_log", "_hess_log", "_pos_log"):
            f = getattr(self, attr, None)
            if f is not None and not f.closed:
                try:
                    f.close()
                except Exception:
                    warnings.warn(f"Failed to close log file {attr} properly.")
        return None

    def __del__(self) -> None:
        """
        Clean up the log files when the object is garbage-collected

        Returns:
            None
        """
        self._close_logs()
        return None


class GADESForceUpdater(GADESBias):
    def __init__(
        self,
        backend: "Backend",
        biased_force: Any,
        bias_atom_indices: Sequence[int],
        hess_func: Callable,
        clamp_magnitude: float,
        kappa: float,
        interval: int,
        stability_interval: Optional[int] = None,
        logfile_prefix: Optional[str] = None,
        eigensolver: str = "numpy",
        lanczos_iterations: Optional[int] = None,
        use_bofill_update: bool = False,
        full_hessian_interval: Optional[int] = None,
        hvp_epsilon: Optional[float] = None,
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
                Must be created using `createGADESBiasForce()`.
            bias_atom_indices (Sequence[int]):
                Indices of atoms that should receive the bias force.
            hess_func (Callable):
                A user-supplied function returning the Hessian matrix for the system.
                Must accept `(backend, atom_indices, step_size, platform)` as input
                and return a 2D array-like Hessian. Choose one of
                `GADES.utils.compute_hessian_force_fd_richardson` (recommended) or
                `GADES.utils.compute_hessian_force_fd_block_serial`.
            clamp_magnitude (float):
                Maximum allowed magnitude for each atom's bias force vector.
                Forces exceeding this magnitude are rescaled (direction preserved).
                Used to prevent unphysical updates or exploration of irrelevant
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
            eigensolver (str, optional):
                Method for computing the softest eigenmode. Options:
                  - ``'numpy'`` (default): Use ``np.linalg.eigh()`` for full eigendecomposition.
                  - ``'lanczos'``: Use Lanczos iteration to find only the smallest eigenvalue.
                  - ``'lanczos_hvp'``: Matrix-free Lanczos using Hessian-vector products.
            lanczos_iterations (int, optional):
                Number of Lanczos iterations when ``eigensolver='lanczos'`` or ``'lanczos_hvp'``.
            use_bofill_update (bool, optional):
                If True, use Bofill quasi-Newton updates between full Hessian calculations.
            full_hessian_interval (int, optional):
                When ``use_bofill_update=True``, how often (in steps) to compute full Hessian.
            hvp_epsilon (float, optional):
                Finite difference step size for HVP when using ``eigensolver='lanczos_hvp'``.

        Raises:
            TypeError: If `hess_func` is not callable.
            TypeError: If `bias_atom_indices` is not a sequence.
            ValueError: If `bias_atom_indices` is empty or contains non-integer/negative values.
            ValueError: If `clamp_magnitude` is not a positive number.
            ValueError: If `interval` is not a positive integer.
            ValueError: If `stability_interval` is provided but not a positive integer.
            ValueError: If `eigensolver` is not one of 'numpy', 'lanczos', or 'lanczos_hvp'.
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
            eigensolver,
            lanczos_iterations,
            use_bofill_update,
            full_hessian_interval,
            hvp_epsilon,
        )
            
    def describeNextReport(self, simulation: Any) -> tuple[int, bool, bool, bool, bool, bool]:
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
                * `self.next_postbias_check_step` → set to `step + defaults["post_bias_check_delay"]`
                  after applying bias, or cleared when the scheduled post-bias check is reached.
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
                * If unstable (ΔT > `defaults["stability_threshold_temp_diff"]` from target),
                  the bias is removed for safety.
                * If stable and `self.is_biasing` is True, the bias is (re)applied and a
                  post-bias check is scheduled in `defaults["post_bias_check_delay"]` steps.
            - If neither `self.check_stability` nor `self.is_biasing` is set, the method
              performs no action for the current step.
        """
        step = self.backend.get_currentStep()
        post_bias_delay = defaults["post_bias_check_delay"]

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
                self.next_postbias_check_step = step + post_bias_delay

            self.check_stability = False
            self.is_biasing = False
            if step == self.next_postbias_check_step:
                self.next_postbias_check_step = None
            return None

        if self.is_biasing:
            logger.info(f"step {step}] Updating bias forces...")
            self.apply_bias()
            self.is_biasing = False
            self.next_postbias_check_step = step + post_bias_delay
            return None

        # If neither flag is True, do nothing
        return None
    
