import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import openmm
import openmm.app
from openmm import unit, CMMotionRemover
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

from .config import defaults

if TYPE_CHECKING:
    from .gades import GADESBias


class Backend:
    """
    A generic interface for the backends to be used with GADES.
    """

    name: str

    def __init__(self) -> None:
        self.name = ""

    def is_stable(self) -> bool:
        return True

    def get_currentStep(self) -> int:
        return 0

    def get_atom_symbols(self, bias_atom_indices: Sequence[int]) -> List[str]:
        return []

    def get_positions(self) -> np.ndarray:
        """
        Retrieve the current atom positions.

        Returns:
            np.ndarray: Positions array with shape ``(N, 3)``.
        """
        raise NotImplementedError

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_bias(
        self,
        bias_force_object: Any,
        biased_force_values: np.ndarray,
        bias_atom_indices: Sequence[int],
    ) -> None:
        pass

    def remove_bias(
        self, bias_force_object: Any, bias_atom_indices: Sequence[int]
    ) -> None:
        pass
    
class OpenMMBackend(Backend):
    """
    A wrapper for OpenMM to be used as a backend for GADES.

    Args:
        simulation: The OpenMM Simulation object.
    """

    simulation: openmm.app.Simulation
    system: openmm.System

    def __init__(self, simulation: openmm.app.Simulation) -> None:
        self.simulation = simulation
        self.system = self.simulation.system
        self.name = "openmm"

    def is_stable(self) -> bool:
        """
        This method estimates the instantaneous temperature from the system's
        kinetic energy and compares it to the target temperature of the integrator.
        If the deviation exceeds the stability threshold (configurable via
        ``defaults["stability_threshold_temp_diff"]``), the system is considered unstable.

        Check if the simulation is stable by evaluating the instantaneous temperature
          return True if stable, False otherwise

        For a small number of biased DOFs, this criterion might give false positives.
        """
        dof = 0
        state = self.simulation.context.getState(getEnergy=True)

        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i) > 0*unit.dalton:
                dof += 3
        for i in range(self.system.getNumConstraints()):
            p1, p2, distance = self.system.getConstraintParameters(i)
            if self.system.getParticleMass(p1) > 0*unit.dalton or self.system.getParticleMass(p2) > 0*unit.dalton:
                dof -= 1
        if any(type(self.system.getForce(i)) == CMMotionRemover for i in range(self.system.getNumForces())):
            dof -= 3
        temperature = (2*state.getKineticEnergy()/(dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)
        target_temperature = self.simulation.integrator.getTemperature().value_in_unit(unit.kelvin)

        threshold = defaults["stability_threshold_temp_diff"]
        if abs(temperature - target_temperature) > threshold:
            return False
        return True

    def get_currentStep(self) -> int:
        return self.simulation.currentStep

    def get_atoms(self) -> Any:
        return self.simulation.topology.atoms()

    def get_atom_symbols(self, bias_atom_indices: Sequence[int]) -> List[str]:
        atom_list = list(self.simulation.topology.atoms())
        atom_symbols = [
            atom_list[i].element.symbol if atom_list[i].element is not None else "X"
            for i in bias_atom_indices
        ]
        return atom_symbols

    def get_positions(self) -> np.ndarray:
        """Retrieve the current atom positions from the OpenMM context."""
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        return positions.value_in_unit(openmm.unit.nanometer)

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        positions = state.getPositions(asNumpy=True)

        positions_array = positions.value_in_unit(openmm.unit.nanometer)
        return positions_array, forces

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the original (unbiased) forces from an OpenMM context (internal use only).

        This function updates the context with the provided positions, then retrieves
        forces from force group `0` only. Group `0` is assumed to correspond to the
        system's original potential (e.g., the PMF) without additional bias terms.
        The forces are converted to units of kJ/mol/nm and flattened into a 1D array.

        Args:
            positions: Atomic positions, shaped ``(N, 3)`` in nanometers.

        Returns:
            Flattened force vector of shape ``(3 * N,)``, in units of kJ/mol/nm.

        Notes:
            By restricting to ``groups={0}``, the returned forces exclude any
            externally applied bias forces (e.g., from GADES).
            Original positions are restored after force computation.
        """
        # Save original positions
        original_positions = self.get_positions()

        positions = positions * openmm.unit.nanometer
        self.simulation.context.setPositions(positions)
        # the `groups` keyword makes sure we're only capturing the forces from the
        # original pmf and not the biased one.
        state = self.simulation.context.getState(getForces=True, groups={0})
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)

        # Restore original positions
        self.simulation.context.setPositions(original_positions * openmm.unit.nanometer)

        # negating the forces does not affect the difference between f and f0 calculation in GADES get_gad_force()
        # the hessian calculators use dU/dx = - force for calculations
        return -forces.flatten()

    def apply_bias(
        self,
        bias_force_object: openmm.CustomExternalForce,
        biased_force_values: np.ndarray,
        bias_atom_indices: Sequence[int],
    ) -> None:
        """
        Apply the bias forces to the specified atoms in the OpenMM simulation.

        Args:
            bias_force_object: CustomExternalForce object.
            biased_force_values: Array of bias forces, shape ``(N_biased, 3)``.
            bias_atom_indices: List of atom indices to apply the bias to.
        """
        for i, idx in enumerate(bias_atom_indices):
            bias_force_object.setParticleParameters(idx, idx, tuple(biased_force_values[i]))
        bias_force_object.updateParametersInContext(self.simulation.context)

    def remove_bias(
        self,
        bias_force_object: openmm.CustomExternalForce,
        bias_atom_indices: Sequence[int],
    ) -> None:
        """
        Remove the bias forces from the specified atoms in the OpenMM simulation.

        Args:
            bias_force_object: CustomExternalForce object.
            bias_atom_indices: List of atom indices to remove the bias from.
        """
        for idx in bias_atom_indices:
            bias_force_object.setParticleParameters(idx, idx, (0.0, 0.0, 0.0))
        bias_force_object.updateParametersInContext(self.simulation.context)


class GADESCalculator(Calculator):
    """
    ASE Calculator wrapper that adds GADES bias forces to an existing ASE Calculator.

    Args:
        base_calc: The base ASE Calculator to which GADES bias forces will be added.
        gades_force_updater: The GADES force updater object responsible for computing
            and applying bias forces.
    """

    implemented_properties = ['energy', 'forces']

    base_calc: Calculator
    force_updater: Any  # GADESBias, but avoiding circular import

    def __init__(self, base_calc: Calculator, gades_force_updater: Any) -> None:
        super().__init__()
        self.base_calc = base_calc
        self.force_updater = gades_force_updater
        self.atoms = base_calc.atoms
        self._name = "gades_calculator"

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Tuple[str, ...] = ('energy', 'forces'),
        system_changes: List[str] = all_changes,
    ) -> None:
        # Let base calculator do its job
        self.base_calc.calculate(atoms, properties, system_changes)

        self.results = self.base_calc.results.copy()

        if self.force_updater is not None and 'forces' in self.results:
            if self.force_updater.applying_bias():
                bias = self.force_updater.get_gad_force()
                # Create full-size bias array for partial atom biasing
                full_bias = np.zeros_like(self.results['forces'])
                full_bias[self.force_updater.bias_atom_indices, :] = bias
                self.results['forces'] = self.results['forces'] + full_bias

class ASEBackend(Backend):
    """
    A wrapper for ASE to be used as a backend for GADES.

    Args:
        calculator: The custom ASE Calculator (i.e. GADESCalculator) that includes
            GADES bias forces.
        atoms: The ASE Atoms object.
        target_temperature: Target temperature in Kelvin for stability checking.
            If not provided, the backend will attempt to read it from the integrator
            (works for Langevin, NVTBerendsen, NPTBerendsen). If neither is available,
            stability checking will be skipped with a warning.
    """

    base_calc: Calculator
    atoms: Atoms
    integrator: Any
    current_step: int
    target_temperature: Optional[float]
    _stability_warning_issued: bool
    gades_bias: Optional["GADESBias"]

    def __init__(
        self,
        calculator: GADESCalculator,
        atoms: Atoms,
        target_temperature: Optional[float] = None,
    ) -> None:
        self.base_calc = calculator.base_calc
        self.atoms = atoms
        atoms.calc = calculator
        self.name = "ase"

        self.integrator = None
        self.current_step = -1
        self.target_temperature = target_temperature
        self._stability_warning_issued = False
        self.gades_bias = None

    def get_atoms(self) -> List[Any]:
        """
        Return the list of atoms in the ASE Atoms object.

        Each entry is an Atom object, e.g.
        ``Atom('Ar', [0.0, 0.0, 0.0], index=0)``.
        """
        return list(self.atoms)

    def get_atom_symbols(self, bias_atom_indices: Sequence[int]) -> List[str]:
        """
        Return the atomic symbols for the specified atom indices.
        """
        atom_list = list(self.atoms)
        atom_symbols = [
            atom_list[i].symbol if atom_list[i].symbol is not None else "X"
            for i in bias_atom_indices
        ]
        return atom_symbols

    def get_currentStep(self) -> int:
        """
        Return the current MD step from the integrator.

        Returns:
            Current step number, or -1 if there is no integrator associated
            with the backend.
        """
        if self.integrator is not None:
            self.current_step = self.integrator.nsteps
        else:
            self.current_step = -1
        return self.current_step

    def get_positions(self) -> np.ndarray:
        """Retrieve the current atom positions."""
        return self.atoms.get_positions()

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the current atom positions and total forces (including bias).

        Returns:
            Tuple of (positions, forces) arrays, both with shape ``(N, 3)``.
        """
        positions = self.atoms.get_positions()
        forces = self.atoms.get_forces()  # Total forces via GADESCalculator
        return positions, forces

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the original (unbiased) forces from the base calculator of my calculator.
        The calculate() funtion of my calculator with the biasing forces
        will be used by the integrator to advance the atom positions.

        This function updates the base calculator with the provided positions,
        then retrieves forces. The forces are returned as a 1D array.

        This function is called by GADES get_gad_force() for the perturbed positions.

        Args:
            positions (np.ndarray):
                Atomic positions, shaped `(N, 3)`.

        Notes:
            Original positions are restored after force computation.
        """
        # Save original positions
        original_positions = self.atoms.get_positions()

        self.atoms.set_positions(positions)
        self.base_calc.calculate(atoms=self.atoms, properties=['forces'], system_changes=all_changes)
        forces = self.base_calc.results['forces']

        # Restore original positions
        self.atoms.set_positions(original_positions)

        # negating the forces does not affect the difference between f and f0 calculation in GADES get_gad_force()
        return -forces.flatten()

    def _get_target_temperature(self) -> Optional[float]:
        """
        Get the target temperature for stability checking.

        Returns the target temperature in Kelvin, trying in order:

        1. ``self.target_temperature`` if set explicitly
        2. Integrator's temperature attribute (for NVT/NPT integrators)
        3. ``None`` if neither is available

        Returns:
            Target temperature in Kelvin, or ``None`` if unavailable.
        """
        # First check if explicitly set
        if self.target_temperature is not None:
            return self.target_temperature

        # Try to get from integrator
        if self.integrator is not None:
            # ASE Langevin integrator stores temperature in 'temp' attribute (in Kelvin)
            if hasattr(self.integrator, 'temp'):
                return self.integrator.temp
            # NVTBerendsen and NPTBerendsen store it in 'temperature'
            if hasattr(self.integrator, 'temperature'):
                return self.integrator.temperature

        return None

    def is_stable(self) -> bool:
        """
        Check if the simulation is stable by comparing instantaneous temperature
        to the target temperature.

        The system is considered unstable if the temperature deviates by more than
        ``defaults["stability_threshold_temp_diff"]`` from the target. If no target
        temperature is available (not set explicitly and cannot be read from the
        integrator), a warning is issued once and the method returns True (stability
        check skipped).

        Returns:
            bool: True if stable or if stability check is skipped, False if unstable.
        """
        target_temp = self._get_target_temperature()

        if target_temp is None:
            if not self._stability_warning_issued:
                warnings.warn(
                    "ASEBackend: Cannot perform stability check - no target temperature available. "
                    "Either set target_temperature in ASEBackend constructor or use an NVT/NPT integrator. "
                    "Stability checking will be skipped.",
                    UserWarning
                )
                self._stability_warning_issued = True
            return True

        current_temp = self.atoms.get_temperature()

        threshold = defaults["stability_threshold_temp_diff"]
        if abs(current_temp - target_temp) > threshold:
            return False
        return True

    @classmethod
    def with_gades(
        cls,
        atoms: Atoms,
        base_calc: Calculator,
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
        target_temperature: Optional[float] = None,
    ) -> "ASEBackend":
        """
        Factory method to create an ASEBackend with GADES bias fully configured.

        This method handles all the internal wiring between GADESBias, GADESCalculator,
        and ASEBackend, eliminating the need for manual post-initialization patching.

        Args:
            atoms: The ASE Atoms object.
            base_calc: The base ASE Calculator (e.g., LAMMPS, EMT).
            bias_atom_indices: Indices of atoms that should receive the bias force.
            hess_func: Function to compute the Hessian matrix.
            clamp_magnitude: Maximum magnitude for bias force components.
            kappa: Scaling factor (0 < κ < 1) for the bias force.
            interval: Number of steps between bias force updates.
            stability_interval: Steps between stability checks (optional).
            logfile_prefix: Prefix for log files (optional).
            eigensolver: Method for computing softest eigenmode ('numpy', 'lanczos', or 'lanczos_hvp').
            lanczos_iterations: Number of Lanczos iterations (if using 'lanczos' or 'lanczos_hvp').
            use_bofill_update: Whether to use Bofill Hessian updates.
            full_hessian_interval: Steps between full Hessian recomputation (if using Bofill).
            hvp_epsilon: Finite difference step size for HVP (if using 'lanczos_hvp').
            target_temperature: Target temperature in Kelvin for stability checking.

        Returns:
            Fully configured ASEBackend with ``gades_bias`` attribute accessible.

        Example:
            >>> from GADES.backend import ASEBackend
            >>> from GADES.utils import compute_hessian_force_fd_richardson as hessian
            >>>
            >>> backend = ASEBackend.with_gades(
            ...     atoms=atoms,
            ...     base_calc=lammps_calc,
            ...     bias_atom_indices=biasing_atom_ids,
            ...     hess_func=hessian,
            ...     clamp_magnitude=1000,
            ...     kappa=0.9,
            ...     interval=100,
            ...     stability_interval=1000,
            ... )
            >>> backend.integrator = dyn  # Attach integrator for step tracking
        """
        # Import here to avoid circular import at module load time
        from .gades import GADESBias

        # Step 1: Create GADESBias with backend=None (will be set later)
        gades_bias = GADESBias(
            backend=None,  # type: ignore[arg-type]
            biased_force=None,
            bias_atom_indices=bias_atom_indices,
            hess_func=hess_func,
            clamp_magnitude=clamp_magnitude,
            kappa=kappa,
            interval=interval,
            stability_interval=stability_interval,
            logfile_prefix=logfile_prefix,
            eigensolver=eigensolver,
            lanczos_iterations=lanczos_iterations,
            use_bofill_update=use_bofill_update,
            full_hessian_interval=full_hessian_interval,
            hvp_epsilon=hvp_epsilon,
        )

        # Step 2: Create GADESCalculator wrapping the base calculator
        gades_calc = GADESCalculator(base_calc, gades_bias)

        # Step 3: Create ASEBackend instance
        backend = cls(gades_calc, atoms, target_temperature=target_temperature)

        # Step 4: Wire up the backend reference in GADESBias
        gades_bias.backend = backend

        # Step 5: Store reference for user access
        backend.gades_bias: "GADESBias" = gades_bias

        return backend
