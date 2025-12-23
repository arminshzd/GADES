import numpy as np
import openmm
import openmm.app
from openmm import CustomExternalForce, unit, CMMotionRemover
from ase.calculators.calculator import Calculator, all_changes

class Backend:
    def __init__(self):
        self.name = ""

    def is_stable(self) -> bool:
        return True

    def get_currentStep(self):
        return 0
    
    def get_atom_symbols(self, atom_symbols: list):
        pass

    def get_current_state(self):
        pass

    def run(self, nsteps: int):
        # Implement the update logic here
        pass

    def apply_bias(self, bias_force_object, biased_force_values, bias_atom_indices: list):
        pass

    def remove_bias(self, bias_force_object, bias_atom_indices: list):
        pass
    
class OpenMMBackend(Backend):
    def __init__(self, simulation: openmm.app.Simulation):
        self.simulation = simulation
        self.system = self.simulation.system
        self.name = "openmm"

    def is_stable(self):
        """
        This method estimates the instantaneous temperature from the system's
        kinetic energy and compares it to the target temperature of the integrator.
        If the deviation exceeds 50 K, the system is considered unstable.

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

        # this criterion can be adjusted if needed
        if abs(temperature - target_temperature) > 50:
            return False
        return True

    def get_currentStep(self):
        return self.simulation.currentStep

    def get_atoms(self):
        return self.simulation.topology.atoms()

    def get_atom_symbols(self, bias_atom_indices: list) -> list:
        atom_list = list(self.simulation.topology.atoms())
        atom_symbols = [
            atom_list[i].element.symbol if atom_list[i].element is not None else "X"
            for i in bias_atom_indices
        ]
        return atom_symbols

    def get_current_state(self):
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        positions = state.getPositions(asNumpy=True)

        positions_array = positions.value_in_unit(openmm.unit.nanometer)
        return positions_array, forces

    def get_forces(self, positions) -> np.ndarray:
        """
        Compute the original (unbiased) forces from an OpenMM context (internal use only).

        This function updates the context with the provided positions, then retrieves
        forces from force group `0` only. Group `0` is assumed to correspond to the
        system's original potential (e.g., the PMF) without additional bias terms.
        The forces are converted to units of kJ/mol/nm and flattened into a 1D array.

        Args:
            context (openmm.Context):
                The OpenMM context containing the current system and integrator state.
            positions (openmm.unit.Quantity):
                Atomic positions, shaped `(N, 3)` with distance units compatible with OpenMM.

        Returns:
            np.ndarray:
                Flattened force vector of shape `(3 * N,)`, in units of kJ/mol/nm.

        Notes:
            - By restricting to `groups={0}`, the returned forces exclude any
            externally applied bias forces (e.g., from GADES).
        """
        positions = positions * openmm.unit.nanometer
        self.simulation.context.setPositions(positions)
        # the `groups` keyword makes sure we're only capturing the forces from the 
        # original pmf and not the biased one.
        state = self.simulation.context.getState(getForces=True, groups={0})
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        return -forces.flatten()

    def apply_bias(self, bias_force_object, biased_force_values, bias_atom_indices: list):
        """
        Apply the bias forces to the specified atoms in the OpenMM simulation.
        :param bias_force: CustomExternalForce
        :param biased_forces: np.ndarray
        :param bias_atom_indices: list a list of atom indices to apply the bias to
        :type bias_atom_indices: list
        """
        for i, idx in enumerate(bias_atom_indices):
            bias_force_object.setParticleParameters(idx, idx, tuple(biased_force_values[i]))
        bias_force_object.updateParametersInContext(self.simulation.context)

    def remove_bias(self, bias_force_object, bias_atom_indices: list):
        """
        Apply the bias forces to the specified atoms in the OpenMM simulation.
        :param bias_force_object: CustomExternalForce
        :param bias_atom_indices: list a list of atom indices to remove the bias
        :type bias_atom_indices: list
        """
        for idx in bias_atom_indices:
            bias_force_object.setParticleParameters(idx, idx, (0.0, 0.0, 0.0))
        bias_force_object.updateParametersInContext(self.simulation.context)


class GADESCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calc: Calculator, gades_force_updater):
        super().__init__()
        self.base_calc = base_calc
        self.force_updater = gades_force_updater
        self.atoms = base_calc.atoms
        self._name = "gades_calculator"

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        # Let base calculator do its job
        self.base_calc.calculate(atoms, properties, system_changes)

        self.results = self.base_calc.results.copy()
       
        if 'forces' in self.results:
            bias = self.force_updater.get_gad_force(self.force_updater.backend)
            self.results['forces'] = self.results['forces'] + bias

class ASEBackend(Backend):
    def __init__(self, calculator: Calculator, atoms):
        self.calculator = calculator
        self.base_calc = calculator.base_calc
        self.atoms = atoms
        self.name = "ase"

    def is_stable(self):
        return True

    def get_atoms(self):
        return list(self.atoms)

    def get_atom_symbols(self, bias_atom_indices: list) -> list:
        atom_list = list(self.atoms)
        atom_symbols = [
            atom_list[i].symbol if atom_list[i].symbol is not None else "X"
            for i in bias_atom_indices
        ]
        return atom_symbols

    def get_current_state(self):
        positions = self.atoms.get_positions()
        self.base_calc.calculate(atoms=self.atoms, properties=['forces'])
        forces = self.base_calc.results['forces']
        return positions, forces
    
    def get_forces(self, positions) -> np.ndarray:
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
        """
        self.atoms.set_positions(positions)
        self.base_calc.calculate(atoms=self.atoms, properties=['forces'])
        forces = self.base_calc.results['forces']
        return -forces.flatten()



    