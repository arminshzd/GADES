import numpy as np
import openmm
import openmm.app
from openmm import CustomExternalForce, unit, CMMotionRemover


class Backend:
    def __init__(self):
        pass

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
    
class OpenMMBackend(Backend):
    def __init__(self, simulation: openmm.app.Simulation):
        self.simulation = simulation
        self.system = self.simulation.system

    def is_stable(self):
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
        if abs(temperature - target_temperature) > 50:
            return False
        return True

    def get_currentStep(self):
        return self.simulation.currentStep

    def get_atom_symbols(self, atom_symbols: list):
        if atom_symbols is None:
            atom_list = list(self.simulation.topology.atoms())
            self.atom_symbols = [
                atom_list[i].element.symbol if atom_list[i].element is not None else "X"
                for i in self.bias_atom_indices
            ]

    def get_current_state(self):
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        forces_u = state.getForces(asNumpy=True)[self.bias_atom_indices, :]
        positions = state.getPositions(asNumpy=True)

        positions_array = positions.value_in_unit(openmm.unit.nanometer)
        return positions_array, forces_u
    
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
        
        self.simulation.context.setPositions(positions)
        # the `groups` keyword makes sure we're only capturing the forces from the 
        # original pmf and not the biased one.
        state = self.simulation.context.getState(getForces=True, groups={0})
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        return -forces.flatten()

class ASEBackend(Backend):

    def __init__(self, simulation: openmm.app.Simulation):
        self.simulation = simulation
        self.system = self.simulation.system

    def is_stable(self):
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
        if abs(temperature - target_temperature) > 50:
            return False
        return True

    def get_atom_symbols(self, atom_symbols: list):
        if atom_symbols is None:
            atom_list = list(self.simulation.topology.atoms())
            self.atom_symbols = [
                atom_list[i].element.symbol if atom_list[i].element is not None else "X"
                for i in self.bias_atom_indices
            ]

    def get_current_state(self):
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        forces_u = state.getForces(asNumpy=True)[self.bias_atom_indices, :]
        positions = state.getPositions(asNumpy=True)

        positions_array = positions.value_in_unit(openmm.unit.nanometer)
        return positions_array, forces_u
    
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
        
        self.simulation.context.setPositions(positions)
        # the `groups` keyword makes sure we're only capturing the forces from the 
        # original pmf and not the biased one.
        state = self.simulation.context.getState(getForces=True, groups={0})
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        return -forces.flatten()