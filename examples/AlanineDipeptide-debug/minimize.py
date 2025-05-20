# minimize.py — build OpenMM system and run energy minimization
from openmm.app import *
from openmm import unit, Platform, MonteCarloBarostat
from openmm.openmm import LangevinIntegrator

pdb = PDBFile("init.pdb")
forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)
system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
platform = Platform.getPlatformByName("CPU")
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

positions = simulation.context.getState(getPositions=True).getPositions()
with open("minimized.pdb", "w") as f:
    PDBFile.writeFile(pdb.topology, positions, f)

print("Minimization complete. Output written to minimized.pdb")