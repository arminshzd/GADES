# equilibrate.py — run short equilibration on minimized system
from openmm.app import *
from openmm import unit, MonteCarloBarostat, Platform
from openmm.openmm import LangevinIntegrator

pdb = PDBFile("minimized.pdb")
forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)
system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
platform = Platform.getPlatformByName("CPU")
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

simulation.reporters.append(DCDReporter("equilibrated.dcd", 100))
simulation.reporters.append(StateDataReporter("equilibration.csv", 100, step=True, potentialEnergy=True, temperature=True))

simulation.step(100000)

positions = simulation.context.getState(getPositions=True).getPositions()
with open("equilibrated.pdb", "w") as f:
    PDBFile.writeFile(pdb.topology, positions, f)

print("Equilibration complete. Output written to equilibrated.pdb")
