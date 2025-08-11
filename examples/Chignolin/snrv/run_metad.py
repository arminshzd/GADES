# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'GADES'))
# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 5e6
PLATFORM = "CUDA"

# ---------------------------------USER SYSTEM DEF------------------------------
from sys import stdout
import numpy as np
import openmm.app as app
from openmm import unit, Platform, MonteCarloBarostat
from openmm.openmm import LangevinIntegrator

from openmmtorch import TorchForce

import time

tstart = time.time()

# LOAD THE SYSTEM TOPOLOGY
pdb = app.PDBFile("/home/armin/Documents/GADES/examples/Chignolin/equilibrated.pdb")

# DEFINE FORCEFIELD
forcefield = app.ForceField("amber14/protein.ff14SB.xml", 
                        "amber14/tip3p.xml")

# SET THE PLATFORM
platform = Platform.getPlatformByName(PLATFORM)

# CREATE SYSTEM OBJECT
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=app.HBonds)

# DEFINE INTEGRATOR
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)

cv1_force = TorchForce(f"cv1.pt")
cv1_force.setUsesPeriodicBoundaryConditions(True)
cv1 = app.BiasVariable(
    cv1_force,
    -5000,
    10000,
    biasWidth=100,
    periodic=False,
    gridWidth=500,
)

cv2_force = TorchForce(f"cv2.pt")
cv2_force.setUsesPeriodicBoundaryConditions(True)
cv2 = app.BiasVariable(
    cv2_force,
    -5000,
    10000,
    biasWidth=100,
    periodic=False,
    gridWidth=500,
)

meta = app.Metadynamics(
    system,
    [cv1, cv2],
    300.0,
    biasFactor=6.0,
    height=1.0,
    frequency=500,
    saveFrequency=500,
    biasDir=f"biases/",
)

# SET UP THE SIMULATION OBJECT
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# SET UP THE REPORTERS
simulation.reporters.append(app.DCDReporter(f"traj_metad.dcd", 500))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))

# RUN THE SIMULATION
meta.step(simulation, 5000000)
free_energy = meta.getFreeEnergy()
np.save("metad_FE.npy", free_energy)

tend = time.time()
print(f"Simulation time: {tend-tstart}")
