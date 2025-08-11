# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os
print(os.environ.get("LD_LIBRARY_PATH", ""))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'GADES'))
# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 5e6
PLATFORM = "CUDA"

# ---------------------------------USER SYSTEM DEF------------------------------
from sys import stdout
import numpy as np
import openmm.app as app
from openmm import unit, Platform
from openmm.openmm import LangevinIntegrator

from openmmtorch import TorchForce

import time

tstart = time.time()

# LOAD THE SYSTEM TOPOLOGY
gro = app.GromacsGroFile("../em.gro")
top = app.GromacsTopFile('../topol.top')

top.topology.setPeriodicBoxVectors(gro.getPeriodicBoxVectors())

# SET THE PLATFORM
platform = Platform.getPlatformByName(PLATFORM)
properties = {'DeviceIndex': '1'}

# CREATE SYSTEM OBJECT
system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)

# DEFINE INTEGRATOR
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)

cv1_force = TorchForce(f"cv1.pt")
cv1_force.setUsesPeriodicBoundaryConditions(True)
cv1 = app.BiasVariable(
    cv1_force,
    -3.5,
    3.5,
    biasWidth=1,
    periodic=False,
    gridWidth=1000,
)

cv2_force = TorchForce(f"cv2.pt")
cv2_force.setUsesPeriodicBoundaryConditions(True)
cv2 = app.BiasVariable(
    cv2_force,
    -2,
    2,
    biasWidth=1,
    periodic=False,
    gridWidth=1000,
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
simulation = app.Simulation(top.topology, system, integrator, platform, properties)
simulation.context.setPositions(gro.positions)

if os.path.isfile("checkpnt.chk"):
    simulation.loadCheckpoint('checkpnt.chk')

# SET UP THE REPORTERS
if os.path.isfile("checkpnt.chk"):
    simulation.reporters.append(app.DCDReporter(f"traj_metad.dcd", 100, append=True))
else: 
    simulation.reporters.append(app.DCDReporter(f"traj_metad.dcd", 100, append=False))

simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))
simulation.reporters.append(app.CheckpointReporter('checkpnt.chk', 1e4))

# RUN THE SIMULATION
meta.step(simulation, NSTEPS)
free_energy = meta.getFreeEnergy()
np.save("metad_FE.npy", free_energy)

tend = time.time()
print(f"Simulation time: {tend-tstart}")
