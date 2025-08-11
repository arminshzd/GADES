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
import openmm.app as app
from openmm import unit, Platform
from openmm.openmm import LangevinIntegrator

import time

tstart = time.time()

# LOAD THE SYSTEM TOPOLOGY
gro = app.GromacsGroFile('../aib9.gro')
top = app.GromacsTopFile('../topol.top')

top.topology.setPeriodicBoxVectors(gro.getPeriodicBoxVectors())

# SET THE PLATFORM
platform = Platform.getPlatformByName(PLATFORM)

# CREATE SYSTEM OBJECT
system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)

# DEFINE INTEGRATOR
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)

# SET UP THE SIMULATION OBJECT
simulation = app.Simulation(top.topology, system, integrator, platform)
simulation.context.setPositions(gro.positions)

simulation.minimizeEnergy()

# SET UP THE REPORTERS
simulation.reporters.append(app.DCDReporter(f"traj_unbiased.dcd", 500))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))

# RUN THE SIMULATION
simulation.step(NSTEPS)

tend = time.time()
print(f"Simulation time: {tend-tstart}")
