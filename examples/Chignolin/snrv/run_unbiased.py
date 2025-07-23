# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'GADES'))
# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 1e6
PLATFORM = "CUDA"

# ---------------------------------USER SYSTEM DEF------------------------------
from sys import stdout
import numpy as np
import openmm.app as app
from openmm import unit, Platform
from openmm.openmm import LangevinIntegrator

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

# SET UP THE SIMULATION OBJECT
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# SET UP THE REPORTERS
simulation.reporters.append(app.DCDReporter(f"traj_unbiased.dcd", 500))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))

# RUN THE SIMULATION
simulation.step(5000000)

tend = time.time()
print(f"Simulation time: {tend-tstart}")
