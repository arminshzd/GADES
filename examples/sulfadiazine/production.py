# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'GADES'))

from utils import compute_hessian_force_fd_richardson as hessian
from gades import getGADESBiasForce
from gades import GADESForceUpdater

# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 5e6
BIASED = 0
KAPPA = 0.9
CLAMP_MAGNITUDE = 2500
STABILITY_CHECK_FREQ = 500
BIAS_UPDATE_FREQ = 2000
LOG_PREFIX = "GADES"
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
gro = app.GromacsGroFile("em.gro")
top = app.GromacsTopFile('topol.top')

top.topology.setPeriodicBoxVectors(gro.getPeriodicBoxVectors())

# CHOOSE THE ATOMS TO BIAS
biasing_atom_ids = np.array([atom.index for atom in top.topology.atoms() if
                              (atom.residue.name != 'HOH' and atom.element.symbol != "H")])
if BIASED:
    print(f"\033[1;32m[GADES] Biasing {len(biasing_atom_ids)} atoms\033[0m")

# SET THE PLATFORM
platform = Platform.getPlatformByName(PLATFORM)
properties = {'DeviceIndex': '1'}

# CREATE SYSTEM OBJECT
system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)

# DEFINE INTEGRATOR
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)

# ADD THE BIAS FORCE TO THE SYSTEM
GAD_force = getGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

# SET UP THE SIMULATION OBJECT
simulation = app.Simulation(top.topology, system, integrator, platform, properties)
simulation.context.setPositions(gro.positions)

# SET UP THE REPORTERS
#simulation.reporters.append(app.DCDReporter(f"traj_{BIAS_UPDATE_FREQ}_{CLAMP_MAGNITUDE}.dcd", 500))
simulation.reporters.append(app.DCDReporter(f"traj_unbiased.dcd", 500))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))

# SET UP THE BIASING
if BIASED:
    simulation.reporters.append(GADESForceUpdater(biased_force=GAD_force, bias_atom_indices=biasing_atom_ids, hess_func=hessian, clamp_magnitude=CLAMP_MAGNITUDE, kappa=KAPPA, interval=BIAS_UPDATE_FREQ, stability_interval=STABILITY_CHECK_FREQ, logfile_prefix=LOG_PREFIX))

# RUN THE SIMULATION
simulation.step(NSTEPS)

tend = time.time()
print(f"Simulation time: {tend-tstart}")
