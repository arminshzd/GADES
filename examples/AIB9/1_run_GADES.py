# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os

from GADES.utils import compute_hessian_force_fd_richardson as hessian
from GADES import createGADESBiasForce, GADESForceUpdater
from GADES.backend import OpenMMBackend

# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 1e4
BIASED = 0
KAPPA = 0.5
CLAMP_MAGNITUDE = 500
STABILITY_CHECK_FREQ = 200
BIAS_UPDATE_FREQ = 500
LOG_PREFIX = "unbiased" #"GADES"
PLATFORM = "CPU"

# ---------------------------------USER SYSTEM DEF------------------------------
from sys import stdout
import numpy as np
import openmm.app as app
from openmm import unit, Platform
from openmm.openmm import LangevinIntegrator

import time

tstart = time.time()

# LOAD THE SYSTEM TOPOLOGY
gro = app.GromacsGroFile('aib9.gro')
top = app.GromacsTopFile('topol.top')

top.topology.setPeriodicBoxVectors(gro.getPeriodicBoxVectors())

# CHOOSE THE ATOMS TO BIAS
biasing_atom_ids = np.array([atom.index for atom in top.topology.atoms() if atom.name == 'CA'] )
if BIASED:
    print(f"\033[1;32m[GADES] Biasing {len(biasing_atom_ids)} atoms\033[0m")

# SET THE PLATFORM
platform = Platform.getPlatformByName(PLATFORM)

# CREATE SYSTEM OBJECT
system = top.createSystem(nonbondedMethod=app.PME,
                           nonbondedCutoff=1.0*unit.nanometers,
                             constraints=app.HBonds)

# DEFINE INTEGRATOR
integrator = LangevinIntegrator(400 * unit.kelvin,
                                 1 / unit.picosecond,
                                   1 * unit.femtoseconds)
# ADD THE BIAS FORCE TO THE SYSTEM
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

# SET UP THE SIMULATION OBJECT
simulation = app.Simulation(top.topology, system, integrator, platform)
simulation.context.setPositions(gro.positions)

simulation.minimizeEnergy()

# SET UP THE BIASING
if BIASED:
    backend = OpenMMBackend(simulation)
    simulation.reporters.append(
        GADESForceUpdater(
            backend=backend,
            biased_force=GAD_force,
            bias_atom_indices=biasing_atom_ids,
            hess_func=hessian,
            clamp_magnitude=CLAMP_MAGNITUDE,
            kappa=KAPPA,
            interval=BIAS_UPDATE_FREQ,
            stability_interval=STABILITY_CHECK_FREQ,
            logfile_prefix=LOG_PREFIX
            )
        )

# SET UP THE REPORTERS
simulation.reporters.append(
    app.DCDReporter(
        f"traj_{KAPPA}k_{BIAS_UPDATE_FREQ}freq_{CLAMP_MAGNITUDE}clamp_{NSTEPS:.1g}steps.dcd",
        100)
        )
simulation.reporters.append(
    app.StateDataReporter(stdout, 100, step=True, temperature=True,
                           elapsedTime=True, potentialEnergy=True)
                           )

# RUN THE SIMULATION
simulation.step(NSTEPS)

tend = time.time()
print(f"Simulation time: {tend-tstart}")
