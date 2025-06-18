# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'GADES'))

from utils import compute_hessian_force_fd_block_serial as hessian
from gades import getGADESBiasForce
from gades import GADESForceUpdater

# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 1e6
BIASED = 1
KAPPA = 0.9
CLAMP_MAGNITUDE = 1000
STABILITY_CHECK_FREQ = 1000
BIAS_UPDATE_FREQ = 2000
LOG_PREFIX = "log_prefix"
PLATFORM = "CPU"

# ---------------------------------USER SYSTEM DEF------------------------------
from sys import stdout
import numpy as np
import openmm.app as app
from openmm import unit, Platform, MonteCarloBarostat, AndersenThermostat
from openmm.openmm import LangevinIntegrator, VerletIntegrator

openmm_app_path = os.path.join(app.__path__[0], 'data')

# LOAD THE SYSTEM TOPOLOGY
pdb = app.PDBFile("topology.pdb")
# CHOOSE THE ATOMS TO BIAS
biasing_atom_ids = np.array([atom.index for atom in pdb.topology.atoms() if atom.residue.name != 'HOH'])
if BIASED:
    print(f"[GADES] Biasing {len(biasing_atom_ids)} atoms")

# DEFINE FORCEFIELD
forcefield = app.ForceField("amber14/protein.ff14SB.xml", 
                        "amber14/tip3p.xml")

# SET THE PLATFORM
platform = Platform.getPlatformByName(PLATFORM)

# CREATE SYSTEM OBJECT
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=app.HBonds)

# DEFINE INTEGRATOR
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
#integrator = VerletIntegrator(2 * unit.femtoseconds)

# DEFINE THERMOSTAT (if needed)
#thermostat = AndersenThermostat(300 * unit.kelvin, 1 / unit.picosecond)
#system.addForce(thermostat)

# DEFINE BAROSTAT (if needed)
barostat = MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin)
system.addForce(barostat)

# ADD THE BIAS FORCE TO THE SYSTEM
GAD_force = getGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

# SET UP THE SIMULATION OBJECT
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# SET UP THE REPORTERS
simulation.reporters.append(app.DCDReporter("traj.dcd", 100))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))

# SET UP THE BIASING
if BIASED:
    simulation.reporters.append(GADESForceUpdater(biased_force=GAD_force, bias_atom_indices=biasing_atom_ids, hess_func=hessian, clamp_magnitude=CLAMP_MAGNITUDE, kappa=KAPPA, interval=BIAS_UPDATE_FREQ, stability_interval=STABILITY_CHECK_FREQ, logfile_prefix=LOG_PREFIX))

# RUN THE SIMULATION
simulation.step(NSTEPS)
