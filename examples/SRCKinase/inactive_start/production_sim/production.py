# ---------------------------------MODULE IMPORTS-------------------------------
import sys
import os

os.chdir(os.path.dirname(__file__))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'GADES'))

# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # keep this line
# repo_root = '/home/siddarthachar/gad-linux/GADES'
# sys.path.insert(0, os.path.join(repo_root, 'GADES'))
# sys.path.insert

from utils import compute_hessian_force_fd_block_serial as hessian
from utils import compute_hessian_force_fd_richardson as hessian # CHECK

from gades import getGADESBiasForce
from gades import GADESForceUpdater

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import openmmplumed
import mdtraj as md
import numpy as np
from openmm.app import modeller
from openmm.app.pdbfile import PDBFile
import pandas as pd
import matplotlib.pyplot as plt
from parmed import load_file

# -----------------------------SIMULATION PARAMETERS----------------------------
NSTEPS = 1e8
BIASED = 1
KAPPA = 0.5
CLAMP_MAGNITUDE = 2500
STABILITY_CHECK_FREQ = 1000
BIAS_UPDATE_FREQ = 2000
LOG_PREFIX = "2src"
PLATFORM = "OpenCL"

# ---------------------------------USER SYSTEM DEF------------------------------
directory = '../'

gro = load_file(directory + '2src_npt.gro')
top = load_file(directory + 'topol.top')

top.box = gro.box[:] # load periodic boundary definition from .gro

# plumed_file = directory + 'plumed_script.dat'
sim_temp = 300.00 # simulation temperature in kelvin
prefix = 'production_run_'
integration_timestep = 0.002 # 2 femtoseconds
# simulation_steps = 1e8 # 10 nanoseconds -> increase this to run longer simulations
NVT_steps =  50000 # 1 ns NVT -> no need to change
NPT_steps =  100000 # 2 ns NPT equilibration -> no need to change
traj_frame_freq = 5000 # saves all atom coordinates as .dcd trajectory every 10 picoseconds
colvar_save_freq = 100 # computes and saves phi,psi dihedral angles every 200 femtoseconds
stdout_freq = 10000 # prints key system indicators every 20 picoseconds in console  -> no need to change
################################################################
platform = Platform.getPlatformByName("OpenCL")

biasing_atom_ids = np.array([atom.index for atom in top.topology.atoms() if atom.name == 'CA' and atom.residue.name != 'HOH'])
if BIASED:
    print(f"[GADES] Biasing {len(biasing_atom_ids)} atoms")
    

system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers, constraints=app.HBonds)

# ADD THE BIAS FORCE TO THE SYSTEM
GAD_force = getGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

integrator = LangevinMiddleIntegrator(sim_temp*kelvin, 1/picosecond, integration_timestep*picoseconds)
simulation = Simulation(top.topology, system, integrator,platform)
simulation.context.setPositions(gro.positions)

print("Minimizing energy")
simulation.minimizeEnergy()
lastpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(top.topology, lastpositions, open('minimized.pdb', 'w'))

print("Running NVT")
simulation.step(NVT_steps)
lastpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(top.topology, lastpositions, open('equilibrated_nvt.pdb', 'w'))

system.addForce(MonteCarloBarostat(1*bar, sim_temp*kelvin))
simulation.context.reinitialize(preserveState=True)
print("Running NPT")
simulation.step(NPT_steps)
lastpositions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(top.topology, lastpositions, open('equilibrated_npt.pdb', 'w'))

print("System setup (energy minimization + equilibration) complete!")

print("Starting production simulation!")

# SET UP THE REPORTERS
simulation.reporters.append(app.DCDReporter("traj.dcd", 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, potentialEnergy=True))

# SET UP THE BIASING
if BIASED:
    simulation.reporters.append(GADESForceUpdater(biased_force=GAD_force, bias_atom_indices=biasing_atom_ids, hess_func=hessian, clamp_magnitude=CLAMP_MAGNITUDE, kappa=KAPPA, interval=BIAS_UPDATE_FREQ, stability_interval=STABILITY_CHECK_FREQ, logfile_prefix=LOG_PREFIX))

# RUN THE SIMULATION
simulation.step(NSTEPS)

    