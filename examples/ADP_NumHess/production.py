# production.py — run GAD-biased production from equilibrated structure
import os
from sys import stdout
import time

import numpy as np
import openmm.app as app
from openmm import unit, CustomExternalForce, Platform, MonteCarloBarostat
from openmm.openmm import LangevinIntegrator
import mdtraj as md

from utils import compute_hessian_force_fd_block_serial as hessian
from utils import clamp_force_magnitudes as fclamp

CLAMP_MAGNITUDE = 200
BIAS_UPDATE_FREQ = 1000

openmm_app_path = os.path.join(app.__path__[0], 'data')
# Load topology and positions
#openmm_app_path = '/home/siddarthachar/miniconda3/envs/dmff/lib/python3.9/site-packages/openmm/app/data/'
cwd = "/Users/arminsh/Documents/GADES/examples/ADP_NumHess/"
pdb = app.PDBFile(cwd+"equilibrated.pdb")
traj = md.load(cwd+"equilibrated.pdb")
adp_atom_indices = np.array([atom.index for atom in traj.topology.atoms if atom.residue.name != 'HOH']) # what atoms do you want to pick

forcefield = app.ForceField("amber14/protein.ff14SB.xml", 
                        "amber14/tip3p.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=app.HBonds)

system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

# External force for GAD
biased_force = CustomExternalForce("fx*x+fy*y+fz*z")
biased_force.addPerParticleParameter("fx")
biased_force.addPerParticleParameter("fy")
biased_force.addPerParticleParameter("fz")
for i in range(system.getNumParticles()):
    biased_force.addParticle(i, [0.0, 0.0, 0.0])
system.addForce(biased_force)

integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
platform_name = "CPU"
platform = Platform.getPlatformByName(platform_name)

simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

positions_all = np.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
solvent_indices = np.array([i for i in range(positions_all.shape[0]) if i not in adp_atom_indices]) # this might not be solven coordinates for all systems
positions_adp = positions_all[adp_atom_indices]
positions_solvent = positions_all[solvent_indices]


def dump_pos(step, simulation, output):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    with open(output, 'a') as f:
        f.write(f"------------------- STEP {step} -----------------\n")
        np.savetxt(f, pos, fmt="%.6f")
        f.write("---------------------------------------------\n")

def gad_force_vec(sim, atom_indices, kappa=0.9):
    state = sim.context.getState(getPositions=True, getForces=True)
    forces_u = state.getForces(asNumpy=True)
    positions = state.getPositions(asNumpy=True)
    hess = hessian(sim.system, positions, atom_indices, 1e-6, platform_name)
    w, v = np.linalg.eigh(hess)
    n = v[:, w.argsort()[0]]
    n /= np.linalg.norm(n)
    ## cast n back to the full position vector
    n_new = np.zeros_like(positions)
    n_new[adp_atom_indices] = n.reshape(-1, 3)
    forces_b = -np.dot(n_new.flatten(), forces_u.flatten()) * n_new.flatten() * kappa
    ## clamping biased forces so their abs is never larger than 500
    forces_b = fclamp(forces_b, CLAMP_MAGNITUDE)
    #forces_b = np.zeros_like(forces_b) ## FOR DEBUG. THIS IS MAKING ALL BIAS ZERO. TAKE IT OUT!
    print(5*"-", "DEBUG", 5*"-")
    print("F_max:", np.abs(forces_u).max())
    print("F_b_max:", np.abs(forces_b).max())
    print(5*"-", "DEBUG", 5*"-")
    return forces_b.reshape(positions.shape), n_new

def update_biased_forces(sim, biased_force, adp_atom_indices):
    biased_forces, n_vec = gad_force_vec(sim, adp_atom_indices)
    # print('-------biased_forces: ', biased_forces)
    for i in range(sim.system.getNumParticles()):
        biased_force.setParticleParameters(i, i, tuple(biased_forces[i]))
    biased_force.updateParametersInContext(sim.context)
    return n_vec

# Reporters
class PosReporter:
    def __init__(self, simulation, interval, output):
        self.simulation = simulation
        self.interval = interval
        self.output = output

    def describeNextReport(self, simulation):
        """
        Return the number of steps until the next report and what data to collect.
        The five `False` values indicate that this reporter does not require any specific data.
        """
        return (self.interval, False, False, False, False, False)

    def report(self, simulation, state):
        """
        This function will be called at intervals specified in describeNextReport().
        It updates the biased forces using the provided function.
        """
        dump_pos(self.interval, self.simulation, self.output)


simulation.reporters.append(app.DCDReporter(cwd+"traj.dcd", 100))
simulation.reporters.append(PosReporter(simulation, 1, cwd+"debug.txt"))
simulation.reporters.append(app.StateDataReporter(cwd+"scalars.csv", 1, time=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True, totalEnergy=True))

# Main simulation loop
n_steps = int(1e6)

n_vec_list = []

start = time.time()
for step in range(0, n_steps, BIAS_UPDATE_FREQ):
    if step > 0:
        print(f"[step {step}] Updating GAD forces...")
        n_vec = update_biased_forces(simulation, biased_force, adp_atom_indices)
    simulation.step(BIAS_UPDATE_FREQ)
end = time.time()

print("[production] GAD-biased simulation complete. Time elapsed (s):", end - start)
np.savetxt(cwd+'time_gad_solvated.dat', np.array([end-start]))
