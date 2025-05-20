import numpy as np
import jax.numpy as jnp
from jax import grad, hessian, lax
from openmm.app import *
from openmm import unit, CustomExternalForce, Platform, MonteCarloBarostat
from openmm.openmm import LangevinIntegrator
from sys import stdout
import mdtraj as md
import time
from dmff import Hamiltonian, NeighborListFreud

# Load equilibrated structure
pdb = PDBFile("equilibrated_chignolin.pdb")
traj = md.load("equilibrated_chignolin.pdb")
openmm_app_path = '/home/siddarthachar/miniconda3/envs/dmff/lib/python3.9/site-packages/openmm/app/data/'

# Select CA atoms (excluding HOH)
biased_atom_indices = jnp.array([atom.index for atom in traj.topology.atoms if atom.name == 'CA' and atom.residue.name != 'HOH'])
print("Selected atom indices (CA atoms):", biased_atom_indices)

# Setup forcefield and system
forcefield = ForceField(f"{openmm_app_path}/amber14/protein.ff14SB.xml", 
                        f"{openmm_app_path}/amber14/tip3p.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)
system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

# Add external GAD force
biased_force = CustomExternalForce("fx*x + fy*y + fz*z")
biased_force.addPerParticleParameter("fx")
biased_force.addPerParticleParameter("fy")
biased_force.addPerParticleParameter("fz")
for i in range(system.getNumParticles()):
    biased_force.addParticle(i, [0.0, 0.0, 0.0])
system.addForce(biased_force)

# Create simulation
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
platform = Platform.getPlatformByName("OpenCL")
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# Setup DMFF potential
h = Hamiltonian(f"{openmm_app_path}/amber14/protein.ff14SB.xml",
                f"{openmm_app_path}/amber14/tip3p.xml")
potentials = h.createPotential(pdb.topology)

positions_all = jnp.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
# box = jnp.array([[34.018, 0.0, 0.0], [0.0, 34.018, 0.0], [0.0, 0.0, 34.018]])
box = jnp.array([jnp.array(system.getDefaultPeriodicBoxVectors()[x].value_in_unit(unit.angstrom)) for x in [0,1,2]])
print(box)
nbList = NeighborListFreud(box, 8.0, potentials.meta["cov_map"])
nbList.allocate(positions_all)
pairs = nbList.pairs

solvent_indices = jnp.array([i for i in range(positions_all.shape[0]) if i not in biased_atom_indices])
positions_adp = positions_all[biased_atom_indices]
positions_solvent = lax.stop_gradient(positions_all[solvent_indices])

def recombined_positions(pos_adp_flat):
    pos_adp = pos_adp_flat.reshape(-1, 3)
    full_pos = []
    j = 0
    for i in range(positions_all.shape[0]):
        if i in biased_atom_indices:
            full_pos.append(pos_adp[j])
            j += 1
        else:
            full_pos.append(positions_solvent[i - j])
    return jnp.stack(full_pos)

def efunc_adp(pos_adp_flat):
    return potentials.getPotentialFunc()(recombined_positions(pos_adp_flat), box, pairs, h.paramset)

def gad_force_vec(position, kappa=0.9):
    forces_u = grad(efunc_adp)(position.flatten())
    hess = hessian(efunc_adp)(position.flatten())
    w, v = jnp.linalg.eigh(hess)
    n = v[:, w.argsort()[0]]
    n /= jnp.linalg.norm(n)
    forces_b = -jnp.dot(n, forces_u) * n * kappa
    return forces_b.reshape(position.shape), n

def update_biased_forces(sim, biased_force, biased_atom_indices):
    state = sim.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    positions_adp = jnp.array(positions)[biased_atom_indices]
    biased_forces, n_vec = gad_force_vec(positions_adp)
    # print('-------biased_forces: ', biased_forces)
    for i in range(sim.system.getNumParticles()):
        if i in biased_atom_indices:
            idx = list(biased_atom_indices).index(i)
            biased_force.setParticleParameters(i, i, tuple(biased_forces[idx]))
        else:
            biased_force.setParticleParameters(i, i, (0.0, 0.0, 0.0))
    biased_force.updateParametersInContext(sim.context)
    return n_vec

# Reporters
simulation.reporters.append(DCDReporter("traj.dcd", 1000))
simulation.reporters.append(StateDataReporter("scalars.csv", 1000, time=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(StateDataReporter(stdout, 100, step=True, temperature=True, elapsedTime=True))


# Main simulation loop
n_steps = int(1e8)
gad_update_interval = 1000

n_vec_list = []

start = time.time()
for step in range(0, n_steps, gad_update_interval):    
    if step > 0:
        print(f"[step {step}] Updating GAD forces...")
        n_vec = update_biased_forces(simulation, biased_force, biased_atom_indices)
        n_vec_np = np.array(n_vec)  
        n_vec_list.append([step] + list(n_vec_np))  # save step + vector - first column is step
        np.savetxt('n_vec.dat',np.array(n_vec_list)) # first column in the step
    simulation.step(gad_update_interval)
end = time.time()

print("[production] GAD-biased simulation complete. Time elapsed (s):", end - start)
np.savetxt('time_gad_solvated.dat', np.array([end-start]))

#TODO: code to save the eigen vectors