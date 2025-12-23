# Note: Need to set LAMMPS executable path in environment variable ASE_LAMMPSRUN_COMMAND
#   export ASE_LAMMPSRUN_COMMAND="$HOME/Codes/lammps-github/build-gpu/lmp"

from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
import numpy as np

from GADES import GADESBias
from GADES.backend import ASEBackend, GADESCalculator
from GADES.utils import compute_hessian_force_fd_richardson as hessian

NSTEPS = 1e6
BIASED = 1  # Set to 1 to enable biasing, 0 to disable
KAPPA = 0.9
CLAMP_MAGNITUDE = 1000
STABILITY_CHECK_FREQ = 1000
BIAS_UPDATE_FREQ = 100
LOG_PREFIX = "log_prefix"
PLATFORM = "CPU"

# Build a small fcc Argon crystal
# Lattice parameter ~5.26 Å for Ar (example value)
atoms = bulk('Ar', 'fcc', a=5.26).repeat((2, 2, 2))

# Lennard-Jones parameters for Argon in 'metal' units:
# epsilon ≈ 0.01034 eV, sigma ≈ 3.405 Å; cutoff 10 Å
parameters = {
    'units': 'metal',
    'atom_style': 'atomic',
    'pair_style': 'lj/cut 10.0',
    'pair_modify': 'shift yes',
    # Use lists of strings for multi-value keywords:
    'mass': ['1 39.948'],
    'pair_coeff': ['1 1 0.01034 3.405 10.0'],
    # Optional neighbor settings:
    'neighbor': '2.0 bin',
    'neigh_modify': ['every 1 delay 0 check yes'],
}

# 2. Create the LAMMPS calculator
lammps_calc = LAMMPS(**parameters)

biasing_atom_ids = np.array([atom.index for atom in atoms if (atom.symbol == 'Ar')])

force_bias = GADESBias(backend=None,
                       biased_force=None, # maybe can be acquired internally
                       bias_atom_indices=biasing_atom_ids,
                       hess_func=hessian,
                       clamp_magnitude=CLAMP_MAGNITUDE,
                       kappa=KAPPA, 
                       interval=BIAS_UPDATE_FREQ, 
                       stability_interval=STABILITY_CHECK_FREQ, 
                       logfile_prefix=LOG_PREFIX
                       )

# ASE calculator that adds GAD forces to LAMMPS forces based on the LAMMPS calculator
# We keep force bias from the calculator so that we can support other force bias later (beyond GADESBias)
gades_calc = GADESCalculator(lammps_calc, force_bias)

# 3. Create the ASE backend for GADES
# In ASE, the Calculator does not know about Atoms
# Atoms gives the calculator the atom positions to calculate forces and energy.
# The backend keeps track of both the Atoms object and the Calculator object.
# Inside the ASEBackend constructor, we attach atoms.calc to gades_calc
backend = ASEBackend(gades_calc, atoms)

# Set the backend for the force bias, because GADESBias needs to query the backend
#  for atom forces, atom symbols and so on.
force_bias.backend = backend

# Relax (optional)
opt = BFGS(atoms, logfile='opt.log')
opt.run(fmax=0.01)
print('Relaxed energy:', atoms.get_potential_energy())

#print(f"{list(atoms)}")
#print(f"{backend.get_atom_symbols(biasing_atom_ids)}")


# Testing force calculation
#f = atoms.get_forces()  # Force calculation
#print('atoms Forces:\n', f)
#f2 = backend.get_forces(atoms.get_positions())
#print('backend Forces:\n', f2.reshape((-1,3)))

# Testing force calculation with perturbed positions
#positions = atoms.get_positions()
#positions = positions + 0.1 * (np.random.rand(*positions.shape) - 0.5)
#atoms.set_positions(positions)
# either with atom.get_forces()
#f = atoms.get_forces()  # Force calculation
# or with atoms.calc.calculate()
#atoms.calc.calculate(atoms=atoms, properties=['forces'])
#f = atoms.calc.results['forces']
#print('Perturbed forces:\n', f)

# Run MD or other simulations with the 'atoms' object

# --- Initialize velocities ---
MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

# --- MD setup ---
timestep_fs = 5
dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
n_steps = 200
steps_per_block = 10

time_ps, epot_list, ekin_list = [], [], []
mdind = 0

# Provide the integrator to the backend for step tracking
# Remember that in ASE, Atoms and Calculator are not aware if MD  timesteps, or Integrator does.
backend.integrator = dyn

for i in range(n_steps // steps_per_block):
    dyn.run(steps_per_block)
    mdind += steps_per_block

    # save energies for plotting
    time_ps.append(mdind * timestep_fs / 1000.0)  # fs -> ps
    pe = atoms.get_potential_energy()
    ke = atoms.get_kinetic_energy()
    T = atoms.get_temperature()
    print(f'Step: {mdind} {pe}, {ke}, {T}')
    epot_list.append(pe)
    ekin_list.append(ke)


print("MD run complete.")

positions, forces = backend.get_current_state()
print('Current positions:\n', positions)
print('Current forces:\n', forces)


#print(epot_list)
