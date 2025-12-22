#export ASE_LAMMPSRUN_COMMAND="$HOME/Codes/lammps-github/build-gpu/lmp"  # Set this to your LAMMPS executable path

from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
import numpy as np

from GADES import GADESForceUpdater
from GADES.backend import ASEBackend, GADESCalculator
from GADES.utils import compute_hessian_force_fd_richardson as hessian

NSTEPS = 1e6
BIASED = 1  # Set to 1 to enable biasing, 0 to disable
KAPPA = 0.9
CLAMP_MAGNITUDE = 1000
STABILITY_CHECK_FREQ = 1000
BIAS_UPDATE_FREQ = 200
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

force_updater = GADESForceUpdater(
            backend=None,
            biased_force=None,
            bias_atom_indices=biasing_atom_ids,
            hess_func=hessian,
            clamp_magnitude=CLAMP_MAGNITUDE,
            kappa=KAPPA, 
            interval=BIAS_UPDATE_FREQ, 
            stability_interval=STABILITY_CHECK_FREQ, 
            logfile_prefix=LOG_PREFIX
            )

gades_calc = GADESCalculator(lammps_calc, force_updater)
backend = ASEBackend(gades_calc, atoms)

force_updater.backend = backend  # Set the backend for the force updater

# 3. Attach the calculator to the atoms object

atoms.calc = gades_calc

# Relax (optional)
opt = BFGS(atoms, logfile='opt.log')
opt.run(fmax=0.01)

print('Relaxed energy:', atoms.get_potential_energy())
f = atoms.get_forces()  # Force calculation
print('atoms Forces:\n', f)

f2 = backend.get_forces(atoms.get_positions())
print('backend Forces:\n', f2.reshape((-1,3)))

positions = atoms.get_positions()
positions = positions + 0.1 * (np.random.rand(*positions.shape) - 0.5)
#atoms.set_positions(positions)
#f = atoms.get_forces()  # Force calculation
#atoms.calc.calculate(atoms=atoms, properties=['forces'])
#f = atoms.calc.results['forces']

print('Perturbed forces:\n', f)




# You can now proceed to run MD or other simulations with the 'atoms' object
# --- Initialize velocities ---
MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

# --- MD setup ---
timestep_fs = 5
dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
n_steps = 100
steps_per_block = 10

time_ps, epot_list, ekin_list = [], [], []
mdind = 0


for i in range(n_steps // steps_per_block):
    dyn.run(steps_per_block)
    mdind += steps_per_block

    # save energies for plotting
    time_ps.append(mdind * timestep_fs / 1000.0)  # fs -> ps
    pe = atoms.get_potential_energy()
    ke = atoms.get_kinetic_energy()
    print(f'Step: {mdind} Epot: {pe}, Ekin: {ke}')
    epot_list.append(atoms.get_potential_energy())
    ekin_list.append(atoms.get_kinetic_energy())

print("MD run complete.")

positions, forces = backend.get_current_state()
print('Current positions:\n', positions)
print('Current forces:\n', forces)


#print(epot_list)
