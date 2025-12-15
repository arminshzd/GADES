import matplotlib.pyplot as plt
import numpy as np

# choose one of the following implementations of EMT:
# included in ase
from ase.calculators.emt import EMT
# faster performance
#from asap3 import EMT

from ase import units
from ase.cluster.cubic import FaceCenteredCubic as ClusterFCC
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic as LatticeFCC
from ase.md.langevin import Langevin  # for later NPT simulations
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,

)
from ase.md.verlet import VelocityVerlet

from GADES.utils import compute_hessian_force_fd_richardson as hessian
from GADES import getGADESBiasForce, GADESForceUpdater

# Set up initial positions of Cu atoms on Fcc crystal lattice
size = 10
atoms = LatticeFCC(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol='Cu',
    size=(size, size, size),
    pbc=True,
)



# Describe the interatomic interactions with the Effective Medium Theory (EMT)
atoms.calc = EMT()

# Set the initial velocities corresponding to T=300K from Maxwell Boltzmann
# Distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We use Velocity Verlet algorithm to integrate the Newton's equations.
timestep_fs = 5
dyn = VelocityVerlet(atoms, timestep_fs * units.fs)  # 5 fs time step.


def printenergy(a):
    """
    Function to print the thermodynamical properties i.e potential energy,
    kinetic energy and total energy
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    print(
        f'Energy per atom: Epot ={epot:6.3f}eV  Ekin = {ekin:.3f}eV '
        f'(T={temp:.3f}K) Etot = {epot + ekin:.3f}eV'
    )


# Now run the dynamics
print('running a NVE simulation of fcc Cu')
printenergy(atoms)
# init lists to for energy vs time data
time_ps, epot, ekin = [], [], []
mdind = 0
steps_per_block = 10
for i in range(20):
    dyn.run(steps_per_block)
    mdind += steps_per_block
    printenergy(atoms)
    # save the energies of the current MD step
    time_ps.append(mdind * timestep_fs / 1000.0)
    epot.append(atoms.get_potential_energy())
    ekin.append(atoms.get_kinetic_energy())

etot = np.array(epot) + np.array(ekin)

# Plot energies vs time
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(time_ps, epot, label='Potential energy')
ax.plot(time_ps, ekin, label='Kinetic energy')
ax.plot(time_ps, etot, label='Total energy')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Energy (eV)')
ax.legend(loc='best')
ax.grid(True, linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()