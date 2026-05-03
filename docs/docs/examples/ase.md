# ASE + LAMMPS: Argon Crystal

This example demonstrates the GADES ASE backend using an argon FCC crystal as the test system, with LAMMPS as the force engine.

**Files:** `examples/ase/mdrun.py`

---

## Overview

The script builds a small 2×2×2 argon FCC supercell, attaches a LAMMPS Lennard-Jones calculator, and runs GADES-enhanced NVE molecular dynamics. It uses the `ASEBackend.with_gades()` factory method, which handles all wiring between the GADES bias, the ASE calculator wrapper, and the integrator.

---

## Prerequisites

LAMMPS must be installed and its path exported:

```bash
export ASE_LAMMPSRUN_COMMAND="$HOME/path/to/lmp"
```

---

## Setup

### System

```python
from ase.build import bulk
atoms = bulk('Ar', 'fcc', a=5.26).repeat((2, 2, 2))   # 32 Ar atoms
```

### LAMMPS calculator

```python
parameters = {
    'units':       'metal',
    'atom_style':  'atomic',
    'pair_style':  'lj/cut 10.0',
    'pair_modify': 'shift yes',
    'mass':        ['1 39.948'],
    'pair_coeff':  ['1 1 0.01034 3.405 10.0'],  # ε=0.01034 eV, σ=3.405 Å
}
lammps_calc = LAMMPS(**parameters)
```

### GADES backend

The `ASEBackend.with_gades()` factory creates a `GADESCalculator`, wraps it around the base LAMMPS calculator, and wires up the bias:

```python
from GADES.backend import ASEBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=lammps_calc,
    bias_atom_indices=biasing_atom_ids,   # all Ar atoms
    hess_func=hessian,
    clamp_magnitude=1000,
    kappa=0.9,
    interval=100,                         # Hessian update frequency (steps)
    stability_interval=1000,
    logfile_prefix="log_prefix",
)
```

### Key parameters

| Parameter | Value | Description |
|---|---|---|
| `kappa` | 0.9 | Bias scaling factor |
| `clamp_magnitude` | 1000 | Max per-atom bias force (eV/Å) |
| `interval` | 100 | Steps between Hessian updates |
| `stability_interval` | 1000 | Steps between temperature stability checks |

---

## Running MD

Energy is minimised with BFGS before dynamics:

```python
from ase.optimize import BFGS
opt = BFGS(atoms, logfile='opt.log')
opt.run(fmax=0.01)
```

Velocities are initialised from a Maxwell-Boltzmann distribution at 300 K and a `VelocityVerlet` integrator is used:

```python
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
dyn = VelocityVerlet(atoms, 5 * units.fs)

# Register integrator for step tracking
backend.integrator = dyn
```

!!! important
    `backend.integrator` must be set after creating the integrator so that GADES can track the current MD step and apply the bias at the correct interval.

The MD loop runs in blocks:

```python
for i in range(n_steps // steps_per_block):
    dyn.run(steps_per_block)
```

---

## Adapting to other systems

To use a different ASE calculator (EMT, FHI-aims, VASP, etc.), replace `lammps_calc` with any ASE `Calculator` object. The GADES wiring is calculator-agnostic.

```python
from ase.calculators.emt import EMT
backend = ASEBackend.with_gades(atoms=atoms, base_calc=EMT(), ...)
```

---

## Requirements

```
ase
lammps          # external binary, see LAMMPS documentation
numpy
GADES
```
