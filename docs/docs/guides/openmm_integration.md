# OpenMM Integration Guide

This guide explains how to use GADES with OpenMM, covering the architecture behind the integration and the constraints you need to respect when setting up your simulation.

## Quick Start

```python
from GADES import createGADESBiasForce, GADESForceUpdater
from GADES.backend import OpenMMBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

import numpy as np
import openmm.app as app
from openmm import unit, Platform
from openmm.openmm import LangevinIntegrator

# 1. Load topology
pdb = app.PDBFile("your_system.pdb")
forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, constraints=app.HBonds)

# 2. Choose atoms to bias
biasing_atom_ids = np.array([
    atom.index for atom in pdb.topology.atoms() if atom.name == 'CA'
])

# 3. Add GADES force to the system — must happen before Simulation is created
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

# 4. Create simulation
integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

# 5. Attach GADES reporter
backend = OpenMMBackend(simulation)
simulation.reporters.append(
    GADESForceUpdater(
        backend=backend,
        biased_force=GAD_force,
        bias_atom_indices=biasing_atom_ids,
        hess_func=hessian,
        clamp_magnitude=1000,
        kappa=0.9,
        interval=500,
        stability_interval=1000,
        logfile_prefix="gades_run",
    )
)

simulation.step(1_000_000)
```

See the [OpenMM script templates](../examples/protein.md) for annotated full-script examples.

---

## How it works

### The GADES bias force

`createGADESBiasForce` creates an OpenMM `CustomExternalForce` with the energy expression:

```
E = -fx*x - fy*y - fz*z
```

This means the force on each particle is exactly `(fx, fy, fz)` — the per-particle parameters are the force components directly. All particles are initialised with `(0, 0, 0)`, so the force has no effect until the first bias update.

```python
force = CustomExternalForce("-fx*x-fy*y-fz*z")
force.addPerParticleParameter("fx")
force.addPerParticleParameter("fy")
force.addPerParticleParameter("fz")
for i in range(n_particles):
    force.addParticle(i, [0.0, 0.0, 0.0])
```

### Force groups

The GADES force is assigned to **force group 1**. All standard force field terms (bonded, non-bonded, PME) default to group 0. This separation is critical: when computing the Hessian, `OpenMMBackend.get_forces()` requests forces from group 0 only, so the bias does not contaminate the curvature estimate:

```python
state = context.getState(getForces=True, groups={0})   # unbiased forces only
```

The full simulation (dynamics) includes both groups, as normal.

!!! warning "Don't reassign your force field forces to group 1"
    If any of your physical forces are in group 1, they will be excluded from the Hessian calculation and GADES will compute the wrong curvature. All standard OpenMM force fields default to group 0, so this is only a concern if you manually set `setForceGroup(1)` on another force.

### The Reporter pattern

`GADESForceUpdater` is an OpenMM `Reporter`. OpenMM calls two methods on every reporter at each relevant step:

- `describeNextReport(simulation)` — returns how many steps until the next event and what data is needed
- `report(simulation, state)` — performs the actual work

GADES uses `describeNextReport` to schedule the next bias update, stability check, or post-bias stability check, whichever comes first. At each scheduled step, `report` either recomputes the GAD force and pushes it to the context, or removes the bias if instability is detected.

Force updates are pushed with `updateParametersInContext`, which modifies the live context without rebuilding the simulation:

```python
for i, idx in enumerate(bias_atom_indices):
    bias_force_object.setParticleParameters(idx, idx, tuple(biased_force_values[i]))
bias_force_object.updateParametersInContext(simulation.context)
```

---

## Setup constraints

### The GADES force must be added before `app.Simulation`

OpenMM finalises the force list when the `Simulation` (and its underlying `Context`) is created. Forces added after this point are not registered in the context and will have no effect.

```python
# Correct
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)                        # before Simulation
simulation = app.Simulation(topology, system, integrator)

# Wrong — GAD_force will be silently ignored
simulation = app.Simulation(topology, system, integrator)
system.addForce(GAD_force)                        # too late
```

### `OpenMMBackend` takes the Simulation, not the System

The backend wraps the live `Simulation` object (which contains the `Context`, `Integrator`, and `System`). It must be created after the simulation is initialised:

```python
backend = OpenMMBackend(simulation)   # after Simulation(...) and setPositions(...)
```

---

## Comparison with the ASE backend

| Aspect | OpenMM backend | ASE backend |
|---|---|---|
| Force application | `CustomExternalForce` updated via `updateParametersInContext` | `GADESCalculator` wrapper added to bias calculator results |
| Bias scheduling | OpenMM Reporter interface (`describeNextReport` / `report`) | Called inside `GADESCalculator.calculate()` |
| Circular dependency | None — force object and backend are independent | Yes — solved by `ASEBackend.with_gades()` factory |
| Setup complexity | Moderate — force must be added before `Simulation` | Handled by factory method |
| Recommended pattern | `GADESForceUpdater` as reporter | `ASEBackend.with_gades()` |

---

## Key parameters

| Parameter | Description | Typical values |
|---|---|---|
| `kappa` | Bias scaling factor. Controls how strongly forces are projected onto the softest mode. | 0.5 – 0.9 |
| `clamp_magnitude` | Maximum per-atom bias force (kJ/mol/nm). Prevents unphysically large updates. | 500 – 2000 |
| `interval` | Steps between Hessian updates. Smaller = more accurate bias direction, higher cost. | 200 – 2000 |
| `stability_interval` | Steps between temperature stability checks. Bias is removed if temperature deviates too far. | 500 – 2000 |
| `logfile_prefix` | If set, writes eigenvector, eigenvalue, and biased-atom trajectory logs. | any string |

## Parameter recommendations

### `kappa`

Set `kappa=0.9` and control the strength of the bias through `clamp_magnitude` instead:

```python
GADESForceUpdater(
    ...
    kappa=0.9,                  # recommended value
    clamp_magnitude=1000,       # adjust this to control exploration aggressiveness
)
```

`kappa` damps **all** forces uniformly, while `clamp_magnitude` only acts on atoms where the bias would otherwise exceed the cap. Using a high κ with a moderate clamp gives effective exploration without over-biasing low-gradient regions.

### `stability_interval`

Set `stability_interval` to `interval // 2` or smaller to catch instabilities before they propagate:

```python
GADESForceUpdater(
    ...
    interval=500,
    stability_interval=200,     # roughly interval // 2
)
```

### `hess_func`

Use `compute_hessian_force_fd_richardson` (Richardson-extrapolated finite differences). It is less sensitive to step size than a plain single-step finite difference and reduces numerical error in the Hessian:

```python
from GADES.utils import compute_hessian_force_fd_richardson as hessian

GADESForceUpdater(..., hess_func=hessian)
```

---

## Advanced options

### Eigensolver

By default, `GADESForceUpdater` uses `numpy.linalg.eigh` for the full Hessian eigendecomposition. For large biased subsets, this becomes expensive. Switch to Lanczos:

```python
GADESForceUpdater(
    ...
    eigensolver="lanczos",        # or "lanczos_hvp" for very large systems
    lanczos_iterations=30,
)
```

See the [large systems guide](large_systems.md) for details on when to use each solver.

### Bofill Hessian update

To reduce the number of full Hessian evaluations, enable the Bofill quasi-Newton update, which approximates the Hessian between full recomputations:

```python
GADESForceUpdater(
    ...
    use_bofill_update=True,
    full_hessian_interval=2000,   # recompute full Hessian every 2000 steps
)
```

### Stability checking

`OpenMMBackend` estimates the instantaneous temperature from the kinetic energy and compares it to the integrator's target temperature. If the deviation exceeds the threshold (default 50 K), the bias is removed until the next update cycle.

For integrators where temperature cannot be auto-detected (e.g. `VerletIntegrator`), set it explicitly:

```python
backend = OpenMMBackend(simulation, target_temperature=300.0)
```

### Running without bias

Set `BIASED = 0` and skip attaching the reporter. The `GAD_force` is still added to the system (required to keep the context valid for potential later use), but with all-zero parameters it contributes nothing to the dynamics.

```python
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)
simulation = app.Simulation(...)

# No GADESForceUpdater — plain MD with zero-cost bias force
simulation.step(n_steps)
```
