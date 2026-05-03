# OpenMM Script Templates

The `examples/BluePrint/` directory contains two template scripts showing the correct structure for setting up a GADES run with OpenMM. They use a PDB file as a placeholder input — the scripts are not intended to run as-is, but to illustrate the required boilerplate and the order in which things must be wired together.

**Files:** `examples/BluePrint/`

| File | Description |
|---|---|
| `sys_example.py` | Template: bias all non-water heavy atoms |
| `sys_example2.py` | Template: bias backbone atoms only (CA, C); factory-function pattern |
| `2src_ref_frame.pdb` | Placeholder PDB file (example input only) |

---

## Required structure for any OpenMM run

Both templates demonstrate the same essential pattern. The ordering of these steps matters.

### 1. Load topology and choose biased atoms

```python
pdb = app.PDBFile("your_system.pdb")

biasing_atom_ids = np.array([
    atom.index for atom in pdb.topology.atoms()
    if atom.name == 'CA' or atom.name == 'C'   # adjust selection as needed
])
```

### 2. Build the system and integrator

```python
forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    constraints=app.HBonds,
)

integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
```

### 3. Add the GADES force before creating the Simulation

The `CustomExternalForce` object must be added to the system **before** `app.Simulation` is constructed. OpenMM freezes the force list at simulation creation time.

```python
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)                         # must come before Simulation(...)

simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
```

### 4. Create the backend and attach the reporter

```python
backend = OpenMMBackend(simulation)

if BIASED:
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
            logfile_prefix=LOG_PREFIX,
        )
    )
```

---

## Template variants

### `sys_example.py` — All non-water atoms

Shows the minimal flat-script pattern. Biases all heavy atoms (non-water):

```python
biasing_atom_ids = np.array([
    atom.index for atom in pdb.topology.atoms()
    if atom.residue.name != 'HOH'
])
```

!!! tip
    For large proteins this makes the Hessian very expensive (3N × 3N). Use the backbone-only selection below in practice.

### `sys_example2.py` — Backbone atoms, factory-function pattern

Biases only Cα and C backbone atoms and wraps system creation in a `generate_simulation()` function. This is the cleaner pattern for production scripts, particularly if the simulation may need to be reinitialised:

```python
def generate_simulation():
    pdb = app.PDBFile("your_system.pdb")
    # ... build system, add GAD_force, create reporters ...
    return simulation, GAD_force, biasing_atom_ids

simulation, GAD_force, biasing_atom_ids = generate_simulation()
backend = OpenMMBackend(simulation)
```

---

## Adapting to your system

1. Replace the PDB file with your own prepared structure (protonated, solvated, equilibrated).
2. Update the `biasing_atom_ids` selection to match your needs.
3. Tune `KAPPA`, `CLAMP_MAGNITUDE`, and `BIAS_UPDATE_FREQ` for your system size — larger systems need larger intervals to keep the Hessian cost manageable.

!!! note "Force field files"
    The templates reference AMBER14 XML files bundled with OpenMM. If your system requires custom parameters, add the relevant XML files to the `ForceField` constructor.

---

## Requirements

```
openmm
numpy
GADES
```
