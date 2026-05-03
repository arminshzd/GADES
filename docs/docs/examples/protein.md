# Protein Simulation Blueprint

The `examples/BluePrint/` directory contains two ready-to-use OpenMM script templates for running GADES on a full solvated protein system. The example system is the 2-SRC kinase (`2src_ref_frame.pdb`), simulated in the NPT ensemble with the AMBER14 force field.

**Files:** `examples/BluePrint/`

| File | Description |
|---|---|
| `sys_example.py` | Bias all non-water heavy atoms |
| `sys_example2.py` | Bias backbone atoms only (CA, C); factory-function pattern |
| `2src_ref_frame.pdb` | 2-SRC kinase reference structure |

---

## Common setup

Both scripts share the same system setup:

```python
# Force field
forcefield = app.ForceField(
    'amber14/protein.ff14SB.xml',
    'amber14/lipid17.xml',
    'amber14/tip3p.xml',
)

# System
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    constraints=app.HBonds,
)

# NPT ensemble
barostat = MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin)
system.addForce(barostat)

# Integrator
integrator = LangevinIntegrator(
    300 * unit.kelvin,
    1 / unit.picosecond,
    2 * unit.femtoseconds,
)
```

The GADES force must be added to the system **before** creating the `Simulation` object:

```python
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

simulation = app.Simulation(pdb.topology, system, integrator, platform)
```

---

## `sys_example.py` — All heavy atoms

Biases every non-water atom. Suitable for exploratory runs on smaller proteins.

```python
biasing_atom_ids = np.array([
    atom.index for atom in pdb.topology.atoms()
    if atom.residue.name != 'HOH'
])
```

### GADES parameters

| Parameter | Value |
|---|---|
| `kappa` | 0.9 |
| `clamp_magnitude` | 1000 |
| `interval` | 200 |
| `stability_interval` | 1000 |

### Full reporter setup

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

!!! tip
    Biasing all heavy atoms makes the Hessian very large (3N × 3N for N heavy atoms). For production runs on large proteins, consider using the backbone-only template below to reduce computational cost.

---

## `sys_example2.py` — Backbone atoms only

A more practical template for large proteins. Biases only Cα and C backbone atoms, dramatically reducing the Hessian size while still capturing the dominant conformational dynamics.

```python
biasing_atom_ids = np.array([
    atom.index for atom in pdb.topology.atoms()
    if atom.name == 'CA' or atom.name == 'C'
])
```

This script also wraps system creation in a `generate_simulation()` factory function, keeping setup logic separate from the main execution block — a cleaner pattern for scripts that may need to reinitialise the simulation (e.g., after instability):

```python
def generate_simulation():
    pdb = app.PDBFile("2src_ref_frame.pdb")
    # ... build system, add forces, create reporters ...
    return simulation, GAD_force, biasing_atom_ids

simulation, GAD_force, biasing_atom_ids = generate_simulation()
backend = OpenMMBackend(simulation)
```

---

## Adapting to your system

To use these templates with a different protein:

1. Replace `2src_ref_frame.pdb` with your prepared PDB file (protonated, solvated, equilibrated).
2. Update the `biasing_atom_ids` selection to match your atom naming.
3. Adjust `KAPPA`, `CLAMP_MAGNITUDE`, and `BIAS_UPDATE_FREQ` for your system size.

!!! note "Force field files"
    The scripts reference standard AMBER14 XML files bundled with OpenMM (`amber14/protein.ff14SB.xml`, etc.). If your system requires custom parameters, add additional XML files to the `ForceField` constructor.

---

## Requirements

```
openmm
numpy
GADES
```
