# AIB9 Peptide

This example applies GADES to the Aib₉ (poly-α-aminoisobutyric acid nonapeptide) peptide in explicit solvent using OpenMM. It then trains a slow relaxation variable (SRV) neural network on the enhanced trajectory and exports the result as a collective variable for downstream use.

**Files:** `examples/aib9/`

| File | Purpose |
|---|---|
| `1_run_GADES.py` | Run GADES-biased (or plain) MD |
| `2_train_srv.ipynb` | Train an SNRV model on the trajectory |
| `3_export_srv.ipynb` | Export the trained SRV as a TorchScript CV |
| `aib9.gro` / `topol.top` | GROMACS topology and coordinates |

---

## Step 1 — Run the simulation

`1_run_GADES.py` runs a Langevin MD simulation using a GROMACS-format force field. Set `BIASED = 1` to enable the GADES bias; set it to `0` for an unbiased reference run.

### Key parameters

```python
BIASED            = 1      # 0 = plain MD, 1 = GADES-enhanced
KAPPA             = 0.5    # bias scaling factor (κ)
CLAMP_MAGNITUDE   = 500    # max per-atom bias force (kJ/mol/nm)
BIAS_UPDATE_FREQ  = 500    # steps between Hessian updates
STABILITY_CHECK_FREQ = 200 # steps between temperature stability checks
NSTEPS            = 10000  # total simulation steps
PLATFORM          = "CPU"  # OpenMM platform
```

### Biased atoms

The bias is applied to all Cα atoms:

```python
biasing_atom_ids = np.array([
    atom.index for atom in top.topology.atoms() if atom.name == 'CA'
])
```

### GADES setup

```python
GAD_force = createGADESBiasForce(system.getNumParticles())
system.addForce(GAD_force)

backend = OpenMMBackend(simulation)
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

### Output

The script writes a DCD trajectory with filename encoding the run parameters:

```
traj_{KAPPA}k_{BIAS_UPDATE_FREQ}freq_{CLAMP_MAGNITUDE}clamp_{NSTEPS:.1g}steps.dcd
```

---

## Step 2 — Train the SRV model

`2_train_srv.ipynb` trains a Slow Relaxation Variable (SNRV) network on the GADES trajectory to discover kinetically relevant collective variables.

### Feature encoding

Backbone φ/ψ dihedral angles are computed for all 9 residues using MDTraj and encoded as sin/cos pairs, giving a 36-dimensional input vector per frame:

```python
traj_x = torch.concat([
    torch.Tensor(np.sin(angs).reshape(angs.shape[0], -1)),
    torch.Tensor(np.cos(angs).reshape(angs.shape[0], -1)),
], dim=1)   # shape: (n_frames, 36)
```

### Model configuration

```python
model = Snrv(
    input_size=36,
    output_size=3,       # number of slow CVs to learn
    hidden_depth=3,
    hidden_size=100,
    batch_norm=True,
    dropout_rate=0.0,
    lr=5e-5,
    n_epochs=100,
    batch_size=20_000,
    VAMPdegree=2,
    is_reversible=True,
)
```

Training uses the VAMP-2 score as the objective, which maximises the kinetic content of the learned CVs. The model is saved to `srv_models/snrv_{lag}.pt`.

---

## Step 3 — Export the CV

`3_export_srv.ipynb` wraps the trained SNRV in a `CV` class that accepts raw dihedral arrays and outputs a single collective variable (TIC 1), then compiles it with TorchScript for use in PLUMED or other tools.

```python
class CV(torch.nn.Module):
    def forward(self, x):          # x: tensor of shape (18,) — 9 φ + 9 ψ in radians
        feats = self.get_colvars(x)
        evecs = self.model.transform(feats.view(1, -1))
        return evecs[:, 1:2]       # TIC 1

torch.jit.script(CV(model)).save('srv_50.ptc')
```

The exported `.ptc` file can be loaded by PLUMED's `pytorch` collective variable interface to drive adaptive sampling in a production run.

---

## Requirements

```
openmm
mdtraj
snrv
torch
numpy
```
