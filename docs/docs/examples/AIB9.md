# AIB9 Peptide

This example applies GADES to the Aib₉ (poly-α-aminoisobutyric acid nonapeptide) peptide in explicit solvent. The workflow uses GADES-enhanced OpenMM sampling to discover a collective variable that captures the right-hand to left-hand helical transition, then drives that transition to completion with PLUMED well-tempered metadynamics inside GROMACS.

**Files:** `examples/aib9/`

| File | Purpose |
|---|---|
| `1_run_GADES.py` | Run GADES-biased (or plain) MD with OpenMM |
| `2_train_srv.ipynb` | Train an SNRV model on the GADES trajectory |
| `3_export_srv.ipynb` | Export the trained SRV as a TorchScript CV |
| `4_GROMACS/` | Equilibration + metadynamics + reweighting with GROMACS/PLUMED |
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

## Step 4 — Metadynamics with GROMACS + PLUMED

`4_GROMACS/` runs the full GROMACS equilibration pipeline followed by well-tempered metadynamics biased along the SRV collective variable exported in step 3. The metadynamics accelerates the right-hand to left-hand helical transition, which is too slow to observe in unbiased MD. The free energy surface is then recovered by reweighting.

### Prerequisites

- GROMACS with PLUMED patch applied
- PLUMED built with the `PYTORCH_MODEL` module (`plumed config has PYTORCH` should return yes)
- The exported `srv_50.ptc` file from step 3 copied into `4_GROMACS/md/` as `srv_in.ptc`

### 4a — Energy minimisation

```bash
cd em/
gmx grompp -f em.mdp -c ../../aib9.gro -p ../../topol.top -o em.tpr -maxwarn 1
gmx mdrun -deffnm em
```

Steepest descent, up to 10 000 steps, converges when max force < 500 kJ/mol/nm.

### 4b — NVT equilibration

```bash
cd nvt/
gmx grompp -f nvt.mdp -c ../em/em.gro -r ../em/em.gro -p ../../topol.top -o nvt.tpr
gmx mdrun -deffnm nvt
```

500 ps at 400 K with V-rescale thermostat. Velocities are generated from a Maxwell-Boltzmann distribution.

### 4c — NPT equilibration

```bash
cd npt/
gmx grompp -f npt.mdp -c ../nvt/nvt.gro -r ../nvt/nvt.gro -t ../nvt/nvt.cpt -p ../../topol.top -o npt.tpr
gmx mdrun -deffnm npt
```

500 ps at 400 K and 1 bar with Parrinello-Rahman barostat.

### 4d — Production metadynamics

```bash
cd md/
gmx grompp -f md.mdp -c ../npt/npt.gro -r ../npt/npt.gro -p ../../topol.top -o md.tpr
gmx mdrun -deffnm md -plumed plumed.dat
```

100 ns NPT run (50 M steps × 2 fs) with PLUMED driving the bias. The `plumed.dat` file loads the SRV model and defines the metadynamics:

```
# Load all 9 φ and 9 ψ backbone dihedrals
phi1: TORSION ATOMS=5,7,9,18
...
psi9: TORSION ATOMS=111,113,122,124

# Evaluate the SRV — output is model.node-0
model: PYTORCH_MODEL FILE=srv_in.ptc \
       ARG=phi1,...,phi9,psi1,...,psi9

# Well-tempered metadynamics on the SRV output
metad: METAD ARG=model.node-0 SIGMA=0.05 PACE=250 HEIGHT=1.2 \
             TEMP=400 BIASFACTOR=10 \
             GRID_MIN=-6 GRID_MAX=6 GRID_BIN=1000 ...

PRINT ARG=phi1,...,psi9,model.*,metad.bias STRIDE=100 FILE=COLVAR
```

| Metadynamics parameter | Value |
|---|---|
| CV | SRV first eigenfunction (`model.node-0`) |
| Gaussian width (σ) | 0.05 |
| Deposition pace | 250 steps (0.5 ps) |
| Initial height | 1.2 kJ/mol |
| Bias factor (γ) | 10 |
| Grid | [−6, 6], 1000 bins |

### 4e — Reweighting

The `reweight/` directory recovers the unbiased free energy surface from the metadynamics trajectory. The `reweight/plumed.dat` is run as a PLUMED post-processing driver to compute the time-dependent offset $c(t)$ and produce `COLVAR_RBIAS` with the reweighted bias (`metad.rbias = bias − c(t)`):

```bash
cd reweight/
gmx mdrun -rerun ../md/md.xtc -plumed plumed.dat -deffnm reweight
# or using plumed driver:
plumed driver --mf_xtc ../md/md.xtc --plumed plumed.dat
```

`reweight.ipynb` then reads `COLVAR_RBIAS` and computes the FES:

```python
# Reweighting factors
kBT = 400 * 8.314 / 1000   # kJ/mol at 400 K
data["weights"] = np.exp(data["metad.bias"] / kBT)

# 1D FES along χ₁ with bootstrap error estimate
density, se, _ = weighted_kde_with_se(data.chi1, data.weights, grid)
F = -kBT * np.log(density)
```

The notebook produces:

- **1D FES** along χ₁ (helical state order parameter) with a 95% confidence band
- **2D FES** in the (χ₁, χ₂) plane

where χ₁ and χ₂ are linear combinations of the per-residue sin(φ)/sin(ψ) values that distinguish the right-handed and left-handed helical basins:

```python
lambdas = np.sum(np.sin(phi_psi), axis=1) * (-0.8)
chi1 = lambdas[:, 2:7].sum(axis=1)          # global helical character
chi2 = lambdas[:, 2] + lambdas[:, 3] - lambdas[:, 5] - lambdas[:, 6]
```

---

## Requirements

```
# Steps 1–3 (OpenMM / Python)
openmm
mdtraj
snrv
torch
numpy

# Step 4 (GROMACS / PLUMED)
gromacs          # with PLUMED patch
plumed           # built with PYTORCH_MODEL module
```
