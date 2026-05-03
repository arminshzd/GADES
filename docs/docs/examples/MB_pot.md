# Müller-Brown 2D Potential

The Müller-Brown potential is a standard 2D test system with three energy minima (A, B, C) connected by two saddle points (D, E). This example demonstrates the complete GADES workflow — from running biased MD to computing the free energy surface — on an analytically defined system where ground truth is available.

**Files:** `examples/MullerBrown/`

| File | Purpose |
|---|---|
| `1_run_MullerBrown.ipynb` | Visualise the potential and run GADES vs unbiased MD |
| `2_find_min.ipynb` | Locate energy minima from the GADES trajectory |
| `3_path_metad.py` | Well-tempered metadynamics with a PLUMED-style path CV |
| `4_metad_reweight.ipynb` | Reweight the metadynamics trajectory to get the FES |
| `mean_shift.py` | Mean-shift clustering utility used in step 2 |

---

## Step 1 — Visualise and run MD

`1_run_MullerBrown.ipynb` introduces the Müller-Brown potential and compares plain Langevin MD against GADES-enhanced dynamics.

### Potential definition

The Müller-Brown potential is imported from `GADES.potentials`:

```python
from GADES.potentials import (
    muller_brown_potential,
    muller_brown_force,
    muller_brown_hess,
)
```

The notebook plots the potential landscape, the gradient field, and the GAD field (the softest-mode eigenvector field, aligned with the gradient), illustrating why GADES preferentially drives the system toward saddle points.

### Integrator

A custom BAOAB Langevin integrator (pure NumPy) is used to integrate the equations of motion:

```python
# Shared parameters for both runs
mass, gamma, dt, kBT = 1.0, 1.0, 0.01, 2.0
n_steps = 1_000_000
```

### Unbiased run

```python
x = np.array([-0.5582, 1.4417])   # starts near minimum A
# forces_b = np.zeros_like  (no bias)
```

Trajectory saved to `muller_brown_unbiased_1e6.npy`.

### GADES-biased run

The GAD bias force is computed analytically (Hessian available in closed form):

```python
def muller_brown_gad_force(position, kappa=0.9):
    w, v = np.linalg.eigh(muller_brown_hess(position))
    n = v[:, 0] / np.linalg.norm(v[:, 0])
    return -np.dot(muller_brown_force(position), n) * n * kappa
```

```python
x = np.array([-1.0, 1.0])   # starts away from any minimum
```

Trajectory saved to `muller_brown_GADES_1e6_clamped.npy`.

---

## Step 2 — Find energy minima

`2_find_min.ipynb` locates the three energy minima from the GADES trajectory using kernel density estimation and mean-shift clustering.

```python
from mean_shift import mean_shift, deduplicate
from scipy.stats import gaussian_kde

kde = gaussian_kde(data.T)
modes = mean_shift(start_points, kde, min_density=0.5)
maxima = deduplicate(modes)
```

The clustering finds three density peaks that correspond to the three energy minima of the Müller-Brown potential:

| Label | x | y |
|---|---|---|
| A | −0.568 | 1.432 |
| B | 0.539 | 0.036 |
| C | 0.182 | 0.347 |

These coordinates are used to define the path in step 3.

---

## Step 3 — Path metadynamics

`3_path_metad.py` runs well-tempered metadynamics using a PLUMED-style path collective variable defined through the minima found in step 2. Requires a CUDA-capable GPU.

### Path CV

The path CV uses the PLUMED algebraic formulation:

$$s(x) = \frac{\sum_i i \cdot e^{-\lambda |x - x_i|^2}}{\sum_i e^{-\lambda |x - x_i|^2}}, \quad
z(x) = -\frac{1}{\lambda} \ln \sum_i e^{-\lambda |x - x_i|^2}$$

where $x_i$ are the ordered reference frames along the A→C→B path.

### Key parameters

```python
beta             = 0.5     # 1/(k_B T) — controls exploration temperature
height           = 5.0     # initial Gaussian height
biasfactor       = 20.0    # well-tempered bias factor γ
deposition_stride = 500    # add a Gaussian hill every N steps
sigma            = 0.5     # Gaussian width in s-space
lambda_path      = 100.0   # path-CV sharpness parameter
z_wall_max       = 0.05    # harmonic wall on z to keep system on-path
nsteps           = 5_000_000
```

### Path definition

```python
minA = np.array([-0.568, 1.432])
minC = np.array([ 0.182, 0.347])
minB = np.array([ 0.539, 0.036])

path_A_to_C = np.linspace(minA, minC, 20, endpoint=False)
path_C_to_B = np.linspace(minC, minB, 21)
path_points = np.vstack([path_A_to_C, path_C_to_B])  # 41 reference frames
```

### Output files

All outputs are written to `path_metad_data/`:

| File | Contents |
|---|---|
| `metad_traj.dat` | `t, s, z, V_bias, kBT, n_hills` per frame |
| `metad_traj_xy.dat` | `t, x, y, V_bias` per frame |
| `hills.dat` | Gaussian hill centers and heights |
| `bias_grid.npy` | Accumulated bias potential on the s-grid |
| `checkpoint_step_*/` | Restart checkpoints |

To restart an interrupted run, pass `restart_from='path_metad_data/checkpoint_step_N'`.

---

## Step 4 — Reweight and compute the FES

`4_metad_reweight.ipynb` uses the accumulated metadynamics bias to reweight the trajectory and recover the unbiased free energy surface.

### Reweighting

Each frame is assigned a weight proportional to the exponential of the accumulated bias:

```python
beta    = 1.0 / cv_pd["kBT"].iloc[0]
weights = np.exp(beta * bias)
```

### Free energy surface

The 2D FES is estimated with a weighted KDE:

```python
from scipy.stats import gaussian_kde as kde
kde_f = kde(np.vstack([x, y]), weights=weights, bw_method='silverman')
F_xy  = -1/beta * np.log(pdf)
F_xy -= np.nanmin(F_xy)
```

### Saddle point detection

Saddle points are identified from the FES by finding critical points with $\nabla F \approx 0$ and $\det(\nabla^2 F) < 0$:

| Label | x | y | F (kBT) |
|---|---|---|---|
| D | 0.165 | 0.325 | 77 |
| E | −0.762 | 0.651 | 108 |

---

## Requirements

```
numpy
matplotlib
scipy
torch          # step 3 only — CUDA recommended
tqdm
skimage        # step 4 only
```
