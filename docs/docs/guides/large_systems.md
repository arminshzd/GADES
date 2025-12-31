# Scaling to Large Systems

This guide explains how to use GADES efficiently with large molecular systems (1000+ atoms) using the matrix-free Lanczos eigensolver with Hessian-vector products (HVP).

## The Scaling Challenge

GADES identifies the softest vibrational mode of a molecular system by computing the smallest eigenvalue of the Hessian matrix. For a system with N atoms:

| System Size | DOF (3N) | Hessian Size | Memory (float64) |
|-------------|----------|--------------|------------------|
| 100 atoms   | 300      | 300 × 300    | 0.7 MB           |
| 1,000 atoms | 3,000    | 3,000 × 3,000| 72 MB            |
| 10,000 atoms| 30,000   | 30,000 × 30,000 | 7.2 GB        |
| 100,000 atoms| 300,000 | 300,000 × 300,000 | 720 GB      |

The standard approach of forming the full Hessian and computing eigenvalues via `np.linalg.eigh()` becomes prohibitively expensive for large systems due to:

- **O(N²) memory**: Storing the full Hessian
- **O(N³) time**: Full eigendecomposition

## Matrix-Free Solution

GADES provides a matrix-free eigensolver that uses **Hessian-vector products (HVP)** computed via finite differences:

$$
Hv \approx \frac{\nabla E(x + \epsilon v) - \nabla E(x - \epsilon v)}{2\epsilon} = \frac{-F(x + \epsilon v) + F(x - \epsilon v)}{2\epsilon}
$$

Combined with the Lanczos algorithm, this finds the softest mode with:

- **O(N) memory**: Only stores vectors, not the full matrix
- **O(k·N) time**: Where k is the number of Lanczos iterations (typically 10-30)

## Quick Start

Use `eigensolver='lanczos_hvp'` to enable the matrix-free approach:

```python
from GADES.backend import ASEBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

# For large systems, use matrix-free Lanczos
backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=calculator,
    bias_atom_indices=list(range(len(atoms))),  # All atoms
    hess_func=hessian,  # Still required but not used with lanczos_hvp
    clamp_magnitude=1000,
    kappa=0.9,
    interval=1000,
    eigensolver='lanczos_hvp',      # Matrix-free eigensolver
    lanczos_iterations=20,           # Number of Lanczos iterations
    hvp_epsilon=1e-5,                # Finite difference step size
)
```

## Eigensolver Options

GADES provides three eigensolver options:

| Eigensolver | Method | Memory | Time | Best For |
|-------------|--------|--------|------|----------|
| `'numpy'` (default) | Full `eigh()` | O(N²) | O(N³) | Small systems (< 500 atoms) |
| `'lanczos'` | Matrix-based Lanczos | O(N²) | O(k·N²) | Medium systems, approximate |
| `'lanczos_hvp'` | Matrix-free Lanczos | O(N) | O(k·N) | Large systems (1000+ atoms) |

### When to Use Each

**Use `'numpy'` (default) when:**

- System has fewer than ~500 atoms
- You need exact eigenvalues for analysis
- Memory is not a concern

**Use `'lanczos'` when:**

- System has 500-2000 atoms
- You already have the Hessian computed
- You want faster than full eigendecomposition

**Use `'lanczos_hvp'` when:**

- System has 1000+ atoms
- Memory is limited
- You only need the softest mode (GADES default use case)

## Parameters

### `lanczos_iterations`

Number of Lanczos iterations. More iterations improve accuracy but increase computation time.

```python
# Default: 20 (from config)
backend = ASEBackend.with_gades(
    ...,
    eigensolver='lanczos_hvp',
    lanczos_iterations=30,  # More iterations for better accuracy
)
```

**Recommendations:**

- 10-20 iterations: Fast, sufficient for most GADES applications
- 20-30 iterations: Good balance of speed and accuracy
- 30-50 iterations: High accuracy, use when eigenvalues are close together

### `hvp_epsilon`

Finite difference step size for computing HVP.

```python
# Default: 1e-5 (from config)
backend = ASEBackend.with_gades(
    ...,
    eigensolver='lanczos_hvp',
    hvp_epsilon=1e-6,  # Smaller for higher accuracy
)
```

**Recommendations:**

- `1e-5` (default): Good balance for most systems
- `1e-6`: Higher accuracy, may be affected by numerical noise
- `1e-4`: Faster but less accurate, use for rough exploration

## Example: Large Protein System

```python
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS
from GADES.backend import ASEBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

# Load a large protein structure
atoms = read('protein.pdb')
print(f"System size: {len(atoms)} atoms")

# Set up LAMMPS calculator
calc = LAMMPS(parameters={'pair_style': 'lj/cut 10.0', ...})
atoms.calc = calc

# Configure GADES with matrix-free eigensolver
backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=calc,
    bias_atom_indices=list(range(len(atoms))),
    hess_func=hessian,
    clamp_magnitude=500,
    kappa=0.9,
    interval=2000,
    eigensolver='lanczos_hvp',
    lanczos_iterations=25,
    hvp_epsilon=1e-5,
    target_temperature=300,
)

# Run dynamics
from ase.md.langevin import Langevin
from ase import units

dyn = Langevin(atoms, 2 * units.fs, temperature_K=300, friction=0.01)
backend.integrator = dyn

dyn.run(100000)
```

## Benchmarking

GADES includes a benchmark script to compare eigensolver performance:

```bash
# Run benchmark with various system sizes
python benchmarks/hvp_scaling.py --sizes 50 100 200 500 1000

# Quick test
python benchmarks/hvp_scaling.py --sizes 20 50 100 --trials 2
```

Sample output:

```
### Time Comparison (ms)
----------------------------------------------------------------------
   Atoms      DOF    Full Hessian         Lanczos     Lanczos HVP
----------------------------------------------------------------------
      50      150            2.42            1.40            2.11
     100      300           13.00            1.66            2.95
     500     1500          720.00           15.20           12.50
    1000     3000         5800.00           58.00           24.00
```

## Accuracy Considerations

The matrix-free approach trades some accuracy for scalability:

1. **Lanczos approximation**: The Lanczos algorithm provides approximate eigenvalues. More iterations improve accuracy.

2. **Finite difference error**: HVP computation introduces O(ε²) error. Richardson extrapolation (used internally) reduces this.

3. **Eigenvector convergence**: The softest mode eigenvector converges faster than interior eigenvalues, which is ideal for GADES.

For GADES applications, these approximations are typically acceptable because:

- We only need the *direction* of the softest mode
- The bias force magnitude is controlled by `kappa` and `clamp_magnitude`
- Slight inaccuracies in eigenvector direction have minimal impact on dynamics

## Combining with Bofill Updates

For maximum efficiency with large systems, combine matrix-free Lanczos with Bofill Hessian updates:

```python
backend = ASEBackend.with_gades(
    ...,
    eigensolver='lanczos_hvp',
    lanczos_iterations=20,
    use_bofill_update=True,        # Not used with lanczos_hvp
    full_hessian_interval=10000,   # Not used with lanczos_hvp
)
```

!!! note
    When using `eigensolver='lanczos_hvp'`, the Bofill update parameters are ignored because no explicit Hessian is computed. The HVP approach recomputes the softest mode from scratch at each bias update interval using only force evaluations.

## Troubleshooting

### Poor eigenvector convergence

**Symptom**: Bias direction changes erratically between updates.

**Solutions**:

1. Increase `lanczos_iterations` (try 30-50)
2. Check if system has nearly degenerate eigenvalues
3. Reduce `interval` to update more frequently

### Numerical instability

**Symptom**: NaN or very large bias forces.

**Solutions**:

1. Reduce `hvp_epsilon` (try 1e-6)
2. Ensure forces are computed accurately by base calculator
3. Check for atoms with very close positions

### Slow performance

**Symptom**: Matrix-free approach is slower than expected.

**Solutions**:

1. Ensure base calculator is efficient
2. Reduce `lanczos_iterations` if accuracy permits
3. Increase `interval` between bias updates
