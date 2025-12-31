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
- **O(k·N) time per iteration**: Where k is the number of Lanczos iterations

!!! warning "Memory vs Speed Trade-off"
    The primary advantage of Lanczos is **memory savings**, not speed. The number of iterations k required for accurate eigenvector convergence depends on the eigenvalue spectrum:

    - **Well-separated smallest eigenvalue**: k can be small (50-100), providing both memory and speed benefits
    - **Dense/clustered eigenvalues**: k may need to approach N (the system dimension), eliminating speed benefits

    For a system with N degrees of freedom requiring k ≈ N iterations:

    - Matrix Lanczos: O(k·N²) ≈ O(N³) — **same as full eigendecomposition**
    - HVP Lanczos: Still O(N) memory, but requires ~2N force evaluations

    **The memory advantage always remains; the speed advantage depends on your system's eigenvalue structure.**

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
| `'numpy'` (default) | Full `eigh()` | O(N²) | O(N³) | Small/medium systems where memory permits |
| `'lanczos'` | Matrix-based Lanczos | O(N²) | O(k·N²) | When Hessian is available and eigenvalues are well-separated |
| `'lanczos_hvp'` | Matrix-free Lanczos | O(N) | O(k · force_cost) | Large systems where Hessian doesn't fit in memory |

### When to Use Each

**Use `'numpy'` (default) when:**

- The Hessian fits comfortably in memory (rule of thumb: < 1000 atoms for ~72 MB)
- You need guaranteed accurate eigenvector direction
- You want the simplest, most reliable option

**Use `'lanczos'` when:**

- You already have the Hessian computed
- The smallest eigenvalue is well-separated from others (check your system!)
- You accept the risk of inaccurate eigenvector if iterations are insufficient

**Use `'lanczos_hvp'` when:**

- The Hessian is too large to fit in memory
- Memory is the primary constraint, not speed
- You are willing to use many iterations to ensure eigenvector accuracy

!!! note "When Does Lanczos Provide Speed Benefits?"
    Lanczos converges quickly when the smallest eigenvalue is **well-separated** from the rest of the spectrum. This is often the case for:

    - Systems near transition states (the reaction coordinate has a distinct negative eigenvalue)
    - Stiff systems with a clear softest mode

    Lanczos converges slowly when eigenvalues are **clustered** or nearly degenerate. In the worst case, you may need k ≈ N iterations, at which point:

    - Matrix Lanczos offers no speed benefit over `numpy`
    - HVP Lanczos still saves memory but is slow (2k force evaluations)

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

- 20-30 iterations: Minimum for reasonable eigenvector accuracy
- 50-100 iterations: Recommended for production runs
- 100+ iterations: High accuracy, use when eigenvalues are close together

!!! warning
    Do not use fewer than 20 iterations. While eigenvalues may appear converged, the eigenvector direction may be significantly wrong, leading to incorrect bias forces.

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

# Test with more iterations for realistic eigenvector accuracy
python benchmarks/hvp_scaling.py --sizes 100 500 --lanczos-iters 100
```

The benchmark reports time, memory, and eigenvalue accuracy. However, note that:

!!! warning "Benchmark Limitations"
    The benchmark uses **random test matrices** which may have different eigenvalue distributions than real molecular Hessians. The eigenvector alignment reported in the benchmark output tells you whether the Lanczos eigenvector matches the true eigenvector.

    **If alignment is low (< 0.95), the Lanczos result would be unsuitable for GADES**, regardless of how fast it runs. Always prioritize accuracy over speed.

## Accuracy Considerations

!!! warning "Eigenvector Accuracy is Critical"
    In GADES, the eigenvector direction is **essential** for correct bias force computation:

    ```
    F_GAD = F - 2 * (F · n) * n
    ```

    If the eigenvector `n` is inaccurate, the bias force will point in the wrong direction, preventing the system from correctly ascending toward the saddle point.

### Eigenvalue vs Eigenvector Convergence

The Lanczos algorithm converges **eigenvalues faster than eigenvectors**. This means:

- With few iterations, you may get a reasonable eigenvalue but a poor eigenvector
- GADES requires accurate eigenvector direction, not just eigenvalue magnitude
- **You must use sufficient iterations to ensure eigenvector convergence**

### Iteration Requirements

The number of iterations required depends critically on your system's eigenvalue spectrum:

**Well-separated smallest eigenvalue** (e.g., near transition state):

| Iterations | Eigenvector Quality | Memory Savings |
|------------|---------------------|----------------|
| 50-100     | Often sufficient    | Yes (O(N))     |
| 100-200    | Good                | Yes (O(N))     |

**Clustered/dense eigenvalues** (worst case):

| Iterations | Eigenvector Quality | Memory Savings |
|------------|---------------------|----------------|
| k ≈ N/2    | Moderate            | Yes (O(N))     |
| k ≈ N      | Good                | Yes (O(N))     |

!!! danger "Speed vs Accuracy Trade-off"
    For a system with N degrees of freedom:

    - If you need k ≈ N iterations for eigenvector convergence, **Lanczos provides no speed benefit** over full eigendecomposition
    - Matrix Lanczos: O(k·N²) ≈ O(N³) when k ≈ N
    - The **only remaining benefit is memory savings** with HVP Lanczos (O(N) vs O(N²))

    **Do not assume Lanczos will be faster. Always validate eigenvector accuracy first.**

### Validating Eigenvector Accuracy

Before production runs, validate your Lanczos settings by comparing against the full Hessian eigenvector on a representative configuration:

```python
import numpy as np

# Compute eigenvector with full Hessian (reference)
H = hess_func(atoms.get_positions(), atoms.get_forces(), 0)
eigvals_full, eigvecs_full = np.linalg.eigh(H)
v_reference = eigvecs_full[:, 0]

# Compute eigenvector with Lanczos
from GADES.lanczos import lanczos_smallest
eigval_lanczos, v_lanczos = lanczos_smallest(H, n_iter=30)

# Check alignment (should be > 0.99 for reliable results)
alignment = abs(np.dot(v_reference, v_lanczos))
print(f"Eigenvector alignment: {alignment:.4f}")

if alignment < 0.95:
    print("WARNING: Increase lanczos_iterations!")
```

!!! danger "Always Verify Alignment"
    If the alignment is below ~0.95, the bias direction may be significantly wrong. Increase `lanczos_iterations` until you achieve acceptable alignment for your system.

## Lanczos and Bofill Updates

### Understanding Bofill Approximation

When `use_bofill_update=True`, GADES approximates the Hessian between full computations using the Bofill quasi-Newton update:

1. **Full Hessian** is computed at `full_hessian_interval` steps
2. **Between full computations**, the Hessian is updated incrementally based on position and gradient changes
3. This saves expensive Hessian calculations while maintaining reasonable accuracy

The Bofill approximation is anchored to periodic full Hessian computations, which limits error accumulation.

### Frequent Bias Updates with Bofill

!!! tip "Update bias every 1-10 steps with Bofill"
    When using Bofill, consider setting `interval` to a small value (1-10 steps) rather than the typical 100-1000 steps used with full Hessian calculations.

    **Why this works better:**

    - **Bofill updates are cheap**: Only a rank-2 matrix update, no expensive Hessian computation
    - **Better curvature tracking**: The bias direction follows the changing curvature of the PES as the system evolves
    - **More accurate than infrequent full Hessian**: Even though each Bofill Hessian is approximate, frequently updating the bias can track the softest mode more accurately than computing an exact Hessian every 1000 steps

    ```python
    backend = ASEBackend.with_gades(
        ...,
        use_bofill_update=True,
        interval=5,                    # Update bias every 5 steps (cheap with Bofill)
        full_hessian_interval=1000,    # Reset with full Hessian every 1000 steps
    )
    ```

    This approach gives you the best of both worlds: the bias continuously adapts to the local curvature while periodic full Hessian calculations prevent error accumulation.

### Combining Lanczos with Bofill

!!! warning "Two Sources of Approximation"
    Using `eigensolver='lanczos'` with `use_bofill_update=True` introduces two approximations:

    1. **Bofill**: Approximates the Hessian matrix based on gradient changes
    2. **Lanczos**: Approximates the eigenvector of that Hessian

    These errors are **additive, not multiplicative**. The combination is not inherently dangerous, but requires care:

    - Bofill preserves the general eigenvalue structure (rank-2 updates don't dramatically change spectra)
    - If eigenvalue separation is maintained, Lanczos should converge similarly
    - Both approximations need to be accurate enough for your application

**Recommendations:**

| Eigensolver | Bofill Update | Notes |
|-------------|---------------|-------|
| `'numpy'` | Yes | ✅ Simplest - exact eigenvector from approximate Hessian |
| `'numpy'` | No | ✅ Most accurate - no approximations in eigensolver |
| `'lanczos'` | No | ⚠️ Verify alignment is sufficient |
| `'lanczos'` | Yes | ⚠️ Two approximations - verify both are accurate |
| `'lanczos_hvp'` | N/A | ✅ Bofill not applicable (no stored Hessian) |

!!! tip "When to use Lanczos + Bofill"
    This combination can be useful when:

    - Memory allows storing the Hessian (rules out `lanczos_hvp`)
    - Full Hessian computation is expensive
    - The eigenvalue gap is well-preserved by Bofill updates

    Always validate that eigenvector alignment remains acceptable throughout the simulation.

When using `eigensolver='lanczos_hvp'`, the Bofill update parameters are ignored because no explicit Hessian is computed. The HVP approach recomputes the softest mode from scratch at each bias update interval using only force evaluations.

```python
backend = ASEBackend.with_gades(
    ...,
    eigensolver='lanczos_hvp',
    lanczos_iterations=50,         # Use sufficient iterations!
    use_bofill_update=True,        # Ignored with lanczos_hvp
    full_hessian_interval=10000,   # Ignored with lanczos_hvp
)
```

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
