# GADES Improvement Plan

This document outlines recommended improvements for the GADES codebase, organized by priority.

---

## Priority 1 (High - Fix Soon)

### 1. ~~Implement `ASEBackend.is_stable()` with Actual Stability Criterion~~ ✅ COMPLETED

**File:** `GADES/backend.py`

**Implementation (completed):**

- Added optional `target_temperature` parameter to `ASEBackend.__init__()`
- Added `_get_target_temperature()` helper that:
  - Returns explicit `target_temperature` if set
  - Auto-detects from integrator (`Langevin.temp`, `NVTBerendsen.temperature`)
  - Returns `None` if neither available
- Implemented `is_stable()` with 50 K threshold (matching OpenMMBackend)
- Issues warning once if target temperature unavailable, then skips check

**Usage:**

```python
# Option 1: Explicit target temperature
backend = ASEBackend(gades_calc, atoms, target_temperature=300.0)

# Option 2: Auto-detect from NVT integrator
backend = ASEBackend(gades_calc, atoms)
dyn = Langevin(atoms, timestep, temperature_K=300, friction=0.01)
backend.integrator = dyn  # Will read temp from integrator
```

---

### 2. ~~Add Input Validation for Critical Parameters~~ ✅ COMPLETED

**Files:** `GADES/gades.py`

**Implementation (completed):**

| Parameter | Location | Validation | Action |
|-----------|----------|------------|--------|
| `n_particles` | `createGADESBiasForce()` | Non-negative integer | `raise ValueError` |
| `hess_func` | `GADESBias.__init__()` | Must be callable | `raise TypeError` |
| `bias_atom_indices` | `GADESBias.__init__()` | Non-empty sequence of non-negative integers | `raise TypeError/ValueError` |
| `clamp_magnitude` | `GADESBias.__init__()` | Positive number | `raise ValueError` |
| `interval` | `GADESBias.__init__()` | Positive integer | `raise ValueError` |
| `stability_interval` | `GADESBias.__init__()` | Positive integer (if provided) | `raise ValueError` |
| `kappa` | `GADESBias.__init__()` | Should be in (0, 1] | `warnings.warn` |
| `delta` | `set_hess_step_size()` | Positive number | `raise TypeError/ValueError` |

**Docstrings updated** with comprehensive `Raises:` and `Warns:` sections.

---

### 3. ~~Add Basic Unit Tests~~ ✅ COMPLETED

**Directory:** `tests/`

**Implementation (completed):**

```
tests/
├── __init__.py
├── conftest.py            # MockBackend, fixtures, sample hess functions
├── test_utils.py          # 14 tests: clamp_force_magnitudes, Muller-Brown
├── test_validation.py     # 40 tests: all input validation
├── test_gades.py          # 20 tests: GADESBias core functionality
└── test_backend.py        # 22 tests: ASEBackend (skipped without ASE)
```

**Test coverage:**

| Module | Tests | Status |
|--------|-------|--------|
| `clamp_force_magnitudes` | 10 | ✅ Pass |
| `Muller-Brown potential` | 4 | ✅ Pass |
| `createGADESBiasForce` validation | 7 | ✅ Pass |
| `GADESBias.__init__` validation | 27 | ✅ Pass |
| `set_hess_step_size` validation | 6 | ✅ Pass |
| `GADESBias` core methods | 20 | ✅ Pass |
| `ASEBackend` | 22 | ⏭️ Skip (ASE not installed) |

**Run tests:** `conda activate GADES && python -m pytest tests/ -v`

---

### 4. ~~Fix HVP Logging Crash and Log Cleanup Error Handling~~ ✅ COMPLETED

**Files:** `GADES/gades.py`

**Problems (fixed):**
- `get_gad_force()` calls `_logging` with `w=None` when `eigensolver='lanczos_hvp'` and logging is enabled, causing `TypeError` on `w[w_sorted]`.
- `_close_logs()` raises `warnings.warn(...)`, which raises instead of warning and can mask cleanup failures.

**Implementation (completed):**

| Fix | Location | Change |
|-----|----------|--------|
| Guard eigenvalue logging | `_logging()` line 592 | Added `and w is not None` check |
| Fix invalid raise | `_close_logs()` line 698 | Removed `raise` from `warnings.warn()` |
| Add user warning | `__init__()` line 284-289 | Warn when `lanczos_hvp` + logging enabled |

**Regression tests added** (`tests/test_gades.py`):
- `test_lanczos_hvp_with_logging_no_crash` - Verifies no TypeError
- `test_lanczos_hvp_logs_eigenvector_but_skips_eigenvalues` - Confirms correct file contents
- `test_lanczos_hvp_logging_warning_issued` - Validates warning message
- `test_close_logs_handles_exception_gracefully` - Verifies warning instead of raise
- `test_close_logs_skips_already_closed_files` - Confirms idempotent behavior

**Test count:** 193 tests passing (up from 188)

---

## Priority 2 (Medium - Improve Quality)

### 4. Replace Print Statements with Logging Module

**Files:** `GADES/gades.py`

**Current State:** Uses ANSI color codes in print statements:

```python
print(f"\033[1;33m[GADES| WARNING]...\033[0m")
```

**Recommendation:** Use Python's `logging` module:

```python
import logging

logger = logging.getLogger("GADES")

# In __init__.py or module setup:
handler = logging.StreamHandler()
formatter = logging.Formatter('[GADES | %(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage:
logger.warning("Bias update interval must be larger than 100 steps...")
logger.info(f"step {step}] Updating bias forces...")
```

**Benefits:**

- Users can control log levels
- Logs can be redirected to files
- Consistent with Python best practices

---

### 5. Extract Magic Numbers to Configurable Constants

**File:** `GADES/gades.py`

**Current magic numbers:**

| Value | Location | Meaning |
|-------|----------|---------|
| `50` | backend.py:74, 322 | Temperature stability threshold (K) |
| `100` | gades.py:717, 729 | Post-bias check delay (steps) |
| `110` | gades.py:160 | Minimum interval override (steps) |
| `1` | gades.py:507 | Force group for GADES bias |

**Recommendation:** Add class-level constants or constructor parameters:

```python
class GADESBias:
    DEFAULT_POST_BIAS_CHECK_DELAY = 100
    DEFAULT_MIN_INTERVAL = 110
    DEFAULT_FORCE_GROUP = 1

    def __init__(self, ..., post_bias_check_delay=None, force_group=None):
        self.post_bias_check_delay = post_bias_check_delay or self.DEFAULT_POST_BIAS_CHECK_DELAY
        # ...
```

---

### 6. ~~Integrate Lanczos Algorithm as Optional Eigenvalue Solver~~ ✅ COMPLETED

**Files:** `GADES/lanczos.py`, `GADES/gades.py`

**Implementation (completed):**

- Rewrote `lanczos.py` in pure NumPy (removed JAX dependency)
- Added `eigensolver` parameter to `GADESBias.__init__()`: `'numpy'` (default) or `'lanczos'`
- Added `lanczos_iterations` parameter (default: 20, from `defaults`)
- Added `_compute_softest_mode()` helper method for eigenvalue computation
- Added comprehensive tests in `tests/test_lanczos.py` (22 tests)

---

### 7. ~~Integrate Bofill Hessian Update for Efficiency~~ ✅ COMPLETED

**Files:** `GADES/bofill.py`, `GADES/gades.py`

**Implementation (completed):**

- Rewrote `bofill.py` in pure NumPy (removed JAX dependency)
- Added `use_bofill_update` parameter to `GADESBias.__init__()` (default: `False`)
- Added `full_hessian_interval` parameter (default: `interval * defaults["bofill_full_hessian_multiplier"]`)
- Added `_get_hessian()` helper method that:
  - Computes full Hessian on first call and at `full_hessian_interval` steps
  - Uses Bofill approximation between full Hessian calculations
  - Stores previous positions, forces, and Hessian for updates
- Added comprehensive tests in `tests/test_bofill.py` (18 tests)
- Added integration tests in `tests/test_gades.py` (`TestBofillIntegration`: 8 tests)

---

## Priority 3 (Low - Nice to Have)

### 8. ~~Refactor ASE Initialization Pattern~~ ✅ COMPLETED

**Problem:** Circular dependency between GADESBias, GADESCalculator, and ASEBackend.

**Old 4-step workaround:**

```python
force_bias = GADESBias(backend=None, ...)  # Incomplete initialization
gades_calc = GADESCalculator(lammps_calc, force_bias)
backend = ASEBackend(gades_calc, atoms)
force_bias.backend = backend  # Manual patching required!
```

**Solution:** Added `ASEBackend.with_gades()` factory method:

```python
backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=lammps_calc,
    bias_atom_indices=biasing_atom_ids,
    hess_func=hessian,
    clamp_magnitude=1000,
    kappa=0.9,
    interval=100,
    stability_interval=1000,
)
# Access GADESBias via backend.gades_bias if needed
```

**Implementation details:**
- Factory method handles all internal wiring
- Uses lazy import to avoid circular import at module load time
- Uses `TYPE_CHECKING` for type hints without runtime issues
- Added `gades_bias` attribute to ASEBackend for user access
- Added comprehensive documentation in `docs/docs/guides/ase_integration.md`

---

### 9. ~~Clean Up Utils Module~~ ✅ COMPLETED

**Changes made:**

- Created `GADES/potentials.py` with pure NumPy Muller-Brown implementation
- Moved BAOAB integrator to `examples/MullerBrown/integrators.py`
- `utils.py` now contains only: `clamp_force_magnitudes`, Hessian computation functions
- Removed JAX dependencies

---

### 10. ~~Add Type Hints to All Public APIs~~ ✅ COMPLETED

**Files updated:**

- `GADES/backend.py`: All Backend classes, GADESCalculator
- `GADES/lanczos.py`: All functions
- `GADES/bofill.py`: All functions
- `GADES/gades.py`: GADESBias, GADESForceUpdater
- `GADES/utils.py`: Hessian functions, clamp_force_magnitudes

---

### 11. ~~Fix README OpenMM Usage and Parallel Hessian Caveat~~ ✅ COMPLETED

**Files:** `README.md`, `GADES/utils.py`

**Problems (fixed):**
- README OpenMM example omits required `backend` argument when constructing `GADESForceUpdater`; copy/paste raises `TypeError`.
- `compute_hessian_force_fd_block_parallel` uses `joblib` with `loky`, which cannot pickle OpenMM/ASE backends; this path fails silently for users.

**Implementation (completed):**
- Updated README OpenMM example to show `OpenMMBackend(simulation)` creation and `backend=` argument
- Changed `compute_hessian_force_fd_block_parallel` from `backend='loky'` to `backend='threading'`
- Added documentation explaining why `loky` cannot be used (non-picklable backends)
- Recommended serial version or Richardson method for most use cases

---

### 12. ~~Add Reporter/Logging Coverage Without Heavy Backends~~ ✅ COMPLETED

**Files:** `tests/test_gades.py`

**Implementation (completed):**

Added 13 new tests covering `GADESForceUpdater` reporter interface using `MockBackend`:

| Test Class | Tests | What it validates |
|------------|-------|-------------------|
| `TestGADESForceUpdaterReporting` | 4 | `describeNextReport` return format, flag transitions |
| `TestGADESForceUpdaterReport` | 5 | `report()` bias application, removal, scheduling |
| `TestPostBiasScheduling` | 4 | Post-bias check scheduling and priority |

**Notes:**
- HVP + logging tests were already added in Priority 1.4
- All tests use `MockBackend` from `conftest.py`, no OpenMM/ASE required
- Tests verify flag transitions, bias application, and post-bias stability check scheduling

---

## Implementation Checklist

- [x] Priority 1.1: Implement `ASEBackend.is_stable()`
- [x] Priority 1.2: Add input validation
- [x] Priority 1.3: Create test suite
- [x] Priority 1.4: Fix HVP logging crash and log cleanup error handling
- [x] Priority 2.4: Replace print with logging
- [x] Priority 2.5: Extract magic numbers to configurable defaults
- [x] Priority 2.6: Integrate Lanczos solver
- [x] Priority 2.7: Integrate Bofill update
- [x] Priority 3.8: Refactor ASE initialization
- [x] Priority 3.9: Clean up utils module
- [x] Priority 3.10: Add type hints
- [x] Priority 3.11: Fix README OpenMM usage and parallel Hessian caveat
- [x] Priority 3.12: Add reporter/logging coverage without heavy backends

---

## Changelog

### 2026-01-01

- ✅ Implemented persistent bias for ASE backend (F1 + F2)
  - Added `_stored_bias` and `_bias_active` state to `GADESCalculator`
  - Rewrote `GADESCalculator.calculate()` with 3-step flow:
    1. Check stability - clear bias if unstable
    2. At bias update intervals, recompute and store bias
    3. Apply stored bias at every step (if active)
  - ASE now matches OpenMM behavior: bias persists between updates
  - Added 5 tests in `TestGADESCalculatorPersistentBias`
  - Fixed integration tests for `_get_hessian()` signature
- ✅ Added bounds validation to `ASEBackend.with_gades()` (F3)
  - Validates `bias_atom_indices` against `atoms` size before creating GADESBias
  - Added 2 tests for bounds validation
- Total test count: 264 tests passing

### 2025-12-31

- ✅ Added reporter/logging coverage without heavy backends (Priority 3.12)
  - Added 13 tests for `GADESForceUpdater` using `MockBackend`
  - Tests cover `describeNextReport`, `report()`, and post-bias scheduling
  - Updated `tests/README.md` with new test documentation
- ✅ Fixed HVP logging crash and log cleanup error handling (Priority 1.4)
  - Added `w is not None` guard in `_logging()` to prevent TypeError when `eigensolver='lanczos_hvp'`
  - Removed invalid `raise` from `warnings.warn()` in `_close_logs()`
  - Added warning when `lanczos_hvp` + logging enabled (eigenvalue logging unavailable)
  - Added 6 regression tests in `tests/test_gades.py`
  - Updated `tests/README.md` with new test documentation
- ✅ Fixed README OpenMM usage and parallel Hessian caveat (Priority 3.11)
  - Added missing `backend` argument to README OpenMM example
  - Changed `compute_hessian_force_fd_block_parallel` from `loky` to `threading` backend
  - Documented why multiprocessing cannot be used (non-picklable backends)
- Total test count: 206 tests passing

### 2025-12-30 (Session 2)

- ✅ Refactored ASE initialization with `ASEBackend.with_gades()` factory method
  - Eliminates circular dependency workaround (4-step pattern → 1-step)
  - Uses lazy import to avoid circular import at module load time
  - Uses `TYPE_CHECKING` for type hints without runtime issues
  - Added `gades_bias` attribute to ASEBackend
  - Added 5 unit tests in `tests/test_backend.py`
- ✅ Added comprehensive documentation
  - Created `docs/docs/guides/ase_integration.md` explaining circular dependency
  - Created `docs/docs/reference/backend.md` for Backend API reference
  - Updated `docs/mkdocs.yml` with new navigation
- ✅ Updated `.gitignore` with common Python/macOS entries
- ✅ Fixed type annotation warnings in `gades.py` and `backend.py`
- Total test count: 157 tests passing

### 2025-12-30 (Session 2)

- ✅ Implemented Matrix-Free Lanczos with HVP (Phases 1-3)
  - Created `GADES/hvp.py` with 3 HVP functions (central, Richardson, forward)
  - Added `lanczos_hvp()` and `lanczos_hvp_smallest()` to `lanczos.py`
  - Added `eigensolver='lanczos_hvp'` option to `GADESBias`
  - Added `hvp_epsilon` parameter (default: 1e-5)
  - Added `_compute_softest_mode_hvp()` method for matrix-free eigensolver
  - Updated `get_gad_force()` to bypass Hessian computation when using HVP
- ✅ Updated all affected files:
  - `config.py`: Added `hvp_epsilon` default
  - `gades.py`: Added `hvp_epsilon` parameter to `GADESBias` and `GADESForceUpdater`
  - `backend.py`: Added `hvp_epsilon` parameter to `ASEBackend.with_gades()`
- ✅ Created comprehensive test suite:
  - 17 tests in `tests/test_hvp.py`
  - 8 tests in `tests/test_lanczos.py` for matrix-free Lanczos
  - 6 tests in `tests/test_gades.py` for `TestLanczosHVPIntegration`
- ✅ Created benchmark script `benchmarks/hvp_scaling.py`:
  - Compares full Hessian, matrix-based Lanczos, and HVP-based Lanczos
  - Measures time, memory, and eigenvalue accuracy
  - Configurable system sizes and Lanczos iterations
- ✅ Added documentation:
  - Created `docs/docs/guides/large_systems.md` (comprehensive scaling guide)
  - Created `examples/MullerBrown/eigensolver_comparison.py` (example script)
  - Updated `docs/docs/guides/index.md` and `mkdocs.yml`
- Total test count: 188 tests passing
- **Matrix-Free Lanczos with HVP implementation complete (all 5 phases)**

### 2025-12-30 (Session 1)

- ✅ Integrated Lanczos eigenvalue solver as optional eigensolver
  - Rewrote `lanczos.py` in pure NumPy (removed JAX dependency)
  - Added `eigensolver` parameter: `'numpy'` (default) or `'lanczos'`
  - Added `lanczos_iterations` parameter (default: 20)
  - Added `_compute_softest_mode()` helper method
  - Added 22 unit tests in `tests/test_lanczos.py`
- ✅ Integrated Bofill Hessian update for computational efficiency
  - Rewrote `bofill.py` in pure NumPy (removed JAX dependency)
  - Added `use_bofill_update` parameter (default: `False`)
  - Added `full_hessian_interval` parameter for controlling update frequency
  - Added `_get_hessian()` helper method with state tracking
  - Added 18 unit tests in `tests/test_bofill.py`
  - Added 8 integration tests in `tests/test_gades.py`
- ✅ Updated `config.py` with new defaults:
  - `lanczos_iterations`: 20
  - `bofill_full_hessian_multiplier`: 10
- ✅ Updated `GADESForceUpdater` with same parameters as `GADESBias`
- Total test count: 152 tests passing

### 2024-12-30

- ✅ Implemented `ASEBackend.is_stable()` with temperature-based stability check
  - Added `target_temperature` parameter to constructor
  - Auto-detection from NVT/NPT integrators
  - Warning issued when target temperature unavailable
- ✅ Added comprehensive input validation
  - `createGADESBiasForce()`: validates `n_particles`
  - `GADESBias.__init__()`: validates all critical parameters
  - `set_hess_step_size()`: added type checking
  - Updated docstrings with exception documentation
- ✅ Created test suite with 96 tests (74 pass, 22 skip without ASE)
  - `test_utils.py`: clamp_force_magnitudes, Muller-Brown potential
  - `test_validation.py`: all input validation
  - `test_gades.py`: GADESBias core functionality
  - `test_backend.py`: ASEBackend (requires ASE)
- ✅ Replaced print statements with Python logging module
  - Created `ColorFormatter` class in `__init__.py` for ANSI color output
  - Configured GADES logger with colored output format
  - Replaced 5 print statements in `gades.py` with logger calls
  - Updated tests to use `caplog` fixture for log verification
- ✅ Extracted magic numbers to configurable `defaults` dictionary
  - Created `GADES/config.py` with `defaults` dict containing:
    - `stability_threshold_temp_diff`: 50 (K)
    - `post_bias_check_delay`: 100 (steps)
    - `min_bias_update_interval`: 110 (steps)
    - `gades_force_group`: 1
  - Updated `gades.py` and `backend.py` to use defaults
  - Users can modify defaults at runtime: `defaults["key"] = value`

---

## Future Work: Matrix-Free Lanczos with Finite-Difference HVP

### Motivation

For large molecular systems, the full Hessian matrix becomes a memory and computational bottleneck:

| System Size | DOF (3N) | Hessian Size | Memory (float64) |
|-------------|----------|--------------|------------------|
| 100 atoms   | 300      | 300 × 300    | 0.7 MB           |
| 1,000 atoms | 3,000    | 3,000 × 3,000| 72 MB            |
| 10,000 atoms| 30,000   | 30,000 × 30,000 | 7.2 GB        |
| 100,000 atoms| 300,000 | 300,000 × 300,000 | 720 GB      |

Current GADES approach:
1. Compute full Hessian via finite differences: O(N) force evaluations, O(N²) memory
2. Eigendecomposition via `np.linalg.eigh()`: O(N³) time
3. Extract softest mode

**Key insight:** GADES only needs the *softest mode* (smallest eigenvalue + eigenvector), not the full spectrum.

---

### Proposed Solution: Matrix-Free Lanczos

Instead of forming the full Hessian, use **Hessian-vector products (HVP)** with Lanczos iteration:

```
Hv ≈ (∇E(x + εv) - ∇E(x - εv)) / (2ε)
```

where:
- `∇E(x)` = negative force at position x (already computed for dynamics)
- `v` = arbitrary vector
- `ε` = small displacement (e.g., 1e-5 Å)

**Complexity comparison:**

| Approach | Force Evals | Memory | Time Complexity |
|----------|-------------|--------|-----------------|
| Full Hessian + eigh | O(N) | O(N²) | O(N³) |
| Matrix-free Lanczos | O(k) | O(N) | O(k·N) |

Where k = number of Lanczos iterations (typically 10-30).

For 10,000 atoms: **720 GB → 2.4 MB memory**, **O(N³) → O(N) time**

---

### Implementation Plan

#### Phase 1: Core HVP Infrastructure

**File:** `GADES/hvp.py` (new)

```python
def finite_difference_hvp(force_func, positions, vector, epsilon=1e-5):
    """
    Compute Hessian-vector product using central finite differences.

    Args:
        force_func: Callable that returns forces given positions
        positions: Current atomic positions (N, 3)
        vector: Direction vector for HVP (N, 3) or (3N,)
        epsilon: Finite difference step size

    Returns:
        hvp: Hessian-vector product (3N,)
    """
    v = vector.reshape(-1)
    v_normalized = v / np.linalg.norm(v)

    pos_flat = positions.reshape(-1)

    # Central difference: Hv ≈ (-F(x+εv) + F(x-εv)) / (2ε)
    # Note: Hessian of energy = negative gradient of force
    pos_plus = (pos_flat + epsilon * v_normalized).reshape(-1, 3)
    pos_minus = (pos_flat - epsilon * v_normalized).reshape(-1, 3)

    force_plus = force_func(pos_plus).reshape(-1)
    force_minus = force_func(pos_minus).reshape(-1)

    # H = -dF/dx, so Hv = -(F+ - F-) / (2ε)
    hvp = -(force_plus - force_minus) / (2 * epsilon)

    return hvp * np.linalg.norm(v)  # Scale back
```

#### Phase 2: Matrix-Free Lanczos

**File:** `GADES/lanczos.py` (extend)

```python
def lanczos_hvp(hvp_func, n_dof, n_iter=20, seed=None):
    """
    Lanczos algorithm using only Hessian-vector products.

    Args:
        hvp_func: Callable that computes H @ v given v
        n_dof: Number of degrees of freedom (3N)
        n_iter: Number of Lanczos iterations
        seed: Random seed for initial vector

    Returns:
        eigvals: Approximate eigenvalues (n_iter,)
        eigvecs: Approximate eigenvectors (n_dof, n_iter)
    """
    # Same algorithm as lanczos(), but replace A @ v with hvp_func(v)
    ...

def lanczos_hvp_smallest(hvp_func, n_dof, n_iter=20, seed=None):
    """Find smallest eigenvalue/eigenvector using matrix-free Lanczos."""
    eigvals, eigvecs = lanczos_hvp(hvp_func, n_dof, n_iter, seed)
    idx = np.argmin(eigvals)
    return eigvals[idx], eigvecs[:, idx]
```

#### Phase 3: Integration with GADESBias

**File:** `GADES/gades.py` (modify)

```python
class GADESBias:
    def __init__(self, ...,
                 eigensolver='numpy',      # 'numpy', 'lanczos', 'lanczos_hvp'
                 lanczos_iterations=20,
                 hvp_epsilon=1e-5):
        self.eigensolver = eigensolver
        self.lanczos_iterations = lanczos_iterations
        self.hvp_epsilon = hvp_epsilon

    def _get_softest_mode_hvp(self, positions, force_func):
        """Compute softest mode without forming full Hessian."""
        from .hvp import finite_difference_hvp
        from .lanczos import lanczos_hvp_smallest

        n_dof = positions.size

        def hvp_func(v):
            return finite_difference_hvp(force_func, positions, v, self.hvp_epsilon)

        eigval, eigvec = lanczos_hvp_smallest(
            hvp_func, n_dof,
            n_iter=self.lanczos_iterations
        )
        return eigval, eigvec

    def get_gad_force(self):
        positions, forces = self.backend.get_current_state()

        if self.eigensolver == 'lanczos_hvp':
            # Matrix-free approach
            eigval, eigvec = self._get_softest_mode_hvp(
                positions,
                lambda pos: self.backend.compute_forces(pos)
            )
        elif self.eigensolver == 'lanczos':
            # Full Hessian + Lanczos
            hess = self.hess_func(positions, forces, ...)
            eigval, eigvec = lanczos_smallest(hess)
        else:
            # Full Hessian + numpy.eigh
            hess = self.hess_func(positions, forces, ...)
            w, v = np.linalg.eigh(hess)
            eigval, eigvec = w[0], v[:, 0]

        # Continue with GAD force calculation...
```

#### Phase 4: Backend Support

Each backend needs a method to compute forces at arbitrary positions:

```python
class Backend:
    def compute_forces(self, positions):
        """Compute forces at given positions without modifying simulation state."""
        raise NotImplementedError

class OpenMMBackend(Backend):
    def compute_forces(self, positions):
        # Create temporary context or use setPositions carefully
        context = self.simulation.context
        old_positions = context.getState(getPositions=True).getPositions()
        context.setPositions(positions)
        forces = context.getState(getForces=True).getForces(asNumpy=True)
        context.setPositions(old_positions)  # Restore
        return forces

class ASEBackend(Backend):
    def compute_forces(self, positions):
        old_positions = self.atoms.get_positions()
        self.atoms.set_positions(positions)
        forces = self.atoms.get_forces()
        self.atoms.set_positions(old_positions)  # Restore
        return forces
```

---

### Accuracy Considerations

1. **Finite difference step size (ε):**
   - Too large: Truncation error dominates
   - Too small: Round-off error dominates
   - Optimal: ε ≈ √(machine_epsilon) × characteristic_length ≈ 1e-5 to 1e-6 Å
   - Consider Richardson extrapolation for improved accuracy

2. **Lanczos convergence:**
   - Extreme eigenvalues converge fastest (good for softest mode)
   - 10-20 iterations typically sufficient
   - Add convergence check: compare eigenvalue between iterations

3. **Re-orthogonalization:**
   - Essential for numerical stability
   - Already implemented in current Lanczos code

---

### Testing Strategy

1. **Unit tests for HVP:**
   - Compare FD-HVP against analytical Hessian × vector for Muller-Brown
   - Test different epsilon values

2. **Integration tests:**
   - Compare matrix-free vs full-Hessian eigenvalues on small systems
   - Verify GAD force direction matches between methods

3. **Benchmarks:**
   - Memory usage comparison at different system sizes
   - Timing comparison: full Hessian vs matrix-free
   - Accuracy vs iterations tradeoff

---

### Detailed Implementation Plan

#### Phase 1: Create `GADES/hvp.py`

**Status:** ✅ Complete

**Deliverable:** New module with HVP computation

```python
# GADES/hvp.py
def finite_difference_hvp(
    force_func: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    vector: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute H @ v using central finite differences."""

def finite_difference_hvp_richardson(
    force_func: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    vector: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """HVP with Richardson extrapolation for better accuracy."""
```

**Tests:**

- Compare FD-HVP vs analytical `H @ v` on Muller-Brown potential
- Test different epsilon values for accuracy/stability tradeoff

**Checklist:**

- [x] Create `GADES/hvp.py` with `finite_difference_hvp()`
- [x] Add Richardson extrapolation variant
- [x] Add forward difference variant (for reusing pre-computed forces)
- [x] Create `tests/test_hvp.py` with 17 unit tests
- [x] Verify accuracy against Muller-Brown analytical Hessian

---

#### Phase 2: Add `lanczos_hvp()` to `lanczos.py`

**Status:** ✅ Complete

**Deliverable:** Matrix-free Lanczos functions

```python
# GADES/lanczos.py (additions)
def lanczos_hvp(
    hvp_func: Callable[[np.ndarray], np.ndarray],
    n_dof: int,
    n_iter: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lanczos using only matrix-vector products."""

def lanczos_hvp_smallest(
    hvp_func: Callable[[np.ndarray], np.ndarray],
    n_dof: int,
    n_iter: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Find smallest eigenvalue/eigenvector using matrix-free Lanczos."""
```

**Implementation Note:** Nearly identical to existing `lanczos()`, just replace `A @ v` with `hvp_func(v)`.

**Checklist:**

- [x] Add `lanczos_hvp()` function
- [x] Add `lanczos_hvp_smallest()` convenience function
- [x] Add 8 tests comparing eigenvalues from `lanczos_hvp` vs `lanczos`
- [x] Test integration with finite difference HVP on Muller-Brown
- [x] Verify convergence with different iteration counts

---

#### Phase 3: Integrate into `GADESBias`

**Status:** ✅ Complete

**Deliverable:** New eigensolver option `'lanczos_hvp'`

```python
# GADES/gades.py modifications
class GADESBias:
    def __init__(
        self,
        ...,
        eigensolver: str = "numpy",  # 'numpy', 'lanczos', 'lanczos_hvp'
        hvp_epsilon: float = 1e-5,
    ):
        self.hvp_epsilon = hvp_epsilon

    def _compute_softest_mode_hvp(self, positions, atom_indices):
        """Matrix-free softest mode computation."""
        from .hvp import finite_difference_hvp
        from .lanczos import lanczos_hvp_smallest

        def hvp_func(v):
            return finite_difference_hvp(
                lambda pos: self.backend.get_forces(pos),
                positions, v, self.hvp_epsilon,
            )

        n_dof = len(atom_indices) * 3
        return lanczos_hvp_smallest(hvp_func, n_dof, self.lanczos_iterations)
```

**Note:** The existing `Backend.get_forces(positions)` method already supports computing forces at arbitrary positions.

**Checklist:**

- [x] Add `hvp_epsilon` parameter to `GADESBias.__init__()`
- [x] Add `_compute_softest_mode_hvp()` method
- [x] Update `get_gad_force()` to use HVP path when `eigensolver='lanczos_hvp'`
- [x] Add `hvp_epsilon` to `config.py` defaults
- [x] Update `GADESForceUpdater` with same parameters
- [x] Update `ASEBackend.with_gades()` factory with `hvp_epsilon` parameter
- [x] Add 6 integration tests in `TestLanczosHVPIntegration`

---

#### Phase 4: Tests & Benchmarks

**Status:** ✅ Complete

**Unit Tests (`tests/test_hvp.py`):** ✅

- `test_hvp_matches_explicit_hessian` - Compare HVP vs H@v on Muller-Brown
- `test_hvp_richardson_more_accurate` - Richardson vs simple FD
- `test_hvp_epsilon_sensitivity` - Different epsilon values
- 17 total tests covering all HVP functions

**Integration Tests (`tests/test_gades.py`):** ✅

- `test_eigensolver_lanczos_hvp_accepted` - Accept 'lanczos_hvp' option
- `test_hvp_epsilon_default` - Default from config
- `test_hvp_epsilon_custom` - Custom epsilon value
- `test_hvp_epsilon_invalid_raises` - Validation
- `test_lanczos_hvp_computes_softest_mode` - Correctness
- `test_lanczos_hvp_skips_hessian_computation` - HVP path skips Hessian

**Benchmark script (`benchmarks/hvp_scaling.py`):** ✅

```bash
# Run benchmark
python benchmarks/hvp_scaling.py --sizes 50 100 200 500 1000

# Quick test
python benchmarks/hvp_scaling.py --sizes 20 50 100 --trials 2
```

Compares memory, time, and accuracy for:
- Full Hessian + `np.linalg.eigh()`
- Matrix-based Lanczos
- Matrix-free Lanczos with HVP

**Checklist:**

- [x] Create comprehensive test suite for HVP (17 tests)
- [x] Add integration tests for `eigensolver='lanczos_hvp'` (6 tests)
- [x] Create benchmarking script (`benchmarks/hvp_scaling.py`)
- [x] Document performance characteristics in benchmark output

---

#### Phase 5: Documentation

**Status:** ✅ Complete

**Checklist:**

- [x] Add large-system usage guide (`docs/docs/guides/large_systems.md`)
- [x] Add eigensolver comparison table to guide
- [x] Add example script (`examples/MullerBrown/eigensolver_comparison.py`)
- [x] Update mkdocs.yml navigation
- [x] Update guides index with new content

---

### Recommended Usage (after implementation)

```python
from GADES.backend import ASEBackend

# Small systems (< 500 atoms): default
backend = ASEBackend.with_gades(..., eigensolver='numpy')

# Medium systems (500-5000 atoms): Lanczos on full Hessian
backend = ASEBackend.with_gades(..., eigensolver='lanczos', lanczos_iterations=15)

# Large systems (> 5000 atoms): matrix-free
backend = ASEBackend.with_gades(
    ...,
    eigensolver='lanczos_hvp',
    lanczos_iterations=25,
    hvp_epsilon=1e-5,
)
```

---

## Notes

- The existing `Backend.get_forces(positions)` method already supports arbitrary position queries
- Items 1-3 from the original audit (critical bugs) have been addressed separately
- Each phase should be implemented and tested before moving to the next

---

## Audit Findings (from codex_analysis.md)

The following issues were identified during a code audit and validated against the source code.

### Priority 1 (Critical - Must Fix)

#### A1/A2. `compute_hessian_force_fd_block_parallel` Uses Wrong Force Source

**Files:** `GADES/utils.py:217-218`

**Problem:** The function uses `get_current_state()` for reference forces, which causes two issues:

1. **Shape mismatch:** `get_current_state()` returns forces as `(N, 3)`, but `coord_indices` expects a flattened 1D array. Indexing `forces_u[coord_indices]` treats them as row indices, not DOF indices.

2. **Biased vs unbiased:** `get_current_state()` returns biased forces (for OpenMM), while `get_forces()` returns unbiased forces. Mixing them produces incorrect Hessian.

```python
# Current (wrong):
positions_array, forces_u = backend.get_current_state()  # forces_u is (N, 3), may include bias
f0 = forces_u[coord_indices]  # Wrong indexing + wrong force source
```

**Fix:** Use `get_forces()` for both reference and perturbed forces:

```python
positions_array, _ = backend.get_current_state()
f0 = backend.get_forces(positions_array)[coord_indices]  # Flattened, unbiased
```

This single change fixes both the indexing bug and the bias inconsistency.

---

#### A5. Force API Inconsistency and Incorrect Usage in `get_gad_force()`

**Files:** `GADES/backend.py:302-312`, `GADES/gades.py:564-578`

**Design Principle:**

| Method | Should Return | Use Case |
|--------|---------------|----------|
| `get_current_state()` | Total forces (including bias) | State reporting, dynamics, logging |
| `get_forces(positions)` | Unbiased forces | Hessian computation, bias calculation |

**Problem 1 - ASE `get_current_state()` returns wrong forces:**

```python
# Current (wrong) - returns unbiased forces from base_calc
def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
    positions = self.atoms.get_positions()
    self.base_calc.calculate(...)
    forces = self.base_calc.results['forces']  # Unbiased!
    return positions, forces
```

OpenMM correctly returns total forces; ASE should do the same for consistency.

**Problem 2 - `get_gad_force()` uses `get_current_state()` for bias calculation:**

```python
positions, forces = self.backend.get_current_state()
forces_u = forces[self.bias_atom_indices, :]
forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
```

GADES needs unbiased forces to compute bias from the true PES curvature, but OpenMM's `get_current_state()` includes the previous bias.

**Fix 1 - ASE `get_current_state()` should return total forces:**

```python
def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
    positions = self.atoms.get_positions()
    forces = self.atoms.get_forces()  # Goes through GADESCalculator → includes bias
    return positions, forces
```

**Fix 2 - Add `get_positions()` method to Backend interface:**

To avoid recursion when `get_current_state()` calls `atoms.get_forces()` (which triggers `GADESCalculator.calculate()` → `get_gad_force()` → infinite loop), add a dedicated `get_positions()` method:

```python
# Backend base class
def get_positions(self) -> np.ndarray:
    raise NotImplementedError

# OpenMMBackend
def get_positions(self) -> np.ndarray:
    state = self.simulation.context.getState(getPositions=True)
    return state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)

# ASEBackend
def get_positions(self) -> np.ndarray:
    return self.atoms.get_positions()
```

**Fix 3 - `get_gad_force()` and Hessian functions use `get_positions()`:**

```python
positions = self.backend.get_positions()  # No recursion risk
forces_unbiased = self.backend.get_forces(positions).reshape(-1, 3)
forces_u = forces_unbiased[self.bias_atom_indices, :]
forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
```

---

#### A3. Hessian/HVP Paths Mutate Simulation State Without Restoring

**Files:** `GADES/backend.py:146, 329`

**Problem:** `get_forces()` calls `setPositions()`/`set_positions()` to compute forces at perturbed positions but never restores the original positions. After Hessian computation, the simulation context/atoms are left at the last perturbed position.

```python
# OpenMMBackend.get_forces()
self.simulation.context.setPositions(positions)  # Never restored!

# ASEBackend.get_forces()
self.atoms.set_positions(positions)  # Never restored!
```

**Fix:** Save and restore original positions after force computation:

```python
def get_forces(self, positions: np.ndarray) -> np.ndarray:
    # Save original positions
    original_positions = ... # get current positions

    # Compute forces at new positions
    self.set_positions(positions)
    forces = self.compute_forces()

    # Restore original positions
    self.set_positions(original_positions)

    return forces
```

---

#### A4. `GADES/backend.py` Unconditionally Imports ASE

**Files:** `GADES/backend.py:8-9`

**Problem:** The module imports ASE at the top level:

```python
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
```

This causes `import GADES.backend` to fail on OpenMM-only installations even though ASE is not a hard dependency.

**Fix:** Use lazy imports or conditional imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from ase import Atoms

# Then import dynamically where needed:
class ASEBackend(Backend):
    def __init__(self, ...):
        from ase.calculators.calculator import Calculator, all_changes
        from ase import Atoms
        ...
```

---

### Priority 2 (Medium - Should Fix)

#### A6. `OpenMMBackend.is_stable()` Assumes `integrator.getTemperature()`

**Files:** `GADES/backend.py:96`

**Problem:** The method calls:

```python
target_temperature = self.simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
```

This raises `AttributeError` for integrators without temperature (e.g., `VerletIntegrator`, `BrownianIntegrator`).

**Fix:** Add try/except or hasattr check, similar to ASEBackend pattern:

```python
try:
    target_temperature = self.simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
except AttributeError:
    if not self._stability_warning_issued:
        warnings.warn("Cannot get target temperature from integrator...")
        self._stability_warning_issued = True
    return True  # Skip stability check
```

---

#### A10. GADESCalculator Shape Mismatch for Partial Atom Biasing

**Files:** `GADES/backend.py:240-242`

**Problem:** In `GADESCalculator.calculate()`:

```python
if self.force_updater.applying_bias():
    bias = self.force_updater.get_gad_force()
    self.results['forces'] = self.results['forces'] + bias
```

`get_gad_force()` returns shape `(N_bias, 3)` but `results['forces']` has shape `(N_atoms, 3)`. When `N_bias < N_atoms` (partial atom biasing), this causes:
- NumPy broadcasting error if shapes are incompatible
- Silent incorrect physics if shapes happen to broadcast

**Latent bug:** Not triggered in current examples/tests because:
1. Example `mdrun.py` biases ALL atoms (`N_bias == N_atoms`)
2. Tests use `MockGADESCalculator` that doesn't implement `calculate()`
3. Bug only manifests when integrator is attached AND step is multiple of interval

**Fix:** Create full-size bias array and add only to biased atoms:

```python
if self.force_updater.applying_bias():
    bias = self.force_updater.get_gad_force()
    full_bias = np.zeros_like(self.results['forces'])
    full_bias[self.force_updater.bias_atom_indices, :] = bias
    self.results['forces'] = self.results['forces'] + full_bias
```

---

### Priority 3 (Low - Nice to Have)

#### A7. Docstring References Wrong Function Name

**Files:** `GADES/gades.py:802`, `README.md:35`

**Problem:**
- GADESForceUpdater docstring says: `Must be created using getGADESBiasForce()`
- README step 1 says: `getGADESBiasForce`
- Actual function is: `createGADESBiasForce()`

**Fix:** Update all references to use `createGADESBiasForce`.

---

#### A8. Hessian Function Signature Mismatch in Docstrings

**Files:** `GADES/gades.py:62-63`

**Problem:** Docstring says:

```
Must accept `(system, positions, atom_indices, step_size, platform)` as input
```

But actual Hessian function signatures are:

```python
def compute_hessian_force_fd_richardson(
    backend: "Backend",
    atom_indices: Sequence[int],
    step_size: Optional[float] = 1e-4,
    platform_name: Optional[str] = 'CPU',
) -> np.ndarray:
```

**Fix:** Update docstring to reflect actual signature `(backend, atom_indices, step_size, platform)`.

---

#### A9. Python Version Requirement Mismatch

**Files:** `README.md:16`, `pyproject.toml:14`

**Problem:**
- README says: `python >= 3.10`
- pyproject.toml says: `requires-python = ">=3.9"`

**Fix:** Align both to the same version requirement.

---

#### A11. Bofill Gradient Sign Convention - Potential Confusion

**Files:** `GADES/gades.py:524-529`

**Problem:** The code comment says "Bofill uses gradients (negative forces)" but the sign handling appears inconsistent:

```python
# In _get_hessian():
bias_forces = forces[self.bias_atom_indices, :]  # forces from get_forces() = -F = ∇V (gradient)
...
# Bofill uses gradients (negative forces)
hess = get_bofill_H(
    ...
    grad_new=-bias_forces,  # This is -∇V = F (force, not gradient!)
    grad_old=-self._last_forces,
    ...
)
```

The `get_forces()` method returns `-F` (negative forces = gradient ∇V). Then `-bias_forces` gives `F` (physical force), but the parameter is named `grad_new`.

**Status:** Needs investigation - the Bofill update uses `y = grad_new - grad_old` (a difference), so the sign might cancel out and produce correct results. But the naming is confusing.

**Fix:** Verify correctness and either fix the sign or update the comment to clarify the convention.

---

#### A12. Redundant `get_positions()` Call in Parallel Hessian

**Files:** `GADES/utils.py:203, 217`

**Problem:** In `compute_hessian_force_fd_block_parallel`:

```python
def compute_hessian_force_fd_block_parallel(...):
    positions_array = backend.get_positions()  # Line 203: outer scope
    ...
    def compute_block_column(j):
        positions_array = backend.get_positions()  # Line 217: redundant inner call
        f0 = backend.get_forces(positions_array)[coord_indices]
        ...
```

The inner `get_positions()` call is redundant since `positions_array` is already captured from outer scope. This adds unnecessary overhead in the parallel function.

**Fix:** Remove the inner `get_positions()` call and use the outer-scope variable.

---

#### A13. Base Backend Class Interface Inconsistency

**Files:** `GADES/backend.py:17-62`

**Problem:** The base `Backend` class has inconsistent abstract method patterns:
- Some methods return defaults: `is_stable()` → `True`, `get_currentStep()` → `0`
- Others raise `NotImplementedError`: `get_positions()`, `get_current_state()`, `get_forces()`

This makes the interface contract unclear - which methods MUST be overridden vs. which have sensible defaults?

**Fix:** Either:
1. Make all abstract methods raise `NotImplementedError` for a strict interface, or
2. Document clearly which methods have defaults vs. which must be overridden

---

#### A14. Redundant Type Annotation in `ASEBackend.with_gades()`

**Files:** `GADES/backend.py:507`

**Problem:** Redundant inline type annotation:

```python
backend.gades_bias: "GADESBias" = gades_bias  # Type already known from assignment
```

**Fix:** Remove the redundant annotation:

```python
backend.gades_bias = gades_bias
```

---

### Checklist

- [x] A1/A2: Fix `compute_hessian_force_fd_block_parallel` to use `get_forces()` for reference forces
- [x] A5: Fix force API (ASE `get_current_state()` + `get_gad_force()` + add `get_positions()` method)
- [x] A3: Restore original positions after `get_forces()` calls
- [ ] A4: Make ASE import conditional/lazy
- [ ] A6: Handle missing `getTemperature()` in OpenMM stability check
- [x] A10: Fix GADESCalculator shape mismatch for partial atom biasing
- [ ] A11: Investigate Bofill gradient sign convention
- [x] A12: Remove redundant `get_positions()` in parallel Hessian
- [x] A13: Clarify Backend base class interface contract
- [x] A14: Remove redundant type annotation in `with_gades()`
- [x] A7: Fix wrong function name in docstrings
- [x] A8: Fix Hessian function signature in docstrings
- [x] A9: Align Python version requirements

---

## Audit Findings (from gemini_analysis.md)

The following issues were identified during a second code audit and validated against the source code.

### Priority 2 (Medium - Should Fix)

#### B1. DOF Calculation in `OpenMMBackend.is_stable()` May Miss Edge Cases

**Files:** `GADES/backend.py:83-95`

**Problem:** The DOF calculation handles standard cases (+3 per particle, -1 per constraint, -3 for CMMotionRemover) but may not correctly handle virtual sites, rigid bodies, barostats, or other motion removers.

**Fix:** Use OpenMM's built-in DOF calculation if available, or document limitations clearly.

---

#### B2. Missing `OpenMMBackend` Unit Tests

**Files:** `tests/test_backend.py`

**Problem:** Test suite has comprehensive `ASEBackend` coverage but no corresponding `OpenMMBackend` tests. This is a significant gap, especially for the complex `is_stable()` method.

**Fix:** Add `TestOpenMMBackend*` test classes (requires mock OpenMM simulation or simple test system).

---

### Priority 3 (Low - Nice to Have)

#### B3. Move OpenMM-Specific Code to `backend.py`

**Files:** `GADES/utils.py:105-136`, `GADES/gades.py:722-764`

**Problem:** OpenMM-specific functions are scattered: `_get_openMM_forces()` in utils.py, `createGADESBiasForce()` in gades.py.

**Fix:** Consolidate OpenMM-specific code in `backend.py`.

---

#### B4. Add Development Dependencies to `pyproject.toml`

**Files:** `pyproject.toml`

**Problem:** pytest and coverage are not listed as dev dependencies.

**Fix:** Add `[project.optional-dependencies]` with `dev = ["pytest", "pytest-cov", "coverage"]`.

---

#### B5. Missing Developer Documentation in README

**Files:** `README.md`

**Problem:** README lacks instructions for running tests, setting up dev environment, or contributing.

**Fix:** Add "Development" section with setup and test commands.

---

#### B6. Clarify Recommended Hessian Function in `utils.py`

**Files:** `GADES/utils.py`

**Problem:** Multiple Hessian functions exist; recommended one (`compute_hessian_force_fd_richardson`) not prominently highlighted.

**Fix:** Add module docstring or `__all__` to highlight recommended function.

---

## Priority 0 (Critical - Correctness Issues)

These issues affect the correctness of GADES results and should be addressed before production use.

### C1. OpenMM vs ASE Bias Direction Inconsistency

**Files:** `GADES/backend.py:328-344`, `GADES/backend.py:512-518`

**Problem:** OpenMM and ASE backends apply bias forces in opposite directions:
- OpenMM: `CustomExternalForce` energy is `E = fx*x + fy*y + fz*z`, so force = `(-fx, -fy, -fz)`
- ASE: `GADESCalculator` adds bias directly: `forces + full_bias`

When the same `biased_force_values` array is passed, OpenMM applies `-bias` while ASE applies `+bias`.

**Fix:** Either:
- (A) Negate bias values before passing to OpenMM's `apply_bias()`, or
- (B) Change `CustomExternalForce` energy expression to `E = -fx*x - fy*y - fz*z`

**Impact:** One backend produces incorrect dynamics relative to the GAD formulation.

---

### C2. lanczos_hvp Sign Flip (Finds Wrong Mode for Large Systems)

**Files:** `GADES/gades.py:439-469`, `GADES/hvp.py:20-87`

**Problem:** The HVP module expects a force function returning `F = -∇E`, but `backend.get_forces()` returns gradients `∇E = -F`. This flips the sign of the Hessian-vector product, causing `lanczos_hvp_smallest` to find the **largest** curvature mode instead of the softest.

**Evidence:**
- `hvp.py:35`: "Forces should be the negative gradient of energy (F = -∇E)"
- `backend.py:326`: `return -forces.flatten()` (returns gradient, not force)
- `gades.py:454`: Passes `backend.get_forces()` directly to HVP

**Fix:** Either:
- (A) Negate the result in `force_func_biased` before passing to HVP, or
- (B) Change `backend.get_forces()` to return actual forces (remove negation)

**Impact:** `eigensolver='lanczos_hvp'` produces incorrect results for large systems.

---

### C3. createGADESBiasForce Documentation Error

**Files:** `GADES/backend.py:367-398`

**Problem:** Docstring states the force is `F(x,y,z) = fx*x + fy*y + fz*z`, but `CustomExternalForce` defines an **energy** expression. The actual force is `F = (-fx, -fy, -fz)`.

**Fix:** Correct the docstring to accurately describe the sign convention.

---

## Priority 1 (High - Thread Safety)

### H1. Parallel Hessian Thread Safety Issue

**Files:** `GADES/utils.py:202-222`

**Problem:** `compute_hessian_force_fd_block_parallel` calls `backend.get_forces()` from multiple threads. Each call invokes `setPositions()` on a shared OpenMM context or ASE atoms object without synchronization, causing race conditions that corrupt Hessian calculations.

**Current mitigation:** Docstring recommends serial version for most use cases.

**Fix options:**
- (A) Add mutex/lock around position updates in parallel version
- (B) Deprecate parallel version with warning
- (C) Create per-thread context copies (expensive)

---

## Priority 2 (Medium - Dependencies)

### D1. Remove Unused jax Dependency

**Files:** `pyproject.toml:20`

**Problem:** `jax` is listed as a hard dependency but is not used anywhere in the GADES codebase. This adds unnecessary install friction and dependency conflicts.

**Fix:** Remove `jax` from dependencies, or move to optional extras if planned for future use.

---

### Checklist

**Completed:**
- [x] B1: Improve DOF calculation robustness in OpenMMBackend.is_stable()
- [x] B2: Add OpenMMBackend unit tests
- [x] B3: Move OpenMM-specific code to backend.py
- [x] B4: Add dev dependencies to pyproject.toml
- [x] B5: Add developer documentation to README
- [x] B6: Clarify recommended Hessian function

**New Issues (from codex_analysis.md):**
- [x] C1: Fix OpenMM vs ASE bias direction inconsistency (Critical) ✅
- [x] C2: Fix lanczos_hvp sign flip (Critical) ✅ (fixed by C1)
- [x] C3: Fix createGADESBiasForce documentation (Critical) ✅ (fixed by C1)
- [x] H1: Address parallel Hessian thread safety (High) ✅ (deprecated function)
- [x] D1: Remove unused jax dependency (Medium) ✅

**Additional Issues (from codex_analysis.md round 2):**
- [x] A1: Fix get_forces docstring - says returns (N,3) but implementations return (3N,) (High) ✅
- [x] A2: Add bounds checking for bias_atom_indices against system size (Medium) ✅
- [x] A3: Fix stability_interval for ASE - accepted but never used (Medium) ✅
- [x] A4: Add integration tests for lanczos_hvp and Bofill update code paths (Medium) ✅
- [x] A5: Fix clamp_magnitude docstring - says per-component but implementation is per-vector (Low) ✅

**Follow-up Audit Findings (from codex_analysis.md round 3):**

- [x] F1: ASE bias persistence and stability handling (High) ✅ - Implemented persistent bias in GADESCalculator
- [x] F2: ASE bias only applied on update steps, not persisted between updates unlike OpenMM (High) ✅ - Fixed by F1
- [x] F3: `ASEBackend.with_gades` bypasses bounds validation since GADESBias created with `backend=None` (Medium) ✅
- [x] F4: Warn when `use_bofill_update=True` with `eigensolver='lanczos_hvp'` since Bofill is silently ignored (Low) ✅
