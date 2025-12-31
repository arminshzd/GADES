# GADES Test Suite Documentation

This document describes the test suite for GADES (Gentlest Ascent Dynamics for Enhanced Sampling). Each test file validates a specific component of the library.

## Overview

| Test File | Component | Tests |
|-----------|-----------|-------|
| `conftest.py` | Test fixtures | Mock objects and shared fixtures |
| `test_hvp.py` | Hessian-Vector Products | 16 tests |
| `test_lanczos.py` | Lanczos Eigensolvers | 28 tests |
| `test_bofill.py` | Hessian Update Algorithms | 17 tests |
| `test_gades.py` | Core GADESBias | 48 tests |
| `test_validation.py` | Input Validation | 35 tests |
| `test_utils.py` | Utility Functions | 14 tests |
| `test_backend.py` | ASE Backend | 24 tests |

---

## conftest.py - Test Fixtures

Provides shared fixtures and mock objects used across all test files.

### MockBackend

A mock simulation backend that simulates OpenMM/ASE behavior without requiring actual MD engines.

**Purpose**: Allow testing GADESBias logic without expensive MD setup.

**Key features**:
- Configurable number of atoms
- Controllable positions, forces, and step counter
- Records bias application for verification
- Stability flag for testing stability checks

### Hessian Function Fixtures

#### `simple_hessian_func`
Returns a diagonal Hessian with eigenvalues [1, 2, 3, ...]. Used for predictable eigenvector testing since the smallest eigenvector is always [1, 0, 0, ...].

#### `negative_eigenvalue_hessian_func`
Returns a Hessian with one negative eigenvalue (-1), simulating a saddle point. Tests that GADES correctly identifies negative curvature directions.

---

## test_hvp.py - Hessian-Vector Product Tests

Tests the finite difference Hessian-vector product (HVP) computation used for matrix-free eigensolvers.

### TestFiniteDifferenceHVP

**What it tests**: Central difference HVP approximation: `H @ v ≈ (F(x-εv) - F(x+εv)) / (2ε)`

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_hvp_matches_explicit_hessian_x_direction` | HVP along x-axis matches analytical H @ v | Error < 0.01% |
| `test_hvp_matches_explicit_hessian_y_direction` | HVP along y-axis matches analytical H @ v | Error < 0.01% |
| `test_hvp_matches_explicit_hessian_diagonal` | HVP along diagonal matches analytical H @ v | Error < 0.01% |
| `test_hvp_matches_explicit_hessian_random_vector` | HVP for random vectors matches analytical | Error < 0.01% |
| `test_hvp_zero_vector_returns_zero` | H @ 0 = 0 | Exact zero output |
| `test_hvp_linearity_in_vector` | H @ (αv₁ + v₂) = α(H @ v₁) + (H @ v₂) | Linearity preserved |
| `test_hvp_different_positions` | HVP works at various PES locations | Consistent accuracy |
| `test_hvp_shape_mismatch_raises` | Mismatched position/vector shapes | ValueError raised |

**Why these tests matter**: HVP is the foundation of matrix-free Lanczos. Errors here propagate to eigenvector estimates, causing wrong bias directions.

### TestFiniteDifferenceHVPRichardson

**What it tests**: Richardson extrapolation for improved HVP accuracy (O(ε⁴) vs O(ε²)).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_richardson_more_accurate` | Richardson beats simple central difference | Lower error |
| `test_richardson_matches_explicit_hessian` | Richardson HVP matches analytical H @ v | Error < 0.0001% |
| `test_richardson_high_accuracy` | Richardson achieves machine precision | Error < 10⁻⁷ |

**Why it matters**: Richardson extrapolation enables accurate HVP with larger ε, improving numerical stability.

### TestFiniteDifferenceHVPForward

**What it tests**: Forward difference HVP (cheaper but less accurate).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_forward_hvp_approximate` | Forward HVP approximately correct | Error < 1% |
| `test_forward_hvp_reuses_force` | Can reuse existing force evaluation | Reasonable accuracy |

### TestHVPEpsilonSensitivity

**What it tests**: Sensitivity of HVP accuracy to step size ε.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_epsilon_too_large` | Large ε has more truncation error | error(ε=0.01) > error(ε=10⁻⁵) |
| `test_optimal_epsilon_range` | ε ∈ [10⁻⁴, 10⁻⁶] gives good accuracy | Error < 0.1% |

**Why it matters**: Too large ε → truncation error; too small ε → numerical noise. Tests verify optimal range.

### TestHVPHigherDimensions

**What it tests**: HVP with multi-atom (3D position) systems.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_hvp_3d_positions` | HVP works with (N, 3) shaped positions | Correct result |
| `test_hvp_preserves_shape_info` | Output is always flattened | Shape (3N,) |

---

## test_lanczos.py - Lanczos Eigensolver Tests

Tests both matrix-based and matrix-free Lanczos algorithms.

### TestLanczosBasic

**What it tests**: Core Lanczos algorithm correctness.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_identity_matrix` | I has all eigenvalues = 1 | All λ = 1 |
| `test_diagonal_matrix` | Diagonal entries recovered | Exact eigenvalues |
| `test_symmetric_2x2` | 2×2 symmetric matrix | Matches np.linalg.eigh |
| `test_symmetric_random` | Random symmetric matrix | Matches np.linalg.eigh |
| `test_eigenvector_orthogonality` | V^T V = I | Orthonormal eigenvectors |
| `test_eigenvector_correctness` | Av = λv | Eigenpair identity satisfied |
| `test_fewer_iterations` | Extreme eigenvalues converge first | min/max λ accurate with few iterations |

**Why it matters**: Verifies Lanczos finds true eigenvalues/eigenvectors, not numerical artifacts.

### TestLanczosSmallest

**What it tests**: `lanczos_smallest()` convenience function for finding the softest mode.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_finds_smallest_eigenvalue` | Finds λ_min correctly | Exact smallest eigenvalue |
| `test_negative_eigenvalue` | Finds negative λ (saddle point) | Returns -2.0 for diag with -2 |
| `test_eigenvector_normalized` | Returned eigenvector has ||v|| = 1 | Normalized output |
| `test_eigenvector_correct` | Returned eigenvector satisfies Av = λv | Valid eigenpair |

**Why it matters**: GADES relies on `lanczos_smallest` to find the softest mode direction.

### TestLanczosShiftInvert

**What it tests**: Shift-and-invert Lanczos for interior eigenvalues.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_finds_eigenvalue_near_shift` | Finds λ closest to σ | Returns λ ≈ σ |
| `test_finds_eigenvalue_near_different_shifts` | Works for various σ | Correct interior λ |
| `test_random_matrix_interior_eigenvalue` | Works on random matrices | High accuracy |

### TestLanczosHVP

**What it tests**: Matrix-free Lanczos using matvec function instead of explicit matrix.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_lanczos_hvp_matches_lanczos` | HVP and matrix Lanczos give same results | Identical eigenvalues/vectors |
| `test_lanczos_hvp_diagonal` | Works on diagonal via efficient matvec | Correct eigenvalues |
| `test_lanczos_hvp_identity` | I via matvec gives all λ = 1 | All eigenvalues = 1 |

**Why it matters**: Matrix-free Lanczos enables O(N) memory scaling for large systems.

### TestLanczosHVPSmallest

**What it tests**: `lanczos_hvp_smallest()` function combining HVP with smallest eigenvalue search.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_hvp_smallest_matches_smallest` | HVP version matches matrix version | Same results |
| `test_hvp_smallest_negative_eigenvalue` | Finds negative eigenvalue via HVP | Correct saddle detection |

### TestLanczosHVPWithFiniteDifference

**What it tests**: Full integration: finite difference HVP → Lanczos → eigenvalue.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_lanczos_hvp_with_muller_brown` | End-to-end on Muller-Brown potential | Matches analytical |
| `test_lanczos_hvp_smallest_muller_brown` | Finds softest mode at saddle point | Correct negative λ, aligned eigenvector |
| `test_hvp_lanczos_higher_dimensions` | Works in 6D | Finds smallest eigenvalue |

**Why it matters**: Validates the complete matrix-free pipeline used in GADES for large systems.

---

## test_bofill.py - Hessian Update Algorithm Tests

Tests Bofill, SR1, and BFGS quasi-Newton Hessian update formulas.

### TestSpectralAbs

**What it tests**: Helper function that computes |H| (spectral absolute value).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_positive_eigenvalues_unchanged` | Positive definite H unchanged | |H| = H |
| `test_negative_eigenvalues_flipped` | Negative λ → positive | Signs flipped |
| `test_symmetric_matrix` | Mixed eigenvalues handled | All λ > 0 in output |

**Why it matters**: Bofill update uses |H| internally; incorrect |H| breaks the update.

### TestBofillQuadratic

**What it tests**: Bofill on pure quadratic potentials (constant Hessian).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_perfect_quadratic_2d` | Bofill satisfies secant condition H·Δx = Δg | Exact condition |
| `test_diagonal_quadratic_3d` | Works in 3D | Secant condition satisfied |

**Why it matters**: For quadratic potentials, Bofill should exactly satisfy the secant condition.

### TestBofillMullerBrown

**What it tests**: Bofill on the non-quadratic Muller-Brown potential.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_single_step_approximation` | One Bofill step improves Hessian | Eigenvalues within 15% |
| `test_multiple_steps_convergence` | Multiple steps converge toward true H | Eigenvalue signs match |

**Why it matters**: Real potentials are non-quadratic; Bofill should still track the Hessian reasonably.

### TestSR1Update

**What it tests**: Symmetric Rank-1 Hessian update.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_quadratic_secant_condition` | SR1 satisfies H·Δx = Δg | Exact condition |
| `test_sr1_symmetry` | SR1 preserves Hessian symmetry | H = H^T |
| `test_sr1_skip_small_denominator` | Skips update when unstable | Returns original H |

### TestBFGSUpdate

**What it tests**: BFGS Hessian update (for comparison).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_bfgs_secant_condition` | BFGS satisfies its update condition | Valid update |
| `test_bfgs_symmetry` | BFGS preserves symmetry | H = H^T |
| `test_bfgs_positive_definite` | BFGS maintains positive definiteness | All λ > 0 |

**Why Bofill vs BFGS matters**: BFGS enforces positive definiteness (good for minimization), but Bofill can capture negative curvature (needed for transition states).

### TestBofillVsBFGS

**What it tests**: Comparison of Bofill and BFGS in different PES regions.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_near_minimum` | Both give reasonable results near minimum | Valid Hessians |
| `test_near_saddle` | Bofill captures negative curvature | Negative λ preserved |

### TestBofillEdgeCases

**What it tests**: Edge cases and numerical stability.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_zero_step` | Zero step returns original H | No change |
| `test_large_step` | Large steps don't cause NaN/Inf | Finite output |
| `test_flattened_vs_shaped_input` | (3N,) and (N,3) inputs work | Same result |

---

## test_gades.py - Core GADESBias Tests

Tests the main GADESBias class that computes and applies bias forces.

### TestGADESBiasInitialization

**What it tests**: Constructor behavior and default values.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_basic_initialization` | All parameters stored correctly | Attributes match inputs |
| `test_default_hess_step_size` | Default δ = 10⁻⁵ | Correct default |
| `test_is_biasing_initial_state` | Not biasing initially | is_biasing = False |
| `test_check_stability_initial_state` | Not checking stability initially | check_stability = False |

### TestGADESBiasSetters

**What it tests**: Runtime parameter modification.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_set_kappa` | Can change κ at runtime | New value stored |
| `test_set_hess_step_size` | Can change Hessian step size | New value stored |

### TestGetGadForce

**What it tests**: The core GADES force computation: F_bias = -κ(F·n)n

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_get_gad_force_shape` | Output shape = (n_bias_atoms, 3) | Correct shape |
| `test_get_gad_force_direction` | Bias force along softest mode | Correct direction |
| `test_get_gad_force_clamping` | Force magnitude ≤ clamp_magnitude | Clamped output |
| `test_get_gad_force_kappa_scaling` | F(κ=0.5) = 0.5 × F(κ=1.0) | Linear in κ |

**Why it matters**: This is the heart of GADES. Wrong force computation → wrong dynamics.

### TestApplyingBias

**What it tests**: Timing of bias updates.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_applying_bias_at_interval` | True at steps 0, 100, 200... | Updates at multiples |
| `test_applying_bias_between_intervals` | False at steps 50, 99, 150... | No updates between |
| `test_applying_bias_negative_step` | False for negative steps | Handles edge case |

### TestRegisterNextStep

**What it tests**: Computing steps until next event (bias or stability check).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_register_next_step_basic` | Returns correct steps to next event | Correct count |
| `test_register_next_step_no_stability_interval` | Works without stability checks | Only bias interval |

### TestEigensolverIntegration

**What it tests**: Different eigensolver backends (numpy, lanczos, lanczos_hvp).

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_default_eigensolver_is_numpy` | Default = 'numpy' | Correct default |
| `test_eigensolver_lanczos` | Accepts 'lanczos' | Valid option |
| `test_invalid_eigensolver_raises` | Rejects invalid options | ValueError |
| `test_lanczos_gives_same_direction` | Lanczos and numpy give same direction | Forces aligned |

**Why it matters**: All eigensolvers must produce consistent bias directions.

### TestLanczosHVPIntegration

**What it tests**: Matrix-free Lanczos integration with GADESBias.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_eigensolver_lanczos_hvp_accepted` | Accepts 'lanczos_hvp' | Valid option |
| `test_hvp_epsilon_default` | Uses default ε | Correct value |
| `test_hvp_epsilon_custom` | Custom ε works | Value stored |
| `test_hvp_epsilon_invalid_raises` | Rejects negative/zero ε | ValueError |
| `test_lanczos_hvp_computes_softest_mode` | HVP finds same mode as numpy | Aligned eigenvectors |
| `test_lanczos_hvp_skips_hessian_computation` | HVP path doesn't call hess_func | Zero hess_func calls |
| `test_lanczos_hvp_with_logging_no_crash` | HVP + logging doesn't crash | No TypeError |
| `test_lanczos_hvp_logs_eigenvector_but_skips_eigenvalues` | Logs evec/xyz, skips eval | Correct file contents |
| `test_lanczos_hvp_logging_warning_issued` | Warning issued on init | Warning in log |

**Why it matters**: Verifies the matrix-free path works correctly, avoids Hessian computation, and handles logging gracefully.

### TestCloseLogsErrorHandling

**What it tests**: Error handling in `_close_logs()` method.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_close_logs_handles_exception_gracefully` | File close errors warn, don't raise | UserWarning issued |
| `test_close_logs_skips_already_closed_files` | Idempotent on already-closed files | No exception |

**Why it matters**: Ensures cleanup errors don't mask other exceptions or crash during shutdown.

### TestBofillIntegration

**What it tests**: Bofill quasi-Newton Hessian update integration.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_bofill_disabled_by_default` | Default = False | Off by default |
| `test_bofill_enabled` | Can enable Bofill | Flag stored |
| `test_full_hessian_interval_default` | Default = interval × multiplier | Correct default |
| `test_bofill_first_call_computes_full_hessian` | First call computes full H | One hess_func call |
| `test_bofill_uses_approximation_between_intervals` | Subsequent calls use Bofill | No extra hess_func calls |
| `test_bofill_recomputes_at_full_interval` | Full H recomputed at interval | hess_func called again |

**Why it matters**: Validates the Bofill workflow saves expensive Hessian computations.

### TestComputeSoftestMode

**What it tests**: Internal `_compute_softest_mode` method.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_compute_softest_mode_numpy` | numpy eigensolver finds smallest λ | Correct eigenvalue/vector |
| `test_compute_softest_mode_lanczos` | lanczos eigensolver finds smallest λ | Correct (within tolerance) |

---

## test_validation.py - Input Validation Tests

Tests that invalid inputs are properly rejected with clear error messages.

### TestCreateGADESBiasForceValidation

**What it tests**: Validation for the OpenMM force creation function.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_valid_n_particles` | Positive integer accepted | No error |
| `test_zero_particles` | Zero is valid (edge case) | No error |
| `test_negative_n_particles` | Negative rejected | ValueError |
| `test_float_n_particles` | Floats rejected | ValueError |
| `test_string_n_particles` | Strings rejected | ValueError |
| `test_none_n_particles` | None rejected | ValueError |
| `test_numpy_int_n_particles` | NumPy integers accepted | No error |

### TestGADESBiasValidation

**What it tests**: Validation for GADESBias constructor parameters.

#### hess_func validation
| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_valid_hess_func` | Callable accepted | No error |
| `test_non_callable_hess_func` | Non-callable rejected | TypeError |
| `test_none_hess_func` | None rejected | TypeError |

#### bias_atom_indices validation
| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_valid_bias_atom_indices_list` | List of ints accepted | No error |
| `test_valid_bias_atom_indices_numpy` | NumPy array accepted | No error |
| `test_valid_bias_atom_indices_tuple` | Tuple accepted | No error |
| `test_empty_bias_atom_indices` | Empty list rejected | ValueError |
| `test_non_sequence_bias_atom_indices` | Single int rejected | TypeError |
| `test_negative_bias_atom_indices` | Negative indices rejected | ValueError |
| `test_float_bias_atom_indices` | Float indices rejected | ValueError |

#### clamp_magnitude validation
| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_valid_clamp_magnitude` | Positive float accepted | No error |
| `test_zero_clamp_magnitude` | Zero rejected | ValueError |
| `test_negative_clamp_magnitude` | Negative rejected | ValueError |
| `test_string_clamp_magnitude` | String rejected | ValueError |

#### interval validation
| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_valid_interval` | Positive int accepted | No error |
| `test_zero_interval` | Zero rejected | ValueError |
| `test_negative_interval` | Negative rejected | ValueError |
| `test_float_interval` | Float rejected | ValueError |
| `test_small_interval_warning` | interval < 100 warns and overrides to 110 | Warning + override |

#### kappa validation
| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_valid_kappa_in_range` | κ ∈ (0, 1] accepted silently | No warning |
| `test_kappa_exactly_one` | κ = 1.0 accepted | No warning |
| `test_kappa_greater_than_one_warning` | κ > 1 warns | UserWarning |
| `test_kappa_zero_warning` | κ = 0 warns | UserWarning |
| `test_kappa_negative_warning` | κ < 0 warns | UserWarning |

**Why validation matters**: Clear error messages help users fix configuration mistakes quickly.

---

## test_utils.py - Utility Function Tests

Tests utility functions used throughout GADES.

### TestClampForceMagnitudes

**What it tests**: Force magnitude clamping to prevent numerical instabilities.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_no_clamping_needed` | Small forces unchanged | Exact preservation |
| `test_single_vector_clamping` | Large vector scaled to max | ||F|| ≤ max |
| `test_multiple_vectors_mixed` | Only large vectors clamped | Selective clamping |
| `test_zero_vector` | Zero vectors stay zero | No division by zero |
| `test_large_array` | Works with many atoms | All ||F|| ≤ max |
| `test_preserves_direction` | Direction preserved when clamping | F/||F|| unchanged |
| `test_exact_max_force` | At exactly max, unchanged | No unnecessary change |
| `test_negative_components` | Signs preserved | Direction maintained |
| `test_input_shape_preserved` | Output shape = input shape | Shape conservation |

**Why it matters**: Unclamped large forces cause MD integrator instabilities.

### TestMullerBrownPotential

**What it tests**: The Muller-Brown test potential used throughout testing.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_muller_brown_potential_shape` | Scalar energy per point | Correct shape |
| `test_muller_brown_force_shape` | 2D force vector per point | Correct shape |
| `test_muller_brown_hessian_shape` | 2×2 Hessian per point | Correct shape |
| `test_muller_brown_known_minimum` | Energy < 0 at known minimum | Deep well |

---

## test_backend.py - ASE Backend Tests

Tests the ASE (Atomic Simulation Environment) integration.

### TestBackendInterface

**What it tests**: Base Backend class interface.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_backend_default_is_stable` | Default is_stable = True | Always stable |
| `test_backend_default_get_current_step` | Default step = 0 | Step counter works |
| `test_backend_has_required_methods` | All interface methods exist | Complete interface |

### TestASEBackendInitialization

**What it tests**: ASEBackend constructor.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_basic_initialization` | Stores atoms, calculator, etc. | Correct attributes |
| `test_initialization_with_target_temperature` | Explicit temperature stored | Value preserved |
| `test_initialization_sets_atoms_calc` | atoms.calc set to calculator | Wired correctly |

### TestASEBackendGetTargetTemperature

**What it tests**: Target temperature detection from various sources.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_explicit_target_temperature` | Explicit overrides all | Returns explicit |
| `test_langevin_integrator_temp` | Reads from Langevin `temp` | Correct value |
| `test_berendsen_integrator_temperature` | Reads from Berendsen `temperature` | Correct value |
| `test_explicit_overrides_integrator` | Explicit beats integrator | Explicit wins |
| `test_no_target_available` | Returns None when unavailable | Handles missing |
| `test_nve_integrator_no_temp` | NVE has no target | Returns None |

### TestASEBackendIsStable

**What it tests**: Temperature stability checks.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_stable_when_temp_close` | |T - T_target| < 50K → stable | True |
| `test_stable_within_threshold` | 40K difference → stable | True |
| `test_unstable_above_threshold` | 60K difference → unstable | False |
| `test_unstable_below_threshold` | T too low → unstable | False |
| `test_exactly_at_threshold` | 50K difference → stable | True (boundary) |
| `test_no_target_returns_true_with_warning` | Missing target warns but returns True | Warning issued |
| `test_warning_only_issued_once` | Warning not repeated | Single warning |

**Why stability matters**: Biasing an unstable (hot) system can cause numerical explosions.

### TestASEBackendWithGades

**What it tests**: Factory method that creates fully-wired ASEBackend + GADESBias.

| Test | What it validates | Expected outcome |
|------|-------------------|------------------|
| `test_with_gades_basic` | Creates working backend | All components present |
| `test_with_gades_wires_backend_reference` | GADESBias.backend set correctly | Circular reference works |
| `test_with_gades_sets_atoms_calc` | atoms.calc is GADESCalculator | Calculator installed |
| `test_with_gades_optional_parameters` | Optional params passed through | All options work |
| `test_with_gades_gades_bias_none_for_regular_init` | Regular init leaves gades_bias=None | Factory vs manual |

**Why the factory matters**: Simplifies setup by handling the complex wiring between components.

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_lanczos.py

# Run specific test class
pytest tests/test_gades.py::TestGetGadForce

# Run specific test
pytest tests/test_hvp.py::TestFiniteDifferenceHVP::test_hvp_linearity_in_vector

# Run with coverage
pytest tests/ --cov=GADES --cov-report=html
```

## Test Design Principles

1. **Ground Truth Comparison**: Tests compare against analytical solutions (e.g., np.linalg.eigh) whenever possible.

2. **Muller-Brown as Reference**: The 2D Muller-Brown potential provides a well-studied test surface with known minima, saddle points, and analytical Hessians.

3. **Mock Objects**: MockBackend and MockAtoms allow testing GADES logic without expensive MD engines.

4. **Edge Cases**: Tests explicitly check boundary conditions (zero vectors, negative indices, exact thresholds).

5. **Integration Tests**: End-to-end tests verify components work together (e.g., HVP → Lanczos → GADESBias).

6. **Validation Tests**: Comprehensive input validation ensures users get clear error messages for misconfigurations.
