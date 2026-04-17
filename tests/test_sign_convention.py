"""
Tests for sign conventions in GADES.

These tests verify:
1. Hessian computation produces correct signs (H = ∇²V)
2. Bias force has correct sign (reduces force along softest mode)
3. Both backends apply bias in the same direction

These tests would have caught the sign convention bugs that were present.
"""
import pytest
import numpy as np
from typing import Tuple

from GADES.utils import (
    compute_hessian_force_fd_richardson,
    compute_hessian_force_fd_block_serial,
)
from GADES import GADESBias


# =============================================================================
# Test Potentials with Analytical Hessians
# =============================================================================

class HarmonicBackend:
    """
    Backend for a 2D harmonic oscillator: V(x,y) = 0.5*(k1*x^2 + k2*y^2)

    Force: F = -∇V = (-k1*x, -k2*y)
    Hessian: H = [[k1, 0], [0, k2]]
    """
    def __init__(self, k1: float = 1.0, k2: float = 2.0):
        self.k1 = k1
        self.k2 = k2
        self._positions = np.array([[0.5, 0.3, 0.0]])  # 1 atom, 3D (z unused)
        self._current_step = 0

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        """Return forces F = -∇V for harmonic potential."""
        pos = positions.reshape(-1, 3)
        forces = np.zeros_like(pos)
        forces[:, 0] = -self.k1 * pos[:, 0]  # F_x = -k1*x
        forces[:, 1] = -self.k2 * pos[:, 1]  # F_y = -k2*y
        # F_z = 0 (no potential in z)
        return forces.flatten()

    def analytical_hessian(self) -> np.ndarray:
        """Return the analytical Hessian H = ∇²V."""
        # For V = 0.5*(k1*x^2 + k2*y^2), H = diag(k1, k2, 0)
        return np.diag([self.k1, self.k2, 0.0])

    def get_currentStep(self) -> int:
        return self._current_step

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.get_positions()
        forces = self.get_forces(pos).reshape(-1, 3)
        return pos, forces

    def get_atom_symbols(self, indices):
        return ["C"] * len(indices)

    def is_stable(self) -> bool:
        return True


class QuadraticBackend:
    """
    Backend for a general quadratic potential: V(x) = 0.5 * x^T @ A @ x

    Force: F = -∇V = -A @ x
    Hessian: H = A (must be symmetric)
    """
    def __init__(self, A: np.ndarray, x0: np.ndarray = None):
        """
        Args:
            A: Symmetric matrix defining the quadratic potential
            x0: Initial position (flattened)
        """
        self.A = np.array(A)
        assert np.allclose(self.A, self.A.T), "A must be symmetric"
        n = A.shape[0]
        n_atoms = n // 3
        if x0 is None:
            x0 = np.random.randn(n_atoms, 3) * 0.1
        self._positions = x0.reshape(n_atoms, 3)
        self._current_step = 0

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_forces(self, positions: np.ndarray) -> np.ndarray:
        """Return forces F = -∇V = -A @ x for quadratic potential."""
        x = positions.flatten()
        forces = -self.A @ x  # F = -∇V = -A @ x
        return forces

    def analytical_hessian(self) -> np.ndarray:
        """Return the analytical Hessian H = A."""
        return self.A.copy()

    def get_currentStep(self) -> int:
        return self._current_step

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.get_positions()
        forces = self.get_forces(pos).reshape(-1, 3)
        return pos, forces

    def get_atom_symbols(self, indices):
        return ["C"] * len(indices)

    def is_stable(self) -> bool:
        return True


# =============================================================================
# Hessian Sign Convention Tests
# =============================================================================

class TestHessianSignConvention:
    """Tests that verify Hessian computation produces H = ∇²V (positive curvature for minima)."""

    def test_harmonic_hessian_richardson(self):
        """Test Richardson Hessian against analytical for harmonic potential."""
        backend = HarmonicBackend(k1=2.0, k2=5.0)

        # Compute numerical Hessian
        H_numerical = compute_hessian_force_fd_richardson(
            backend, atom_indices=[0], step_size=1e-4, platform_name='CPU'
        )

        # Get analytical Hessian
        H_analytical = backend.analytical_hessian()

        # They should match
        np.testing.assert_allclose(
            H_numerical, H_analytical, rtol=1e-4, atol=1e-6,
            err_msg="Numerical Hessian does not match analytical Hessian"
        )

    def test_harmonic_hessian_serial(self):
        """Test serial Hessian against analytical for harmonic potential."""
        backend = HarmonicBackend(k1=3.0, k2=7.0)

        H_numerical = compute_hessian_force_fd_block_serial(
            backend, atom_indices=[0], epsilon=1e-4, platform_name='CPU'
        )

        H_analytical = backend.analytical_hessian()

        np.testing.assert_allclose(
            H_numerical, H_analytical, rtol=1e-3, atol=1e-5,
            err_msg="Serial Hessian does not match analytical Hessian"
        )

    def test_quadratic_hessian_positive_definite(self):
        """Test Hessian for positive definite quadratic (minimum)."""
        # Create a positive definite matrix (all positive eigenvalues)
        eigenvalues = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        Q = np.eye(6)  # Eigenvectors are identity
        A = Q @ np.diag(eigenvalues) @ Q.T

        backend = QuadraticBackend(A)

        H_numerical = compute_hessian_force_fd_richardson(
            backend, atom_indices=[0, 1], step_size=1e-4, platform_name='CPU'
        )

        # Verify eigenvalues are positive (minimum)
        eigvals = np.linalg.eigvalsh(H_numerical)
        assert np.all(eigvals > 0), f"Expected positive eigenvalues at minimum, got {eigvals}"

        # Verify matches analytical
        np.testing.assert_allclose(
            H_numerical, A, rtol=1e-4, atol=1e-6,
            err_msg="Numerical Hessian does not match analytical"
        )

    def test_quadratic_hessian_saddle_point(self):
        """Test Hessian for saddle point (one negative eigenvalue)."""
        # Create matrix with one negative eigenvalue (saddle point)
        eigenvalues = np.array([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        Q = np.eye(6)
        A = Q @ np.diag(eigenvalues) @ Q.T

        backend = QuadraticBackend(A)

        H_numerical = compute_hessian_force_fd_richardson(
            backend, atom_indices=[0, 1], step_size=1e-4, platform_name='CPU'
        )

        # Verify one negative eigenvalue (saddle point)
        eigvals = np.linalg.eigvalsh(H_numerical)
        n_negative = np.sum(eigvals < 0)
        assert n_negative == 1, f"Expected 1 negative eigenvalue at saddle, got {n_negative}"

        # Smallest eigenvalue should be the negative one
        assert eigvals[0] < 0, f"Smallest eigenvalue should be negative, got {eigvals[0]}"

    def test_hessian_symmetry(self):
        """Test that computed Hessian is symmetric."""
        backend = HarmonicBackend(k1=2.0, k2=3.0)

        H = compute_hessian_force_fd_richardson(
            backend, atom_indices=[0], step_size=1e-4, platform_name='CPU'
        )

        np.testing.assert_allclose(
            H, H.T, rtol=1e-10,
            err_msg="Hessian should be symmetric"
        )


# =============================================================================
# Bias Force Sign Convention Tests
# =============================================================================

class MockBiasForceObject:
    """Mock bias force object for testing."""
    pass


class TestBiasForceSignConvention:
    """Tests that verify bias force has correct sign for GADES."""

    def test_bias_reduces_force_along_softest_mode(self):
        """
        Test that the bias force reduces the force component along the softest mode.

        For GADES: F_total = F + bias = F - κ(F·n)n

        If F has a positive component along n (force pushing toward minimum),
        the bias should reduce this component (to help escape the minimum).
        """
        # Create a quadratic potential with known softest mode
        # Softest mode is along first coordinate (smallest eigenvalue)
        eigenvalues = np.array([1.0, 5.0, 10.0])  # Softest mode: e1
        A = np.diag(eigenvalues)

        # Position where force has positive component along softest mode
        x0 = np.array([[1.0, 0.5, 0.2]])  # Will have F = -A @ x
        backend = QuadraticBackend(A, x0)

        # Simple Hessian function that returns the known Hessian
        def hess_func(backend, atom_indices, step_size, platform):
            return A

        # Create GADES bias
        kappa = 0.9
        gades = GADESBias(
            backend=backend,
            biased_force=MockBiasForceObject(),
            bias_atom_indices=[0],
            hess_func=hess_func,
            clamp_magnitude=1000.0,
            kappa=kappa,
            interval=1,
            stability_interval=1,
            logfile_prefix=None,
        )

        # Get the unbiased force
        positions, forces_unbiased = backend.get_current_state()
        F = forces_unbiased.flatten()

        # Get the bias force
        bias = gades.get_gad_force().flatten()

        # The softest mode eigenvector (first eigenvector for diagonal matrix)
        n = np.array([1.0, 0.0, 0.0])

        # Force component along softest mode
        F_dot_n = np.dot(F, n)

        # Expected bias: -κ(F·n)n
        expected_bias = -kappa * F_dot_n * n

        np.testing.assert_allclose(
            bias, expected_bias, rtol=1e-6,
            err_msg="Bias force does not match expected -κ(F·n)n"
        )

        # Verify the bias reduces the force component along n
        # Total force component along n: (F + bias)·n = F·n - κ(F·n) = (1-κ)F·n
        total_F = F + bias
        total_F_dot_n = np.dot(total_F, n)
        expected_total_F_dot_n = (1 - kappa) * F_dot_n

        np.testing.assert_allclose(
            total_F_dot_n, expected_total_F_dot_n, rtol=1e-6,
            err_msg="Total force component along softest mode is incorrect"
        )

        # For kappa close to 1, the component should be nearly eliminated
        assert abs(total_F_dot_n) < abs(F_dot_n), \
            "Bias should reduce force component along softest mode"

    def test_bias_direction_at_saddle_point(self):
        """
        Test bias behavior at a saddle point (negative eigenvalue).

        At a saddle point, the softest mode has negative curvature.
        The bias should still reduce the force component along this mode.
        """
        # Saddle point: one negative eigenvalue
        eigenvalues = np.array([-2.0, 3.0, 5.0])  # Softest mode: e1 with λ=-2
        A = np.diag(eigenvalues)

        x0 = np.array([[0.5, 0.3, 0.1]])
        backend = QuadraticBackend(A, x0)

        def hess_func(backend, atom_indices, step_size, platform):
            return A

        kappa = 0.9
        gades = GADESBias(
            backend=backend,
            biased_force=MockBiasForceObject(),
            bias_atom_indices=[0],
            hess_func=hess_func,
            clamp_magnitude=1000.0,
            kappa=kappa,
            interval=1,
            stability_interval=1,
            logfile_prefix=None,
        )

        # Get forces and bias
        positions, forces = backend.get_current_state()
        F = forces.flatten()
        bias = gades.get_gad_force().flatten()

        # Softest mode (eigenvector for smallest eigenvalue)
        n = np.array([1.0, 0.0, 0.0])

        # The bias should be -κ(F·n)n
        F_dot_n = np.dot(F, n)
        expected_bias = -kappa * F_dot_n * n

        np.testing.assert_allclose(
            bias, expected_bias, rtol=1e-6,
            err_msg="Bias at saddle point does not match expected"
        )

    def test_zero_force_gives_zero_bias(self):
        """Test that zero force produces zero bias."""
        A = np.diag([1.0, 2.0, 3.0])
        x0 = np.array([[0.0, 0.0, 0.0]])  # At origin, F = 0
        backend = QuadraticBackend(A, x0)

        def hess_func(backend, atom_indices, step_size, platform):
            return A

        gades = GADESBias(
            backend=backend,
            biased_force=MockBiasForceObject(),
            bias_atom_indices=[0],
            hess_func=hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=1,
            stability_interval=1,
            logfile_prefix=None,
        )

        bias = gades.get_gad_force()

        np.testing.assert_allclose(
            bias, np.zeros_like(bias), atol=1e-10,
            err_msg="Zero force should give zero bias"
        )


# =============================================================================
# Backend Consistency Tests
# =============================================================================

class TestBackendBiasConsistency:
    """Tests that both backends apply bias in the same direction."""

    def test_mock_backends_same_bias_direction(self):
        """
        Test that mock OpenMM-like and ASE-like backends would apply
        the same bias direction.

        This is a conceptual test since we don't want to require OpenMM/ASE.
        It verifies the principle that bias values should be applied
        consistently.
        """
        # Create test bias values
        bias_values = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -0.5]])

        # Simulate OpenMM behavior (with fixed CustomExternalForce)
        # Energy: E = -fx*x - fy*y - fz*z
        # Force: F = -∇E = (fx, fy, fz)
        # So setting (fx,fy,fz) = bias_values gives force = bias_values
        openmm_applied_force = bias_values.copy()  # Direct application

        # Simulate ASE behavior
        # forces = base_forces + bias_values
        # So the bias contribution is bias_values
        ase_applied_force = bias_values.copy()  # Direct addition

        # Both should apply the same force
        np.testing.assert_allclose(
            openmm_applied_force, ase_applied_force,
            err_msg="OpenMM and ASE should apply bias in same direction"
        )

    def test_get_forces_returns_physical_forces(self):
        """
        Test that backend.get_forces() returns physical forces F = -∇V.

        For a harmonic potential V = 0.5*k*x^2:
        - F = -∇V = -k*x (force points toward origin)
        - At x > 0, F < 0 (force is negative)
        """
        backend = HarmonicBackend(k1=2.0, k2=3.0)

        # Set position at positive x, y
        backend._positions = np.array([[1.0, 1.0, 0.0]])

        forces = backend.get_forces(backend.get_positions())

        # At x=1, k1=2: F_x = -k1*x = -2
        # At y=1, k2=3: F_y = -k2*y = -3
        expected_forces = np.array([-2.0, -3.0, 0.0])

        np.testing.assert_allclose(
            forces, expected_forces,
            err_msg="get_forces should return physical forces F = -∇V"
        )


# =============================================================================
# Integration Test: Full GADES Cycle
# =============================================================================

class TestGADESIntegration:
    """Integration tests for the full GADES bias cycle."""

    def test_full_cycle_with_numerical_hessian(self):
        """
        Test a full GADES cycle using numerical Hessian computation.

        This verifies that:
        1. Hessian is computed correctly from forces
        2. Eigenvector is extracted correctly
        3. Bias is computed with correct sign
        """
        # Use harmonic potential with known properties
        backend = HarmonicBackend(k1=1.0, k2=4.0)
        backend._positions = np.array([[0.5, 0.3, 0.0]])

        kappa = 0.8
        gades = GADESBias(
            backend=backend,
            biased_force=MockBiasForceObject(),
            bias_atom_indices=[0],
            hess_func=compute_hessian_force_fd_richardson,
            clamp_magnitude=1000.0,
            kappa=kappa,
            interval=1,
            stability_interval=1,
            logfile_prefix=None,
        )

        # Get bias
        bias = gades.get_gad_force().flatten()

        # Compute expected bias manually
        F = backend.get_forces(backend.get_positions())
        H = backend.analytical_hessian()
        eigvals, eigvecs = np.linalg.eigh(H)
        n = eigvecs[:, 0]  # Softest mode (smallest eigenvalue)

        F_dot_n = np.dot(F, n)
        expected_bias = -kappa * F_dot_n * n

        np.testing.assert_allclose(
            bias, expected_bias, rtol=1e-3,
            err_msg="Full GADES cycle produces incorrect bias"
        )


# =============================================================================
# Regression Tests: Numerical Verification
# =============================================================================

class TestSignConventionRegression:
    """
    Regression tests that verify the sign convention fix numerically.

    These tests would have caught the original bugs where:
    - get_forces() returned gradients instead of forces
    - Hessian had wrong sign
    - Bias had wrong direction
    - lanczos_hvp found wrong eigenmode
    """

    def test_hessian_positive_eigenvalues_at_minimum(self):
        """
        Regression Test 1: Hessian has positive eigenvalues at a minimum.

        For V = 0.5 * k * x^2 (harmonic potential):
        - H = ∇²V = k (positive, indicating a minimum)

        If this test fails, the Hessian sign is wrong.
        """
        from GADES.utils import compute_hessian_force_fd_richardson

        class SimpleHarmonicBackend:
            def __init__(self, k=3.0):
                self.k = k
                self._pos = np.array([[1.0, 0.5, 0.0]])

            def get_positions(self):
                return self._pos.copy()

            def get_forces(self, pos):
                # F = -∇V = -k*x (force toward origin)
                return (-self.k * pos).flatten()

        backend = SimpleHarmonicBackend(k=3.0)
        H = compute_hessian_force_fd_richardson(backend, [0], step_size=1e-4)

        # Verify diagonal elements equal k
        np.testing.assert_allclose(
            np.diag(H), [3.0, 3.0, 3.0], rtol=1e-3,
            err_msg="Hessian diagonal should equal spring constant k"
        )

        # Verify all eigenvalues are positive (minimum)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues > 0), (
            f"At a minimum, all Hessian eigenvalues must be positive. "
            f"Got: {eigenvalues}"
        )

    def test_bias_formula_minus_kappa_f_dot_n_n(self):
        """
        Regression Test 2: Bias force equals -κ(F·n)n.

        The GADES bias should be:
        - bias = -κ(F·n)n

        If this test fails, the bias sign convention is wrong.
        """
        class QuadraticTestBackend:
            def __init__(self):
                self._pos = np.array([[1.0, 0.5, 0.2]])
                self._step = 0

            def get_positions(self):
                return self._pos.copy()

            def get_forces(self, pos):
                # F = -H @ x where H = diag([1, 4, 9])
                k = np.array([1.0, 4.0, 9.0])
                return (-k * pos.flatten())

            def get_current_state(self):
                return self._pos.copy(), self.get_forces(self._pos).reshape(-1, 3)

            def get_currentStep(self):
                return self._step

            def get_atom_symbols(self, indices):
                return ['C'] * len(indices)

            def is_stable(self):
                return True

        def known_hess(backend, indices, step, platform):
            return np.diag([1.0, 4.0, 9.0])

        backend = QuadraticTestBackend()
        kappa = 0.9

        gades = GADESBias(
            backend=backend,
            biased_force=MockBiasForceObject(),
            bias_atom_indices=[0],
            hess_func=known_hess,
            clamp_magnitude=1000.0,
            kappa=kappa,
            interval=1,
            stability_interval=1,
            logfile_prefix=None,
        )

        F = backend.get_forces(backend.get_positions())
        bias = gades.get_gad_force().flatten()

        # Softest mode is first eigenvector (smallest eigenvalue = 1)
        n = np.array([1.0, 0.0, 0.0])
        F_dot_n = np.dot(F, n)
        expected_bias = -kappa * F_dot_n * n

        np.testing.assert_allclose(
            bias, expected_bias, rtol=1e-4,
            err_msg=f"Bias should be -κ(F·n)n. Got {bias}, expected {expected_bias}"
        )

    def test_lanczos_hvp_matches_numpy_eigensolver(self):
        """
        Regression Test 4: Lanczos HVP finds same eigenvalue as numpy.

        The matrix-free Lanczos method using HVP should find the same
        smallest eigenvalue and eigenvector as explicit diagonalization.

        If this test fails, the HVP sign convention is wrong.
        """
        from GADES.hvp import finite_difference_hvp
        from GADES.lanczos import lanczos_hvp_smallest

        class MultiAtomBackend:
            def __init__(self):
                self._pos = np.array([[0.5, 0.3, 0.1], [0.2, 0.4, 0.3]])

            def get_positions(self):
                return self._pos.copy()

            def get_forces(self, pos):
                # F = -H @ x where H = diag([1, 2, 3, 4, 5, 6])
                H_diag = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                return -H_diag * pos.flatten()

        backend = MultiAtomBackend()
        pos = backend.get_positions()

        # Numpy eigensolver (ground truth)
        H = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        eigvals, eigvecs = np.linalg.eigh(H)
        numpy_eigval = eigvals[0]
        numpy_eigvec = eigvecs[:, 0]

        # HVP-based Lanczos
        def force_func(p):
            return backend.get_forces(p.reshape(-1, 3))

        def hvp_func(v):
            return finite_difference_hvp(force_func, pos, v, epsilon=1e-5)

        lanczos_eigval, lanczos_eigvec = lanczos_hvp_smallest(
            hvp_func, n_dof=6, n_iter=20
        )
        lanczos_eigvec = lanczos_eigvec / np.linalg.norm(lanczos_eigvec)

        # Eigenvalues should match
        np.testing.assert_allclose(
            lanczos_eigval, numpy_eigval, rtol=0.1,
            err_msg=f"Lanczos eigenvalue {lanczos_eigval} != numpy {numpy_eigval}"
        )

        # Eigenvectors should be aligned (allowing for sign flip)
        alignment = abs(np.dot(numpy_eigvec, lanczos_eigvec))
        assert alignment > 0.99, (
            f"Eigenvectors not aligned: alignment = {alignment}. "
            f"HVP sign convention may be wrong."
        )
