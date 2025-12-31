"""
Tests for Hessian-Vector Product (HVP) computation.
"""

import numpy as np
import pytest

from GADES.hvp import (
    finite_difference_hvp,
    finite_difference_hvp_richardson,
    finite_difference_hvp_forward,
)
from GADES.potentials import muller_brown_force, muller_brown_hess


class TestFiniteDifferenceHVP:
    """Tests for finite_difference_hvp function."""

    def test_hvp_matches_explicit_hessian_x_direction(self):
        """HVP should match H @ v for x-direction vector."""
        pos = np.array([0.0, 0.5])
        v = np.array([1.0, 0.0])

        # Analytical result
        H = muller_brown_hess(pos)
        expected = H @ v

        # Finite difference result
        result = finite_difference_hvp(muller_brown_force, pos, v)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_hvp_matches_explicit_hessian_y_direction(self):
        """HVP should match H @ v for y-direction vector."""
        pos = np.array([-0.5, 1.5])
        v = np.array([0.0, 1.0])

        H = muller_brown_hess(pos)
        expected = H @ v

        result = finite_difference_hvp(muller_brown_force, pos, v)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_hvp_matches_explicit_hessian_diagonal(self):
        """HVP should match H @ v for diagonal vector."""
        pos = np.array([0.5, 0.0])
        v = np.array([1.0, 1.0])

        H = muller_brown_hess(pos)
        expected = H @ v

        result = finite_difference_hvp(muller_brown_force, pos, v)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_hvp_matches_explicit_hessian_random_vector(self):
        """HVP should match H @ v for random vectors."""
        rng = np.random.default_rng(42)
        pos = np.array([-0.2, 0.8])
        v = rng.standard_normal(2)

        H = muller_brown_hess(pos)
        expected = H @ v

        result = finite_difference_hvp(muller_brown_force, pos, v)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_hvp_zero_vector_returns_zero(self):
        """HVP of zero vector should be zero."""
        pos = np.array([0.0, 0.5])
        v = np.array([0.0, 0.0])

        result = finite_difference_hvp(muller_brown_force, pos, v)

        np.testing.assert_array_equal(result, np.zeros(2))

    def test_hvp_linearity_in_vector(self):
        """HVP should be linear in the vector argument."""
        pos = np.array([0.0, 0.5])
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        alpha = 2.5

        hvp_v1 = finite_difference_hvp(muller_brown_force, pos, v1)
        hvp_v2 = finite_difference_hvp(muller_brown_force, pos, v2)
        hvp_combined = finite_difference_hvp(muller_brown_force, pos, alpha * v1 + v2)

        np.testing.assert_allclose(hvp_combined, alpha * hvp_v1 + hvp_v2, rtol=1e-4)

    def test_hvp_different_positions(self):
        """HVP should work at different positions on the surface."""
        positions = [
            np.array([-0.5, 1.5]),   # Near saddle point
            np.array([0.6, 0.0]),     # Near minimum
            np.array([-1.0, 1.0]),    # Another region
        ]

        v = np.array([1.0, 0.5])

        for pos in positions:
            H = muller_brown_hess(pos)
            expected = H @ v
            result = finite_difference_hvp(muller_brown_force, pos, v)
            np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_hvp_shape_mismatch_raises(self):
        """Should raise error if position and vector shapes don't match."""
        pos = np.array([0.0, 0.5])
        v = np.array([1.0, 0.0, 0.0])  # Wrong size

        with pytest.raises(ValueError, match="shape"):
            finite_difference_hvp(muller_brown_force, pos, v)


class TestFiniteDifferenceHVPRichardson:
    """Tests for Richardson extrapolation HVP."""

    def test_richardson_more_accurate(self):
        """Richardson should be more accurate than simple central difference."""
        pos = np.array([0.0, 0.5])
        v = np.array([1.0, 0.3])

        H = muller_brown_hess(pos)
        expected = H @ v

        # Use same base epsilon for fair comparison
        epsilon = 1e-3  # Relatively large to show difference

        result_simple = finite_difference_hvp(muller_brown_force, pos, v, epsilon)
        result_richardson = finite_difference_hvp_richardson(
            muller_brown_force, pos, v, epsilon
        )

        error_simple = np.linalg.norm(result_simple - expected)
        error_richardson = np.linalg.norm(result_richardson - expected)

        # Richardson should have smaller error
        assert error_richardson < error_simple

    def test_richardson_matches_explicit_hessian(self):
        """Richardson HVP should match H @ v."""
        pos = np.array([-0.5, 1.5])
        v = np.array([0.7, -0.3])

        H = muller_brown_hess(pos)
        expected = H @ v

        result = finite_difference_hvp_richardson(muller_brown_force, pos, v)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_richardson_high_accuracy(self):
        """Richardson should achieve high accuracy."""
        rng = np.random.default_rng(123)
        pos = rng.uniform(-1, 1, size=2)
        v = rng.standard_normal(2)

        H = muller_brown_hess(pos)
        expected = H @ v

        result = finite_difference_hvp_richardson(muller_brown_force, pos, v)

        # Should be very accurate
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-10)


class TestFiniteDifferenceHVPForward:
    """Tests for forward difference HVP."""

    def test_forward_hvp_approximate(self):
        """Forward HVP should approximately match H @ v."""
        pos = np.array([0.0, 0.5])
        v = np.array([1.0, 0.0])

        H = muller_brown_hess(pos)
        expected = H @ v

        force_at_pos = muller_brown_force(pos)
        result = finite_difference_hvp_forward(
            muller_brown_force, pos, v, force_at_pos
        )

        # Forward difference is less accurate, use larger tolerance
        np.testing.assert_allclose(result, expected, rtol=1e-2)

    def test_forward_hvp_reuses_force(self):
        """Forward HVP should give reasonable results reusing force."""
        pos = np.array([-0.5, 1.5])
        v = np.array([0.5, 0.5])

        H = muller_brown_hess(pos)
        expected = H @ v

        force_at_pos = muller_brown_force(pos)
        result = finite_difference_hvp_forward(
            muller_brown_force, pos, v, force_at_pos, epsilon=1e-6
        )

        # With small epsilon, should be reasonably accurate
        np.testing.assert_allclose(result, expected, rtol=1e-3)


class TestHVPEpsilonSensitivity:
    """Tests for epsilon sensitivity."""

    def test_epsilon_too_large(self):
        """Large epsilon should have more truncation error."""
        pos = np.array([0.0, 0.5])
        v = np.array([1.0, 0.0])

        H = muller_brown_hess(pos)
        expected = H @ v

        result_small = finite_difference_hvp(muller_brown_force, pos, v, epsilon=1e-5)
        result_large = finite_difference_hvp(muller_brown_force, pos, v, epsilon=1e-2)

        error_small = np.linalg.norm(result_small - expected)
        error_large = np.linalg.norm(result_large - expected)

        # Small epsilon should be more accurate
        assert error_small < error_large

    def test_optimal_epsilon_range(self):
        """Epsilon in optimal range should give good accuracy."""
        pos = np.array([-0.3, 0.8])
        v = np.array([0.6, -0.4])

        H = muller_brown_hess(pos)
        expected = H @ v

        # Test various epsilon values in reasonable range
        for epsilon in [1e-4, 1e-5, 1e-6]:
            result = finite_difference_hvp(muller_brown_force, pos, v, epsilon)
            np.testing.assert_allclose(result, expected, rtol=1e-3)


class TestHVPHigherDimensions:
    """Tests for HVP with higher-dimensional systems."""

    def test_hvp_3d_positions(self):
        """HVP should work with 3D position arrays (N, 3 format)."""
        # Create a simple quadratic potential in 3D
        # V(x) = 0.5 * x^T @ A @ x, where A is diagonal
        diagonal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        A = np.diag(diagonal)

        def quadratic_force(pos):
            """Force for quadratic potential: F = -∇V = -A @ x"""
            return -A @ pos.reshape(-1)

        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 atoms, 3D
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # For quadratic potential, H = A, so H @ v = A @ v
        expected = A @ v.reshape(-1)

        result = finite_difference_hvp(quadratic_force, pos, v)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_hvp_preserves_shape_info(self):
        """HVP should return flattened result regardless of input shape."""
        diagonal = np.array([1.0, 2.0, 3.0])
        A = np.diag(diagonal)

        def quadratic_force(pos):
            return -A @ pos.reshape(-1)

        pos_2d = np.array([[1.0, 2.0, 3.0]])  # Shape (1, 3)
        pos_1d = np.array([1.0, 2.0, 3.0])    # Shape (3,)
        v = np.array([0.1, 0.2, 0.3])

        result_2d = finite_difference_hvp(quadratic_force, pos_2d, v)
        result_1d = finite_difference_hvp(quadratic_force, pos_1d, v)

        # Both should give same flattened result
        np.testing.assert_allclose(result_2d, result_1d)
        assert result_2d.shape == (3,)
