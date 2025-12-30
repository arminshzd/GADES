"""
Tests for GADES utility functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from GADES.utils import clamp_force_magnitudes


class TestClampForceMagnitudes:
    """Tests for the clamp_force_magnitudes function."""

    def test_no_clamping_needed(self):
        """Forces below max_force should remain unchanged."""
        forces = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 3.0])  # Two vectors
        max_force = 10.0
        result = clamp_force_magnitudes(forces, max_force)
        assert_array_almost_equal(result, forces)

    def test_single_vector_clamping(self):
        """A single vector exceeding max_force should be scaled down."""
        # Vector [3, 4, 0] has magnitude 5
        forces = np.array([3.0, 4.0, 0.0])
        max_force = 5.0
        result = clamp_force_magnitudes(forces, max_force)
        # Should remain unchanged since magnitude == max_force
        assert_array_almost_equal(result, forces)

        # Now test with max_force < magnitude
        max_force = 2.5
        result = clamp_force_magnitudes(forces, max_force)
        # Magnitude should now be 2.5, direction preserved
        result_magnitude = np.linalg.norm(result)
        assert_array_almost_equal(result_magnitude, max_force)
        # Direction should be preserved
        original_direction = forces / np.linalg.norm(forces)
        result_direction = result / np.linalg.norm(result)
        assert_array_almost_equal(original_direction, result_direction)

    def test_multiple_vectors_mixed(self):
        """Test with multiple vectors, some needing clamping."""
        # Vector 1: [3, 4, 0] magnitude 5 - needs clamping
        # Vector 2: [1, 0, 0] magnitude 1 - no clamping
        forces = np.array([3.0, 4.0, 0.0, 1.0, 0.0, 0.0])
        max_force = 2.5
        result = clamp_force_magnitudes(forces, max_force)

        # First vector should be clamped
        vec1 = result[:3]
        assert_array_almost_equal(np.linalg.norm(vec1), max_force)

        # Second vector should be unchanged
        vec2 = result[3:6]
        assert_array_almost_equal(vec2, [1.0, 0.0, 0.0])

    def test_zero_vector(self):
        """Zero vectors should remain zero."""
        forces = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
        max_force = 2.5
        result = clamp_force_magnitudes(forces, max_force)

        # First vector should remain zero
        assert_array_almost_equal(result[:3], [0.0, 0.0, 0.0])

    def test_large_array(self):
        """Test with a larger array of vectors."""
        n_vectors = 100
        forces = np.random.randn(n_vectors * 3) * 10  # Random forces
        max_force = 5.0
        result = clamp_force_magnitudes(forces, max_force)

        # Reshape to check each vector
        result_vectors = result.reshape(-1, 3)
        for vec in result_vectors:
            magnitude = np.linalg.norm(vec)
            assert magnitude <= max_force + 1e-10  # Small tolerance for floating point

    def test_preserves_direction(self):
        """Clamping should preserve the direction of the force vector."""
        forces = np.array([6.0, 8.0, 0.0])  # Magnitude 10
        max_force = 5.0
        result = clamp_force_magnitudes(forces, max_force)

        # Check direction is preserved
        original_normalized = forces / np.linalg.norm(forces)
        result_normalized = result / np.linalg.norm(result)
        assert_array_almost_equal(original_normalized, result_normalized)

    def test_exact_max_force(self):
        """Vector with exactly max_force magnitude should be unchanged."""
        forces = np.array([3.0, 4.0, 0.0])  # Magnitude 5
        max_force = 5.0
        result = clamp_force_magnitudes(forces, max_force)
        assert_array_almost_equal(result, forces)

    def test_negative_components(self):
        """Test with negative force components."""
        forces = np.array([-3.0, -4.0, 0.0])  # Magnitude 5
        max_force = 2.5
        result = clamp_force_magnitudes(forces, max_force)

        # Magnitude should be clamped
        assert_array_almost_equal(np.linalg.norm(result), max_force)
        # Direction should be preserved (negative components stay negative)
        assert result[0] < 0
        assert result[1] < 0

    def test_input_shape_preserved(self):
        """Output should have same shape as input."""
        forces = np.random.randn(30)
        max_force = 1.0
        result = clamp_force_magnitudes(forces, max_force)
        assert result.shape == forces.shape

    def test_3d_vectors_only(self):
        """Function assumes 3D vectors; input length must be multiple of 3."""
        forces = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Two 3D vectors
        max_force = 10.0
        result = clamp_force_magnitudes(forces, max_force)
        assert len(result) == 6


class TestMullerBrownPotential:
    """Tests for Muller-Brown potential functions (if they need testing)."""

    def test_muller_brown_potential_shape(self):
        """Test that Muller-Brown potential returns correct shape."""
        from GADES.utils import muller_brown_potential
        import jax.numpy as jnp

        X = jnp.array([[0.0, 0.0], [1.0, 1.0], [-0.5, 0.5]])
        result = muller_brown_potential(X)
        assert result.shape == (3,)

    def test_muller_brown_force_shape(self):
        """Test that Muller-Brown force returns correct shape."""
        from GADES.utils import muller_brown_force
        import jax.numpy as jnp

        X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = muller_brown_force(X)
        assert result.shape == (2, 2)

    def test_muller_brown_hessian_shape(self):
        """Test that Muller-Brown Hessian returns correct shape."""
        from GADES.utils import muller_brown_hess
        import jax.numpy as jnp

        X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = muller_brown_hess(X)
        assert result.shape == (2, 2, 2)

    def test_muller_brown_known_minimum(self):
        """Test potential value at known approximate minimum."""
        from GADES.utils import muller_brown_potential_base
        import jax.numpy as jnp

        # One of the minima is approximately at (-0.558, 1.442)
        x_min = jnp.array([-0.558, 1.442])
        potential = muller_brown_potential_base(x_min)
        # The minimum should have negative potential (deep well)
        assert potential < 0
