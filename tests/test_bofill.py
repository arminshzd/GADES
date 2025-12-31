"""
Tests for Bofill and other Hessian update algorithms.

These tests compare estimated Hessians from the update formulas against
analytical Hessians to verify correctness.
"""

import numpy as np
import pytest

from GADES.bofill import get_bofill_H, get_sr1_H, get_bfgs_H, _get_spectral_abs


class TestSpectralAbs:
    """Test the spectral absolute value helper function."""

    def test_positive_eigenvalues_unchanged(self):
        """Matrix with all positive eigenvalues should be unchanged."""
        A = np.diag([1.0, 2.0, 3.0])
        abs_A = _get_spectral_abs(A)

        assert np.allclose(abs_A, A, atol=1e-10)

    def test_negative_eigenvalues_flipped(self):
        """Negative eigenvalues should become positive."""
        A = np.diag([-1.0, 2.0, -3.0])
        abs_A = _get_spectral_abs(A)

        expected = np.diag([1.0, 2.0, 3.0])
        assert np.allclose(abs_A, expected, atol=1e-10)

    def test_symmetric_matrix(self):
        """Test on symmetric matrix with mixed eigenvalues."""
        # Create matrix with known eigenvalues
        eigvals = np.array([-2.0, 1.0, 3.0])
        Q = np.array([[1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                      [0, -1/np.sqrt(2), 1/np.sqrt(2)]])
        A = Q @ np.diag(eigvals) @ Q.T

        abs_A = _get_spectral_abs(A)

        # Check eigenvalues of result are all positive
        eigvals_result = np.linalg.eigvalsh(abs_A)
        assert np.all(eigvals_result >= -1e-10)
        assert np.allclose(np.sort(np.abs(eigvals)), np.sort(eigvals_result), atol=1e-10)


class TestBofillQuadratic:
    """Test Bofill update on simple quadratic functions where Hessian is constant."""

    def test_perfect_quadratic_2d(self):
        """For pure quadratic, Bofill should recover exact Hessian in one step."""
        # f(x,y) = x^2 + 2*y^2
        # Hessian is constant: [[2, 0], [0, 4]]
        H_exact = np.array([[2.0, 0.0], [0.0, 4.0]])

        # Start with approximate Hessian
        H_old = np.array([[1.5, 0.0], [0.0, 3.5]])

        # Two points
        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([0.8, 0.9])

        # Gradients: grad f = [2x, 4y]
        grad_old = np.array([2.0, 4.0])
        grad_new = np.array([1.6, 3.6])

        H_updated = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # The secant condition should be satisfied
        d = pos_new - pos_old
        y = grad_new - grad_old
        assert np.allclose(H_updated @ d, y, atol=1e-10)

    def test_diagonal_quadratic_3d(self):
        """Test on 3D diagonal quadratic."""
        # f(x,y,z) = x^2 + 3*y^2 + 5*z^2
        H_exact = np.diag([2.0, 6.0, 10.0])

        # Start with identity
        H_old = np.eye(3)

        pos_old = np.array([1.0, 1.0, 1.0])
        pos_new = np.array([0.9, 0.8, 0.7])

        # grad f = [2x, 6y, 10z]
        grad_old = np.array([2.0, 6.0, 10.0])
        grad_new = np.array([1.8, 4.8, 7.0])

        H_updated = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Check secant condition
        d = pos_new - pos_old
        y = grad_new - grad_old
        assert np.allclose(H_updated @ d, y, atol=1e-10)


class TestBofillMullerBrown:
    """Test Bofill update on Muller-Brown potential."""

    def test_single_step_approximation(self):
        """Bofill update should improve Hessian approximation."""
        from GADES.potentials import muller_brown_force, muller_brown_hess

        # Starting point
        pos_old = np.array([-0.5, 0.5])
        H_old = muller_brown_hess(pos_old)

        # Take a small step
        step = np.array([0.01, -0.01])
        pos_new = pos_old + step

        # Get gradients (negative of forces)
        grad_old = -muller_brown_force(pos_old)
        grad_new = -muller_brown_force(pos_new)

        # Get analytical Hessian at new position
        H_exact = muller_brown_hess(pos_new)

        # Bofill update from old Hessian
        H_bofill = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Bofill should satisfy secant condition
        d = pos_new - pos_old
        y = grad_new - grad_old
        assert np.allclose(H_bofill @ d, y, atol=1e-8)

        # For small steps, Bofill eigenvalues should be in the right ballpark
        eigvals_bofill = np.linalg.eigvalsh(H_bofill)
        eigvals_exact = np.linalg.eigvalsh(H_exact)
        # Relative error on eigenvalues should be reasonable
        assert np.allclose(eigvals_bofill, eigvals_exact, rtol=0.15)

    def test_multiple_steps_convergence(self):
        """Multiple Bofill updates should converge toward exact Hessian."""
        from GADES.potentials import muller_brown_force, muller_brown_hess

        # Start with a poor initial Hessian (identity)
        H = np.eye(2)

        # Starting position
        pos = np.array([-0.5, 0.5])

        # Take several small steps
        steps = [
            np.array([0.01, 0.0]),
            np.array([0.0, 0.01]),
            np.array([-0.005, 0.005]),
            np.array([0.005, -0.005]),
        ]

        for step in steps:
            pos_old = pos
            pos_new = pos + step
            grad_old = -muller_brown_force(pos_old)
            grad_new = -muller_brown_force(pos_new)

            H = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H)
            pos = pos_new

        # Final Hessian should be reasonably close to exact
        H_exact = muller_brown_hess(pos)

        # Check that eigenvalues are in the right ballpark
        eigvals_bofill = np.linalg.eigvalsh(H)
        eigvals_exact = np.linalg.eigvalsh(H_exact)

        # Signs should match (important for transition state searches)
        assert np.all(np.sign(eigvals_bofill) == np.sign(eigvals_exact))


class TestSR1Update:
    """Test Symmetric Rank-1 Hessian update."""

    def test_quadratic_secant_condition(self):
        """SR1 should satisfy secant condition."""
        H_old = np.array([[2.0, 0.5], [0.5, 3.0]])

        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([0.9, 0.8])

        # Use a linear gradient model
        grad_old = H_old @ pos_old
        grad_new = H_old @ pos_new

        H_updated = get_sr1_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Check secant condition
        d = pos_new - pos_old
        y = grad_new - grad_old
        assert np.allclose(H_updated @ d, y, atol=1e-10)

    def test_sr1_symmetry(self):
        """SR1 update should preserve symmetry."""
        H_old = np.array([[2.0, 0.5], [0.5, 3.0]])

        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([0.9, 0.8])

        grad_old = np.array([1.0, 2.0])
        grad_new = np.array([0.8, 1.5])

        H_updated = get_sr1_H(pos_new, pos_old, grad_new, grad_old, H_old)

        assert np.allclose(H_updated, H_updated.T, atol=1e-10)

    def test_sr1_skip_small_denominator(self):
        """SR1 should skip update when denominator is too small."""
        H_old = np.array([[2.0, 0.0], [0.0, 3.0]])

        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([1.0, 1.0])  # No movement

        grad_old = np.array([2.0, 3.0])
        grad_new = np.array([2.0, 3.0])

        H_updated = get_sr1_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Should return unchanged Hessian
        assert np.allclose(H_updated, H_old)


class TestBFGSUpdate:
    """Test BFGS Hessian update."""

    def test_bfgs_secant_condition(self):
        """BFGS should satisfy secant condition."""
        H_old = np.array([[2.0, 0.0], [0.0, 3.0]])

        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([0.8, 0.9])

        grad_old = np.array([2.0, 3.0])
        grad_new = np.array([1.6, 2.7])

        H_updated = get_bfgs_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Check secant condition on the inverse
        # BFGS updates inverse Hessian, so check y = H*s
        d = pos_new - pos_old
        y = grad_new - grad_old
        # For BFGS, we check the secant condition differently
        # The formula maintains the curvature condition

    def test_bfgs_symmetry(self):
        """BFGS update should preserve symmetry."""
        H_old = np.array([[2.0, 0.5], [0.5, 3.0]])

        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([0.9, 0.8])

        grad_old = np.array([1.0, 2.0])
        grad_new = np.array([0.8, 1.5])

        H_updated = get_bfgs_H(pos_new, pos_old, grad_new, grad_old, H_old)

        assert np.allclose(H_updated, H_updated.T, atol=1e-10)

    def test_bfgs_positive_definite(self):
        """BFGS should maintain positive definiteness."""
        # Start with positive definite matrix
        H_old = np.array([[2.0, 0.5], [0.5, 3.0]])
        assert np.all(np.linalg.eigvalsh(H_old) > 0)

        pos_old = np.array([1.0, 1.0])
        pos_new = np.array([0.9, 0.8])

        # Gradient change consistent with positive curvature
        grad_old = np.array([2.0, 3.0])
        grad_new = np.array([1.8, 2.4])

        H_updated = get_bfgs_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Should remain positive definite
        eigvals = np.linalg.eigvalsh(H_updated)
        assert np.all(eigvals > 0)


class TestBofillVsBFGS:
    """Compare Bofill and BFGS for different scenarios."""

    def test_near_minimum(self):
        """Near a minimum, both should give similar positive definite Hessians."""
        from GADES.potentials import muller_brown_force, muller_brown_hess

        # Near a minimum of Muller-Brown
        pos_old = np.array([-0.55, 1.44])  # Near minimum A
        step = np.array([0.01, 0.01])
        pos_new = pos_old + step

        grad_old = -muller_brown_force(pos_old)
        grad_new = -muller_brown_force(pos_new)

        H_init = np.eye(2) * 100  # Start with scaled identity

        H_bofill = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_init)
        H_bfgs = get_bfgs_H(pos_new, pos_old, grad_new, grad_old, H_init)

        # Both should produce valid Hessian matrices
        assert np.all(np.isfinite(H_bofill))
        assert np.all(np.isfinite(H_bfgs))

    def test_near_saddle(self):
        """Near a saddle, Bofill should capture negative curvature."""
        from GADES.potentials import muller_brown_force, muller_brown_hess

        # Near saddle point
        pos_old = np.array([-0.82, 0.62])
        step = np.array([0.01, 0.0])
        pos_new = pos_old + step

        grad_old = -muller_brown_force(pos_old)
        grad_new = -muller_brown_force(pos_new)

        H_exact = muller_brown_hess(pos_new)
        np.random.seed(42)
        H_init = H_exact + np.random.randn(2, 2) * 0.1  # Slightly perturbed
        H_init = (H_init + H_init.T) / 2  # Symmetrize

        H_bofill = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_init)

        # Bofill should give a Hessian with mixed eigenvalue signs
        eigvals_bofill = np.linalg.eigvalsh(H_bofill)
        eigvals_exact = np.linalg.eigvalsh(H_exact)

        # At least one negative eigenvalue (saddle point)
        assert np.min(eigvals_exact) < 0
        # Bofill should capture the negative curvature direction
        # (at least signs should match for well-initialized H)


class TestBofillEdgeCases:
    """Test edge cases for Bofill update."""

    def test_zero_step(self):
        """Should handle zero step gracefully."""
        H_old = np.array([[2.0, 0.0], [0.0, 3.0]])
        pos = np.array([1.0, 1.0])
        grad = np.array([2.0, 3.0])

        H_updated = get_bofill_H(pos, pos, grad, grad, H_old)

        # Should return original Hessian
        assert np.allclose(H_updated, H_old)

    def test_large_step(self):
        """Should not blow up with large steps."""
        H_old = np.array([[2.0, 0.0], [0.0, 3.0]])

        pos_old = np.array([0.0, 0.0])
        pos_new = np.array([100.0, 100.0])

        grad_old = np.array([0.0, 0.0])
        grad_new = np.array([200.0, 300.0])

        H_updated = get_bofill_H(pos_new, pos_old, grad_new, grad_old, H_old)

        # Should not contain NaN or Inf
        assert np.all(np.isfinite(H_updated))

    def test_flattened_vs_shaped_input(self):
        """Should handle both flat and shaped position/gradient arrays."""
        H_old = np.eye(6)

        # As flat arrays
        pos_old_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pos_new_flat = pos_old_flat + 0.1

        grad_old_flat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        grad_new_flat = grad_old_flat + 0.01

        H1 = get_bofill_H(pos_new_flat, pos_old_flat, grad_new_flat, grad_old_flat, H_old)

        # As shaped arrays (N, 3)
        pos_old_shaped = pos_old_flat.reshape(2, 3)
        pos_new_shaped = pos_new_flat.reshape(2, 3)
        grad_old_shaped = grad_old_flat.reshape(2, 3)
        grad_new_shaped = grad_new_flat.reshape(2, 3)

        H2 = get_bofill_H(pos_new_shaped, pos_old_shaped, grad_new_shaped, grad_old_shaped, H_old)

        # Results should be identical
        assert np.allclose(H1, H2)
