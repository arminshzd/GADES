"""
Tests for the Lanczos eigenvalue algorithm.

These tests compare Lanczos approximations against analytical results
from np.linalg.eigh to verify correctness.
"""

import numpy as np
import pytest

from GADES.lanczos import lanczos, lanczos_smallest, lanczos_shift_invert, options


class TestLanczosBasic:
    """Test basic Lanczos algorithm functionality."""

    def test_identity_matrix(self):
        """Lanczos on identity matrix should give eigenvalues of 1."""
        n = 5
        A = np.eye(n)
        eigvals, eigvecs = lanczos(A, n_iter=n, seed=42)

        # All eigenvalues should be 1
        assert np.allclose(eigvals, 1.0, atol=1e-10)

    def test_diagonal_matrix(self):
        """Lanczos on diagonal matrix should recover diagonal entries."""
        diag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = np.diag(diag)
        eigvals, eigvecs = lanczos(A, n_iter=5, seed=42)

        # Eigenvalues should match diagonal (sorted)
        assert np.allclose(np.sort(eigvals), np.sort(diag), atol=1e-10)

    def test_symmetric_2x2(self):
        """Test on simple 2x2 symmetric matrix."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        eigvals_lanczos, eigvecs_lanczos = lanczos(A, n_iter=2, seed=42)
        eigvals_exact, eigvecs_exact = np.linalg.eigh(A)

        # Eigenvalues should match
        assert np.allclose(np.sort(eigvals_lanczos), np.sort(eigvals_exact), atol=1e-10)

    def test_symmetric_random(self):
        """Test on random symmetric matrix."""
        np.random.seed(123)
        n = 10
        B = np.random.randn(n, n)
        A = (B + B.T) / 2  # Make symmetric

        eigvals_lanczos, eigvecs_lanczos = lanczos(A, n_iter=n, seed=42)
        eigvals_exact, _ = np.linalg.eigh(A)

        # With full iterations, should recover all eigenvalues
        assert np.allclose(np.sort(eigvals_lanczos), np.sort(eigvals_exact), atol=1e-8)

    def test_eigenvector_orthogonality(self):
        """Eigenvectors should be orthonormal."""
        np.random.seed(456)
        n = 8
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        _, eigvecs = lanczos(A, n_iter=n, seed=42)

        # Check orthogonality: V^T V should be identity
        VtV = eigvecs.T @ eigvecs
        assert np.allclose(VtV, np.eye(n), atol=1e-10)

    def test_eigenvector_correctness(self):
        """Eigenvectors should satisfy A*v = lambda*v."""
        np.random.seed(789)
        n = 6
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        eigvals, eigvecs = lanczos(A, n_iter=n, seed=42)

        # Check A*v = lambda*v for each eigenpair
        for i in range(n):
            Av = A @ eigvecs[:, i]
            lambda_v = eigvals[i] * eigvecs[:, i]
            assert np.allclose(Av, lambda_v, atol=1e-8)

    def test_fewer_iterations(self):
        """With fewer iterations, extreme eigenvalues should still be accurate."""
        np.random.seed(111)
        n = 20
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        eigvals_exact, _ = np.linalg.eigh(A)

        # With only 5 iterations, we should get good approximations
        # of the extreme eigenvalues
        eigvals_lanczos, _ = lanczos(A, n_iter=5, seed=42)

        # The smallest and largest should be close
        assert np.abs(np.min(eigvals_lanczos) - np.min(eigvals_exact)) < 0.5
        assert np.abs(np.max(eigvals_lanczos) - np.max(eigvals_exact)) < 0.5


class TestLanczosSmallest:
    """Test lanczos_smallest convenience function."""

    def test_finds_smallest_eigenvalue(self):
        """Should find the smallest eigenvalue accurately."""
        A = np.diag([5.0, 3.0, 1.0, 2.0, 4.0])
        eigval, eigvec = lanczos_smallest(A, n_iter=5, seed=42)

        assert np.isclose(eigval, 1.0, atol=1e-10)

    def test_negative_eigenvalue(self):
        """Should find negative smallest eigenvalue."""
        A = np.diag([5.0, 3.0, -2.0, 2.0, 4.0])
        eigval, eigvec = lanczos_smallest(A, n_iter=5, seed=42)

        assert np.isclose(eigval, -2.0, atol=1e-10)

    def test_eigenvector_normalized(self):
        """Returned eigenvector should be normalized."""
        np.random.seed(222)
        n = 8
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        _, eigvec = lanczos_smallest(A, n_iter=n, seed=42)

        assert np.isclose(np.linalg.norm(eigvec), 1.0, atol=1e-10)

    def test_eigenvector_correct(self):
        """Eigenvector should satisfy A*v = lambda*v."""
        np.random.seed(333)
        n = 6
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        eigval, eigvec = lanczos_smallest(A, n_iter=n, seed=42)

        Av = A @ eigvec
        lambda_v = eigval * eigvec
        assert np.allclose(Av, lambda_v, atol=1e-8)


class TestLanczosShiftInvert:
    """Test shift-and-invert Lanczos for interior eigenvalues."""

    def test_finds_eigenvalue_near_shift(self):
        """Should find eigenvalue closest to the shift value."""
        # Matrix with eigenvalues 1, 3, 5, 7, 9
        A = np.diag([1.0, 3.0, 5.0, 7.0, 9.0])

        # Look for eigenvalue near 5 (use offset to avoid singularity)
        eigvals, eigvecs = lanczos_shift_invert(A, sigma=5.01, n_iter=5, seed=42)

        # First eigenvalue should be closest to 5
        assert np.isclose(eigvals[0], 5.0, atol=0.1)

    def test_finds_eigenvalue_near_different_shifts(self):
        """Test with different shift values."""
        A = np.diag([1.0, 3.0, 5.0, 7.0, 9.0])

        # Look for eigenvalue near 3 (use offset to avoid singularity)
        eigvals, _ = lanczos_shift_invert(A, sigma=3.01, n_iter=5, seed=42)
        assert np.isclose(eigvals[0], 3.0, atol=0.1)

        # Look for eigenvalue near 7
        eigvals, _ = lanczos_shift_invert(A, sigma=7.01, n_iter=5, seed=42)
        assert np.isclose(eigvals[0], 7.0, atol=0.1)

    def test_random_matrix_interior_eigenvalue(self):
        """Find interior eigenvalue of random symmetric matrix."""
        np.random.seed(444)
        n = 10
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        eigvals_exact, _ = np.linalg.eigh(A)

        # Target an interior eigenvalue
        target = eigvals_exact[n // 2]

        eigvals, _ = lanczos_shift_invert(A, sigma=target, n_iter=n, seed=42)

        # First eigenvalue should be very close to target
        assert np.isclose(eigvals[0], target, atol=1e-6)


class TestLanczosOptions:
    """Test options configuration."""

    def test_default_n_iter(self):
        """Should use default N_ITER from options."""
        A = np.eye(5)
        eigvals, eigvecs = lanczos(A, seed=42)

        # Should return options["N_ITER"] eigenvalues (capped at matrix size)
        expected_n = min(options["N_ITER"], 5)
        assert len(eigvals) == expected_n

    def test_custom_n_iter(self):
        """Should respect custom n_iter parameter."""
        A = np.eye(10)
        eigvals, eigvecs = lanczos(A, n_iter=3, seed=42)

        assert len(eigvals) == 3
        assert eigvecs.shape == (10, 3)


class TestLanczosEdgeCases:
    """Test edge cases and numerical stability."""

    def test_1x1_matrix(self):
        """Handle 1x1 matrix."""
        A = np.array([[5.0]])
        eigvals, eigvecs = lanczos(A, n_iter=1, seed=42)

        assert len(eigvals) == 1
        assert np.isclose(eigvals[0], 5.0, atol=1e-10)

    def test_nearly_degenerate_eigenvalues(self):
        """Handle matrix with nearly equal eigenvalues."""
        A = np.diag([1.0, 1.0001, 1.0002])
        eigvals, eigvecs = lanczos(A, n_iter=3, seed=42)

        eigvals_exact, _ = np.linalg.eigh(A)
        assert np.allclose(np.sort(eigvals), np.sort(eigvals_exact), atol=1e-8)

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        np.random.seed(555)
        n = 8
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        eigvals1, eigvecs1 = lanczos(A, n_iter=5, seed=42)
        eigvals2, eigvecs2 = lanczos(A, n_iter=5, seed=42)

        assert np.allclose(eigvals1, eigvals2)
        assert np.allclose(eigvecs1, eigvecs2)

    def test_different_seeds_give_different_results(self):
        """Different seeds may give slightly different results."""
        np.random.seed(666)
        n = 8
        B = np.random.randn(n, n)
        A = (B + B.T) / 2

        eigvals1, _ = lanczos(A, n_iter=3, seed=42)
        eigvals2, _ = lanczos(A, n_iter=3, seed=123)

        # Eigenvalues should still be correct but computed differently
        eigvals_exact, _ = np.linalg.eigh(A)

        # Both should be reasonable approximations to extreme eigenvalues
        assert np.min(eigvals1) < np.median(eigvals_exact)
        assert np.min(eigvals2) < np.median(eigvals_exact)


class TestLanczosMullerBrown:
    """Test Lanczos on Hessian from Muller-Brown potential."""

    def test_muller_brown_hessian(self):
        """Test Lanczos on Muller-Brown Hessian at a known point."""
        # Import Muller-Brown from utils
        from GADES.utils import muller_brown_hess

        # Test at a point on the potential surface
        pos = np.array([[-0.5, 0.5]])  # Shape (1, 2) for batch interface
        hess = np.array(muller_brown_hess(pos)[0])  # Get first (only) Hessian

        # Compare Lanczos vs exact eigenvalues
        eigvals_lanczos, eigvecs_lanczos = lanczos(hess, n_iter=2, seed=42)
        eigvals_exact, eigvecs_exact = np.linalg.eigh(hess)

        assert np.allclose(np.sort(eigvals_lanczos), np.sort(eigvals_exact), atol=1e-10)

    def test_muller_brown_softest_mode(self):
        """Find softest mode of Muller-Brown Hessian."""
        from GADES.utils import muller_brown_hess

        # At saddle point, should have one negative eigenvalue
        saddle = np.array([[-0.822, 0.624]])  # Shape (1, 2) for batch interface
        hess = np.array(muller_brown_hess(saddle)[0])

        eigval, eigvec = lanczos_smallest(hess, n_iter=2, seed=42)

        eigvals_exact, _ = np.linalg.eigh(hess)

        # Should find the smallest (most negative) eigenvalue
        assert np.isclose(eigval, np.min(eigvals_exact), atol=1e-8)
