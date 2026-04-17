#!/usr/bin/env python
"""
Eigensolver Comparison on Muller-Brown Potential

This example demonstrates the three eigensolver options available in GADES:
1. 'numpy' - Full eigendecomposition (default)
2. 'lanczos' - Matrix-based Lanczos iteration
3. 'lanczos_hvp' - Matrix-free Lanczos with Hessian-vector products

The Muller-Brown potential is a 2D test surface commonly used in transition
state theory studies.

Usage:
    python examples/MullerBrown/eigensolver_comparison.py
"""

import numpy as np
import time

from GADES.potentials import muller_brown_potential, muller_brown_force, muller_brown_hess
from GADES.hvp import finite_difference_hvp
from GADES.lanczos import lanczos_smallest, lanczos_hvp_smallest


def compare_eigensolvers_at_point(pos: np.ndarray, n_iter: int = 10) -> None:
    """
    Compare the three eigensolver approaches at a given position.

    Args:
        pos: Position on the Muller-Brown surface, shape (2,)
        n_iter: Number of Lanczos iterations
    """
    print(f"\nPosition: ({pos[0]:.3f}, {pos[1]:.3f})")
    print(f"Potential energy: {muller_brown_potential(pos):.4f}")
    print("-" * 60)

    # Compute the analytical Hessian
    H = muller_brown_hess(pos)

    # Method 1: Full eigendecomposition (numpy)
    start = time.perf_counter()
    eigvals_full, eigvecs_full = np.linalg.eigh(H)
    time_numpy = (time.perf_counter() - start) * 1000

    smallest_eigval_numpy = eigvals_full[0]
    smallest_eigvec_numpy = eigvecs_full[:, 0]

    print(f"\n1. NumPy (full eigh):")
    print(f"   Smallest eigenvalue: {smallest_eigval_numpy:.6f}")
    print(f"   Eigenvector: [{smallest_eigvec_numpy[0]:.4f}, {smallest_eigvec_numpy[1]:.4f}]")
    print(f"   Time: {time_numpy:.3f} ms")

    # Method 2: Matrix-based Lanczos
    start = time.perf_counter()
    eigval_lanczos, eigvec_lanczos = lanczos_smallest(H, n_iter=n_iter, seed=42)
    time_lanczos = (time.perf_counter() - start) * 1000

    # Compute alignment with numpy result
    alignment_lanczos = abs(np.dot(smallest_eigvec_numpy, eigvec_lanczos))

    print(f"\n2. Lanczos (matrix-based, {n_iter} iterations):")
    print(f"   Smallest eigenvalue: {eigval_lanczos:.6f}")
    print(f"   Eigenvector: [{eigvec_lanczos[0]:.4f}, {eigvec_lanczos[1]:.4f}]")
    print(f"   Alignment with numpy: {alignment_lanczos:.6f}")
    print(f"   Eigenvalue error: {abs(eigval_lanczos - smallest_eigval_numpy):.2e}")
    print(f"   Time: {time_lanczos:.3f} ms")

    # Method 3: Matrix-free Lanczos with HVP
    def hvp_func(v):
        return finite_difference_hvp(muller_brown_force, pos, v, epsilon=1e-5)

    start = time.perf_counter()
    eigval_hvp, eigvec_hvp = lanczos_hvp_smallest(hvp_func, n_dof=2, n_iter=n_iter, seed=42)
    time_hvp = (time.perf_counter() - start) * 1000

    # Compute alignment with numpy result
    alignment_hvp = abs(np.dot(smallest_eigvec_numpy, eigvec_hvp))

    print(f"\n3. Lanczos HVP (matrix-free, {n_iter} iterations):")
    print(f"   Smallest eigenvalue: {eigval_hvp:.6f}")
    print(f"   Eigenvector: [{eigvec_hvp[0]:.4f}, {eigvec_hvp[1]:.4f}]")
    print(f"   Alignment with numpy: {alignment_hvp:.6f}")
    print(f"   Eigenvalue error: {abs(eigval_hvp - smallest_eigval_numpy):.2e}")
    print(f"   Time: {time_hvp:.3f} ms")


def main():
    print("=" * 60)
    print("GADES Eigensolver Comparison on Muller-Brown Potential")
    print("=" * 60)

    # Test at different positions on the surface
    test_positions = [
        np.array([-0.5, 1.5]),   # Near a saddle point
        np.array([0.6, 0.0]),    # Near a minimum
        np.array([-0.8, 0.5]),   # Intermediate region
        np.array([0.0, 0.5]),    # Another point
    ]

    for pos in test_positions:
        compare_eigensolvers_at_point(pos, n_iter=10)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
For this 2D system, all methods give essentially identical results.
The key differences appear in larger systems:

- 'numpy': Exact but O(N³) time, O(N²) memory
- 'lanczos': Fast approximate, still needs full Hessian
- 'lanczos_hvp': Fast approximate, O(N) memory (no Hessian stored)

For GADES with large molecular systems (1000+ atoms), use 'lanczos_hvp'.
""")


if __name__ == "__main__":
    main()
