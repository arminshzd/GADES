#!/usr/bin/env python
"""
Benchmark script comparing full Hessian vs matrix-free HVP approaches.

This script measures memory usage, computation time, and accuracy for different
system sizes to demonstrate the scaling advantages of the matrix-free approach.

Usage:
    python benchmarks/hvp_scaling.py
    python benchmarks/hvp_scaling.py --sizes 100 500 1000
    python benchmarks/hvp_scaling.py --lanczos-iters 10 20 30
"""

import argparse
import sys
import time
import tracemalloc
from typing import Callable, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from GADES.hvp import finite_difference_hvp
from GADES.lanczos import lanczos, lanczos_smallest, lanczos_hvp, lanczos_hvp_smallest


def create_test_system(
    n_atoms: int, seed: int = 42, eigenvalue_gap: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """
    Create a test system with a Hessian that has a well-separated smallest eigenvalue.

    The Hessian is constructed with explicit control over the eigenvalue spectrum:
    - One "soft mode" with eigenvalue = 0.1
    - Remaining modes with eigenvalues uniformly in [0.1 * gap, 0.1 * gap + 1.0]

    This mimics molecular systems near transition states where the reaction
    coordinate (softest mode) is well-separated from other vibrational modes.

    Args:
        n_atoms: Number of atoms in the system
        seed: Random seed for reproducibility
        eigenvalue_gap: Ratio between second-smallest and smallest eigenvalue.
            Larger values = better separated soft mode = faster Lanczos convergence.
            Default 10.0 means λ₂/λ₁ = 10.

    Returns:
        positions: Atomic positions, shape (n_atoms, 3)
        hessian: Full Hessian matrix, shape (3*n_atoms, 3*n_atoms)
        force_func: Force function that computes forces at given positions
    """
    rng = np.random.default_rng(seed)
    n_dof = 3 * n_atoms

    # Create positions
    positions = rng.uniform(-1, 1, size=(n_atoms, 3))

    # Define eigenvalue spectrum with controlled gap
    # Smallest eigenvalue is 0.1, others start at 0.1 * eigenvalue_gap
    smallest_eigval = 0.1
    other_eigvals_min = smallest_eigval * eigenvalue_gap
    other_eigvals_max = other_eigvals_min + 1.0

    eigenvalues = np.zeros(n_dof)
    eigenvalues[0] = smallest_eigval
    eigenvalues[1:] = rng.uniform(other_eigvals_min, other_eigvals_max, size=n_dof - 1)

    # Create random orthogonal matrix Q via QR decomposition
    random_matrix = rng.standard_normal((n_dof, n_dof))
    Q, _ = np.linalg.qr(random_matrix)

    # Construct Hessian: H = Q @ diag(eigenvalues) @ Q.T
    hessian = Q @ np.diag(eigenvalues) @ Q.T

    # Ensure symmetry (fix numerical errors)
    hessian = (hessian + hessian.T) / 2

    def force_func(pos: np.ndarray) -> np.ndarray:
        """Force function: F = -H @ x (for quadratic potential V = 0.5 * x^T @ H @ x)"""
        x = pos.reshape(-1)
        # Note: use -(hessian @ x) not -hessian @ x
        # The latter is parsed as (-hessian) @ x, copying the entire matrix!
        return -(hessian @ x)

    return positions, hessian, force_func


def benchmark_full_hessian(
    hessian: np.ndarray, n_trials: int = 3
) -> Tuple[float, float, float, np.ndarray]:
    """
    Benchmark full Hessian eigendecomposition.

    Returns:
        time_avg: Average computation time (seconds)
        time_std: Standard deviation of time
        memory_peak: Peak memory usage (bytes)
        smallest_eigval: Smallest eigenvalue
    """
    times = []
    memory_peaks = []

    for _ in range(n_trials):
        tracemalloc.start()
        start = time.perf_counter()

        eigvals, eigvecs = np.linalg.eigh(hessian)

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        memory_peaks.append(peak)

    smallest_eigval = eigvals[0]
    smallest_eigvec = eigvecs[:, 0]

    return np.mean(times), np.std(times), np.max(memory_peaks), smallest_eigval, smallest_eigvec


def benchmark_lanczos_matrix(
    hessian: np.ndarray, n_iter: int = 20, n_trials: int = 3
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Benchmark matrix-based Lanczos.

    Returns:
        time_avg: Average computation time (seconds)
        time_std: Standard deviation of time
        memory_peak: Peak memory usage (bytes)
        smallest_eigval: Smallest eigenvalue
        smallest_eigvec: Smallest eigenvector
    """
    times = []
    memory_peaks = []
    smallest_eigval = None
    smallest_eigvec = None

    for i in range(n_trials):
        tracemalloc.start()
        start = time.perf_counter()

        eigval, eigvec = lanczos_smallest(hessian, n_iter=n_iter, seed=i)

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        memory_peaks.append(peak)
        smallest_eigval = eigval
        smallest_eigvec = eigvec

    return np.mean(times), np.std(times), np.max(memory_peaks), smallest_eigval, smallest_eigvec


def benchmark_lanczos_hvp(
    force_func: Callable,
    positions: np.ndarray,
    n_dof: int,
    n_iter: int = 20,
    epsilon: float = 1e-5,
    n_trials: int = 3,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Benchmark matrix-free Lanczos with HVP.

    Returns:
        time_avg: Average computation time (seconds)
        time_std: Standard deviation of time
        memory_peak: Peak memory usage (bytes)
        smallest_eigval: Smallest eigenvalue
        smallest_eigvec: Smallest eigenvector
    """
    times = []
    memory_peaks = []
    smallest_eigval = None
    smallest_eigvec = None

    def hvp_func(v: np.ndarray) -> np.ndarray:
        return finite_difference_hvp(force_func, positions, v, epsilon)

    # Warmup call to initialize any lazy allocations in the closure chain
    # This ensures we only measure the actual HVP computation, not Python's
    # closure initialization overhead
    _warmup_v = np.ones(n_dof)
    _ = hvp_func(_warmup_v)
    del _warmup_v

    for i in range(n_trials):
        tracemalloc.start()
        start = time.perf_counter()

        eigval, eigvec = lanczos_hvp_smallest(hvp_func, n_dof, n_iter=n_iter, seed=i)

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        memory_peaks.append(peak)
        smallest_eigval = eigval
        smallest_eigvec = eigvec

    return np.mean(times), np.std(times), np.max(memory_peaks), smallest_eigval, smallest_eigvec


def compute_eigenvector_alignment(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute alignment (absolute cosine) between two eigenvectors."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return abs(np.dot(v1_norm, v2_norm))


def format_bytes(n_bytes: float) -> str:
    """Format bytes as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes:.0f} B"
    elif n_bytes < 1024**2:
        return f"{n_bytes/1024:.1f} KB"
    elif n_bytes < 1024**3:
        return f"{n_bytes/1024**2:.1f} MB"
    else:
        return f"{n_bytes/1024**3:.2f} GB"


def run_benchmark(
    atom_sizes: List[int],
    lanczos_iterations: int = 20,
    n_trials: int = 3,
    verbose: bool = True,
    eigenvalue_gap: float = 10.0,
) -> dict:
    """
    Run full benchmark suite.

    Args:
        atom_sizes: List of atom counts to test
        lanczos_iterations: Number of Lanczos iterations
        n_trials: Number of trials per benchmark
        verbose: Whether to print progress
        eigenvalue_gap: Ratio between second-smallest and smallest eigenvalue

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "atom_sizes": atom_sizes,
        "lanczos_iterations": lanczos_iterations,
        "eigenvalue_gap": eigenvalue_gap,
        "full_hessian": {"times": [], "memory": [], "eigvals": []},
        "lanczos_matrix": {"times": [], "memory": [], "eigvals": [], "alignment": []},
        "lanczos_hvp": {"times": [], "memory": [], "eigvals": [], "alignment": []},
    }

    for n_atoms in atom_sizes:
        n_dof = 3 * n_atoms
        hessian_size = n_dof * n_dof * 8  # float64

        if verbose:
            print(f"\n{'='*60}")
            print(f"System: {n_atoms} atoms ({n_dof} DOF)")
            print(f"Theoretical Hessian size: {format_bytes(hessian_size)}")
            print(f"{'='*60}")

        # Create test system with controlled eigenvalue gap
        positions, hessian, force_func = create_test_system(
            n_atoms, eigenvalue_gap=eigenvalue_gap
        )

        # Benchmark full Hessian (skip for very large systems)
        if n_dof <= 10000:
            if verbose:
                print("\n[1/3] Full Hessian eigendecomposition...")
            time_avg, time_std, memory, eigval, eigvec = benchmark_full_hessian(
                hessian, n_trials
            )
            results["full_hessian"]["times"].append(time_avg)
            results["full_hessian"]["memory"].append(memory)
            results["full_hessian"]["eigvals"].append(eigval)
            reference_eigvec = eigvec

            if verbose:
                print(f"  Time: {time_avg*1000:.2f} ± {time_std*1000:.2f} ms")
                print(f"  Memory: {format_bytes(memory)}")
                print(f"  Smallest eigenvalue: {eigval:.6f}")
        else:
            if verbose:
                print("\n[1/3] Full Hessian skipped (too large)")
            results["full_hessian"]["times"].append(None)
            results["full_hessian"]["memory"].append(None)
            results["full_hessian"]["eigvals"].append(None)
            reference_eigvec = None

        # Benchmark matrix-based Lanczos
        if verbose:
            print(f"\n[2/3] Matrix-based Lanczos ({lanczos_iterations} iterations)...")
        if n_dof <= 10000:
            time_avg, time_std, memory, eigval, eigvec = benchmark_lanczos_matrix(
                hessian, lanczos_iterations, n_trials
            )
            results["lanczos_matrix"]["times"].append(time_avg)
            results["lanczos_matrix"]["memory"].append(memory)
            results["lanczos_matrix"]["eigvals"].append(eigval)

            if reference_eigvec is not None:
                alignment = compute_eigenvector_alignment(reference_eigvec, eigvec)
                results["lanczos_matrix"]["alignment"].append(alignment)
            else:
                results["lanczos_matrix"]["alignment"].append(None)
                reference_eigvec = eigvec  # Use as reference for HVP

            if verbose:
                print(f"  Time: {time_avg*1000:.2f} ± {time_std*1000:.2f} ms")
                print(f"  Memory: {format_bytes(memory)}")
                print(f"  Smallest eigenvalue: {eigval:.6f}")
                if results["lanczos_matrix"]["alignment"][-1] is not None:
                    print(f"  Eigenvector alignment: {results['lanczos_matrix']['alignment'][-1]:.6f}")
        else:
            if verbose:
                print("  Skipped (Hessian too large)")
            results["lanczos_matrix"]["times"].append(None)
            results["lanczos_matrix"]["memory"].append(None)
            results["lanczos_matrix"]["eigvals"].append(None)
            results["lanczos_matrix"]["alignment"].append(None)

        # Benchmark matrix-free Lanczos with HVP
        if verbose:
            print(f"\n[3/3] Matrix-free Lanczos with HVP ({lanczos_iterations} iterations)...")
        time_avg, time_std, memory, eigval, eigvec = benchmark_lanczos_hvp(
            force_func, positions, n_dof, lanczos_iterations, n_trials=n_trials
        )
        results["lanczos_hvp"]["times"].append(time_avg)
        results["lanczos_hvp"]["memory"].append(memory)
        results["lanczos_hvp"]["eigvals"].append(eigval)

        if reference_eigvec is not None:
            alignment = compute_eigenvector_alignment(reference_eigvec, eigvec)
            results["lanczos_hvp"]["alignment"].append(alignment)
        else:
            results["lanczos_hvp"]["alignment"].append(None)

        if verbose:
            print(f"  Time: {time_avg*1000:.2f} ± {time_std*1000:.2f} ms")
            print(f"  Memory: {format_bytes(memory)}")
            print(f"  Smallest eigenvalue: {eigval:.6f}")
            if results["lanczos_hvp"]["alignment"][-1] is not None:
                print(f"  Eigenvector alignment: {results['lanczos_hvp']['alignment'][-1]:.6f}")

    return results


def print_summary(results: dict) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print("\n### Time Comparison (ms) with Speedup vs Full Hessian")
    print("-" * 95)
    print(f"{'Atoms':>8} {'DOF':>8} {'Full Hessian':>14} {'Lanczos':>14} {'Speedup':>10} {'Lanczos HVP':>14} {'Speedup':>10}")
    print("-" * 95)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        full_time = results["full_hessian"]["times"][i]
        lanczos_time = results["lanczos_matrix"]["times"][i]
        hvp_time = results["lanczos_hvp"]["times"][i]

        full_str = f"{full_time*1000:.2f}" if full_time else "N/A"
        lanczos_str = f"{lanczos_time*1000:.2f}" if lanczos_time else "N/A"
        hvp_str = f"{hvp_time*1000:.2f}"

        # Calculate speedup percentages
        if full_time is not None and lanczos_time is not None:
            lanczos_speedup = ((full_time - lanczos_time) / full_time) * 100
            lanczos_speedup_str = f"{lanczos_speedup:+.1f}%"
        else:
            lanczos_speedup_str = "N/A"

        if full_time is not None:
            hvp_speedup = ((full_time - hvp_time) / full_time) * 100
            hvp_speedup_str = f"{hvp_speedup:+.1f}%"
        else:
            hvp_speedup_str = "N/A"

        print(f"{n_atoms:>8} {n_dof:>8} {full_str:>14} {lanczos_str:>14} {lanczos_speedup_str:>10} {hvp_str:>14} {hvp_speedup_str:>10}")

    print("\n### Total Memory Required (Hessian + Algorithm)")
    print("-" * 90)
    print(f"{'Atoms':>8} {'DOF':>8} {'Full Hessian':>15} {'Lanczos':>15} {'Lanczos HVP':>15} {'HVP Savings':>15}")
    print("-" * 90)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        hessian_size = n_dof * n_dof * 8  # Hessian storage in bytes

        full_mem = results["full_hessian"]["memory"][i]
        lanczos_mem = results["lanczos_matrix"]["memory"][i]
        hvp_mem = results["lanczos_hvp"]["memory"][i]

        # Full Hessian and matrix Lanczos REQUIRE the Hessian to be stored
        # HVP Lanczos does NOT require the Hessian
        if full_mem is not None:
            full_total = full_mem + hessian_size
            full_str = format_bytes(full_total)
        else:
            full_str = "N/A"

        if lanczos_mem is not None:
            lanczos_total = lanczos_mem + hessian_size
            lanczos_str = format_bytes(lanczos_total)
        else:
            lanczos_str = "N/A"

        # HVP doesn't need the Hessian - just the algorithm memory
        hvp_str = format_bytes(hvp_mem)

        # Calculate savings vs matrix Lanczos
        if lanczos_mem is not None:
            savings = ((lanczos_total - hvp_mem) / lanczos_total) * 100
            savings_str = f"{savings:.1f}%"
        else:
            savings_str = "N/A"

        print(f"{n_atoms:>8} {n_dof:>8} {full_str:>15} {lanczos_str:>15} {hvp_str:>15} {savings_str:>15}")

    # Theoretical memory comparison
    n_iter = results["lanczos_iterations"]
    print("\n### Theoretical Memory Requirements")
    print("-" * 85)
    print(f"{'Atoms':>8} {'DOF':>8} {'Hessian (N²)':>15} {'Krylov (k×N)':>15} {'HVP: O(N)':>15} {'HVP Savings':>15}")
    print("-" * 85)

    for n_atoms in results["atom_sizes"]:
        n_dof = 3 * n_atoms
        hessian_mem = n_dof * n_dof * 8  # float64
        krylov_mem = n_dof * n_iter * 8  # V matrix in Lanczos
        hvp_mem_theoretical = n_dof * 8 * 5  # ~5 vectors for HVP computation

        savings = ((hessian_mem - hvp_mem_theoretical) / hessian_mem) * 100

        print(f"{n_atoms:>8} {n_dof:>8} {format_bytes(hessian_mem):>15} {format_bytes(krylov_mem):>15} {format_bytes(hvp_mem_theoretical):>15} {savings:>14.1f}%")

    print("\n    Note: In real MD simulations, the force function computes forces directly")
    print("    from atomic positions without ever forming the Hessian matrix, achieving")
    print("    true O(N) memory scaling for the HVP approach.")

    print("\n### Eigenvector Alignment (|v_lanczos · v_true|)")
    print("-" * 70)
    print(f"{'Atoms':>8} {'DOF':>8} {'Lanczos':>15} {'Lanczos HVP':>15} {'Status':>15}")
    print("-" * 70)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        lanczos_align = results["lanczos_matrix"]["alignment"][i]
        hvp_align = results["lanczos_hvp"]["alignment"][i]

        lanczos_str = f"{lanczos_align:.6f}" if lanczos_align is not None else "N/A"
        hvp_str = f"{hvp_align:.6f}" if hvp_align is not None else "N/A"

        # Determine status based on alignment
        min_align = min(
            lanczos_align if lanczos_align is not None else 1.0,
            hvp_align if hvp_align is not None else 1.0
        )
        if min_align >= 0.99:
            status = "✓ Excellent"
        elif min_align >= 0.95:
            status = "⚠ Acceptable"
        else:
            status = "✗ Poor"

        print(f"{n_atoms:>8} {n_dof:>8} {lanczos_str:>15} {hvp_str:>15} {status:>15}")

    print("\n    Alignment should be > 0.95 for reliable GADES results.")
    print("    If alignment is poor, increase --lanczos-iters or check eigenvalue gap.")

    print("\n### Eigenvalue Accuracy (error vs full Hessian)")
    print("-" * 70)
    print(f"{'Atoms':>8} {'DOF':>8} {'True Eigval':>14} {'Lanczos Err':>14} {'HVP Err':>14}")
    print("-" * 70)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        full_eigval = results["full_hessian"]["eigvals"][i]
        lanczos_eigval = results["lanczos_matrix"]["eigvals"][i]
        hvp_eigval = results["lanczos_hvp"]["eigvals"][i]

        if full_eigval is not None:
            full_str = f"{full_eigval:.6f}"

            # Lanczos error
            if lanczos_eigval is not None:
                lanczos_err = abs(lanczos_eigval - full_eigval)
                lanczos_err_str = f"{lanczos_err:.2e}"
            else:
                lanczos_err_str = "N/A"

            # HVP error
            hvp_err = abs(hvp_eigval - full_eigval)
            hvp_err_str = f"{hvp_err:.2e}"
        else:
            full_str = "N/A"
            lanczos_err_str = "N/A"
            hvp_err_str = "N/A"

        print(f"{n_atoms:>8} {n_dof:>8} {full_str:>14} {lanczos_err_str:>14} {hvp_err_str:>14}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark full Hessian vs matrix-free HVP approaches"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="System sizes (number of atoms) to benchmark",
    )
    parser.add_argument(
        "--lanczos-iters",
        type=int,
        default=20,
        help="Number of Lanczos iterations",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per benchmark",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--eigenvalue-gap",
        type=float,
        default=10.0,
        help="Ratio between second-smallest and smallest eigenvalue. "
             "Larger = better separated soft mode = faster Lanczos convergence. "
             "Default: 10.0 (λ₂/λ₁ = 10)",
    )

    args = parser.parse_args()

    print("GADES HVP Scaling Benchmark")
    print(f"System sizes: {args.sizes}")
    print(f"Lanczos iterations: {args.lanczos_iters}")
    print(f"Eigenvalue gap (λ₂/λ₁): {args.eigenvalue_gap}")
    print(f"Trials per benchmark: {args.trials}")

    results = run_benchmark(
        args.sizes,
        lanczos_iterations=args.lanczos_iters,
        n_trials=args.trials,
        verbose=not args.quiet,
        eigenvalue_gap=args.eigenvalue_gap,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
