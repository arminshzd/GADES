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


def create_test_system(n_atoms: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """
    Create a test system with a random positive-definite Hessian.

    Returns:
        positions: Atomic positions, shape (n_atoms, 3)
        hessian: Full Hessian matrix, shape (3*n_atoms, 3*n_atoms)
        force_func: Force function that computes forces at given positions
    """
    rng = np.random.default_rng(seed)
    n_dof = 3 * n_atoms

    # Create positions
    positions = rng.uniform(-1, 1, size=(n_atoms, 3))

    # Create a random symmetric positive-definite Hessian
    # Using H = A^T @ A + small_diagonal for positive definiteness
    A = rng.standard_normal((n_dof, n_dof)) / np.sqrt(n_dof)
    hessian = A.T @ A + 0.1 * np.eye(n_dof)

    # Make it strictly positive definite with known smallest eigenvalue
    # Add a small negative shift to create a clear smallest eigenvalue
    eigvals = np.linalg.eigvalsh(hessian)
    min_eigval = eigvals[0]
    # Shift so smallest eigenvalue is around 0.1
    hessian = hessian - (min_eigval - 0.1) * np.eye(n_dof)

    def force_func(pos: np.ndarray) -> np.ndarray:
        """Force function: F = -H @ x (for quadratic potential V = 0.5 * x^T @ H @ x)"""
        x = pos.reshape(-1)
        return -hessian @ x

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
) -> dict:
    """
    Run full benchmark suite.

    Args:
        atom_sizes: List of atom counts to test
        lanczos_iterations: Number of Lanczos iterations
        n_trials: Number of trials per benchmark
        verbose: Whether to print progress

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "atom_sizes": atom_sizes,
        "lanczos_iterations": lanczos_iterations,
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

        # Create test system
        positions, hessian, force_func = create_test_system(n_atoms)

        # Benchmark full Hessian (skip for very large systems)
        if n_dof <= 3000:  # ~72 MB Hessian
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
        if n_dof <= 3000:
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
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print("\n### Time Comparison (ms)")
    print("-" * 70)
    print(f"{'Atoms':>8} {'DOF':>8} {'Full Hessian':>15} {'Lanczos':>15} {'Lanczos HVP':>15}")
    print("-" * 70)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        full_time = results["full_hessian"]["times"][i]
        lanczos_time = results["lanczos_matrix"]["times"][i]
        hvp_time = results["lanczos_hvp"]["times"][i]

        full_str = f"{full_time*1000:.2f}" if full_time else "N/A"
        lanczos_str = f"{lanczos_time*1000:.2f}" if lanczos_time else "N/A"
        hvp_str = f"{hvp_time*1000:.2f}"

        print(f"{n_atoms:>8} {n_dof:>8} {full_str:>15} {lanczos_str:>15} {hvp_str:>15}")

    print("\n### Memory Comparison")
    print("-" * 70)
    print(f"{'Atoms':>8} {'DOF':>8} {'Full Hessian':>15} {'Lanczos':>15} {'Lanczos HVP':>15}")
    print("-" * 70)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        full_mem = results["full_hessian"]["memory"][i]
        lanczos_mem = results["lanczos_matrix"]["memory"][i]
        hvp_mem = results["lanczos_hvp"]["memory"][i]

        full_str = format_bytes(full_mem) if full_mem else "N/A"
        lanczos_str = format_bytes(lanczos_mem) if lanczos_mem else "N/A"
        hvp_str = format_bytes(hvp_mem)

        print(f"{n_atoms:>8} {n_dof:>8} {full_str:>15} {lanczos_str:>15} {hvp_str:>15}")

    print("\n### Accuracy (eigenvalue error vs full Hessian)")
    print("-" * 70)
    print(f"{'Atoms':>8} {'DOF':>8} {'True Eigval':>15} {'Lanczos Err':>15} {'HVP Err':>15}")
    print("-" * 70)

    for i, n_atoms in enumerate(results["atom_sizes"]):
        n_dof = 3 * n_atoms
        full_eigval = results["full_hessian"]["eigvals"][i]
        lanczos_eigval = results["lanczos_matrix"]["eigvals"][i]
        hvp_eigval = results["lanczos_hvp"]["eigvals"][i]

        if full_eigval is not None:
            full_str = f"{full_eigval:.6f}"
            lanczos_err = abs(lanczos_eigval - full_eigval) if lanczos_eigval else None
            hvp_err = abs(hvp_eigval - full_eigval)

            lanczos_str = f"{lanczos_err:.2e}" if lanczos_err else "N/A"
            hvp_str = f"{hvp_err:.2e}"
        else:
            full_str = "N/A"
            lanczos_str = "N/A"
            hvp_str = "N/A"

        print(f"{n_atoms:>8} {n_dof:>8} {full_str:>15} {lanczos_str:>15} {hvp_str:>15}")


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

    args = parser.parse_args()

    print("GADES HVP Scaling Benchmark")
    print(f"System sizes: {args.sizes}")
    print(f"Lanczos iterations: {args.lanczos_iters}")
    print(f"Trials per benchmark: {args.trials}")

    results = run_benchmark(
        args.sizes,
        lanczos_iterations=args.lanczos_iters,
        n_trials=args.trials,
        verbose=not args.quiet,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
