"""
GADES SurfAnalysis — Step 1: Preprocessing and diagnostics.

Reads the surface-analysis log files produced by GADESBias when
``logfile_prefix`` is set, projects out rigid-body modes from each
Hessian snapshot, diagonalises, and assembles the per-snapshot summary
table described in the CV workflow.

Expected log files (all produced by GADESBias):
    <prefix>_pos.log     — biased-atom positions  (step, x1,y1,z1, ...)
    <prefix>_epot.log    — potential energy        (step, U)
    <prefix>_forces.log  — unbiased forces         (step, fx1,fy1,fz1, ...)
    <prefix>_hess.log    — Hessian subspace        (step, H_00, H_01, ...)

Units follow the backend that produced the run:
    OpenMM: positions in nm, energy in kJ/mol, forces in kJ/mol/nm,
            Hessian in kJ/mol/nm².
    ASE:    positions in Å,  energy in eV,       forces in eV/Å,
            Hessian in eV/Å².

All derived quantities (A_i, spectral gap, …) inherit those units.
Pass ``kB`` matching your backend when calling ``run()``.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional, Tuple

import numpy as np


# ── physical constants ────────────────────────────────────────────────────────
_KB_KJ_MOL_K = 8.314462618e-3   # kJ mol⁻¹ K⁻¹  (OpenMM / GADES default)
_KB_EV_K     = 8.617333262e-5   # eV K⁻¹         (ASE default)


# ── result container ─────────────────────────────────────────────────────────

class SnapshotData(NamedTuple):
    """Raw per-snapshot arrays loaded from GADES log files."""
    steps:   np.ndarray   # (N,)       int64  — MD step numbers
    pos:     np.ndarray   # (N, M, 3)  float  — biased-atom positions
    U:       np.ndarray   # (N,)       float  — potential energy
    forces:  np.ndarray   # (N, M, 3)  float  — unbiased forces on biased atoms
    hess:    np.ndarray   # (N, 3M, 3M) float — Hessian of biased-atom subspace


class Step1Result(NamedTuple):
    """
    Per-snapshot summary table produced by Step 1 of the CV workflow.

    All arrays are indexed along the first axis by snapshot (length N).

    Attributes:
        steps        (N,)        MD step number
        U            (N,)        Potential energy
        A            (N,)        Quasiharmonic free energy; NaN if no positive eigenvalues
        F_norm       (N,)        Euclidean norm of unbiased forces on biased atoms
        f_parallel   (N,)        Force component along softest non-rigid mode  (F · v₁)
        eigenvalues  (N, 3M-r)   Sorted eigenvalues after removing r rigid-body modes
        eigenvectors (N, K, 3M)  First K soft-mode eigenvectors (K ≤ n_eigvecs)
        morse_index  (N,)        Number of negative eigenvalues (int)
        spectral_gap (N,)        λ₂ − λ₁ (gap between two softest non-rigid modes)
        n_rigid      int         Number of rigid-body modes removed (same for all snapshots)
    """
    steps:        np.ndarray
    U:            np.ndarray
    A:            np.ndarray
    F_norm:       np.ndarray
    f_parallel:   np.ndarray
    eigenvalues:  np.ndarray
    eigenvectors: np.ndarray
    morse_index:  np.ndarray
    spectral_gap: np.ndarray
    n_rigid:      int


# ── log file parsing ──────────────────────────────────────────────────────────

def _read_log(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a GADES surface-analysis log file.

    Lines beginning with ``#`` are treated as header comments.
    Each data line has the form ``step val1 val2 …``.

    Returns:
        steps: (N,) int64
        data:  (N, D) float64
    """
    steps: list = []
    rows:  list = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            steps.append(int(parts[0]))
            rows.append([float(x) for x in parts[1:]])
    return np.array(steps, dtype=np.int64), np.array(rows, dtype=np.float64)


def load_logs(prefix: str) -> SnapshotData:
    """
    Load all four surface-analysis log files for *prefix*.

    Args:
        prefix: The ``logfile_prefix`` passed to ``GADESBias`` at run time.

    Returns:
        :class:`SnapshotData` with positions, energy, forces, and Hessian
        arrays aligned by snapshot index.

    Raises:
        FileNotFoundError: If any of the four required log files is missing.
        ValueError: If the step arrays across files are not identical.
    """
    steps_pos, pos_flat   = _read_log(f"{prefix}_pos.log")
    steps_U,   U_data     = _read_log(f"{prefix}_epot.log")
    steps_F,   F_flat     = _read_log(f"{prefix}_forces.log")
    steps_H,   H_flat     = _read_log(f"{prefix}_hess.log")

    for s, name in [(steps_U, "epot"), (steps_F, "forces"), (steps_H, "hess")]:
        if not np.array_equal(steps_pos, s):
            raise ValueError(
                f"Step mismatch between pos.log and {name}.log. "
                "All four log files must come from the same run."
            )

    N   = len(steps_pos)
    M   = pos_flat.shape[1] // 3
    dof = 3 * M

    # Reconstruct symmetric Hessian — support both storage formats:
    #   full matrix  : ncols == dof²            (legacy)
    #   upper triangle: ncols == dof*(dof+1)//2 (current, ~50% smaller)
    ncols = H_flat.shape[1]
    if ncols == dof * dof:
        hess = H_flat.reshape(N, dof, dof)
    elif ncols == dof * (dof + 1) // 2:
        hess = np.zeros((N, dof, dof), dtype=H_flat.dtype)
        rows, cols = np.triu_indices(dof)
        hess[:, rows, cols] = H_flat
        hess[:, cols, rows] = H_flat   # fill lower triangle
    else:
        raise ValueError(
            f"Cannot interpret _hess.log: expected {dof*dof} (full) or "
            f"{dof*(dof+1)//2} (upper-triangle) columns per row, got {ncols}."
        )

    return SnapshotData(
        steps  = steps_pos,
        pos    = pos_flat.reshape(N, M, 3),
        U      = U_data[:, 0],
        forces = F_flat.reshape(N, M, 3),
        hess   = hess,
    )


# ── Hessian projection ────────────────────────────────────────────────────────

def _build_projector(
    positions: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Build the (3M × 3M) projector that removes translations and rotations.

    Constructs up to six mass-weighted rigid-body displacement vectors
    (three translations, three infinitesimal rotations about the centre of
    mass), orthonormalises them with Gram-Schmidt, and returns

        P = I − D Dᵀ

    where D has those vectors as columns.  Near-zero vectors (e.g. the
    third rotation for a linear molecule) are silently dropped.

    Args:
        positions: (M, 3) biased-atom Cartesian positions.
        masses:    (M,)   atomic masses.  If ``None``, uniform masses are used.

    Returns:
        P:       (3M, 3M) projector.
        n_rigid: number of rigid-body modes removed (≤ 6).
    """
    M   = len(positions)
    dof = 3 * M

    if masses is None:
        masses = np.ones(M, dtype=float)
    masses   = np.asarray(masses, dtype=float)
    sqrt_m   = np.sqrt(masses)
    total_m  = masses.sum()

    com = (masses[:, None] * positions).sum(axis=0) / total_m
    r   = positions - com                            # (M, 3) centred

    # Six candidate displacement vectors in Cartesian space (length 3M)
    # Translation along x, y, z; rotation about x, y, z through the CoM.
    candidates = np.zeros((6, dof))
    for i in range(M):
        # Translations
        for k in range(3):
            candidates[k, 3 * i + k] = sqrt_m[i]
        rx, ry, rz = r[i]
        # Rotation about x: d = (0, −z, y)
        candidates[3, 3 * i + 1] = -rz * sqrt_m[i]
        candidates[3, 3 * i + 2] =  ry * sqrt_m[i]
        # Rotation about y: d = (z, 0, −x)
        candidates[4, 3 * i + 0] =  rz * sqrt_m[i]
        candidates[4, 3 * i + 2] = -rx * sqrt_m[i]
        # Rotation about z: d = (−y, x, 0)
        candidates[5, 3 * i + 0] = -ry * sqrt_m[i]
        candidates[5, 3 * i + 1] =  rx * sqrt_m[i]

    # Gram-Schmidt — drop vectors that are linearly dependent (norm < 1e-10)
    basis: list = []
    for v in candidates:
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            basis.append(v / norm)

    n_rigid = len(basis)
    if n_rigid == 0:
        return np.eye(dof), 0

    D = np.column_stack(basis)          # (3M, n_rigid)
    P = np.eye(dof) - D @ D.T
    return P, n_rigid


def project_hessian(
    H: np.ndarray,
    positions: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Project translations and rotations out of a Hessian matrix.

    Args:
        H:         (3M, 3M) Hessian of the biased-atom subspace.
        positions: (M, 3)   biased-atom positions (same units as the run).
        masses:    (M,)     atomic masses; ``None`` → uniform.

    Returns:
        H_proj:  (3M, 3M) projected Hessian (symmetric).
        n_rigid: number of rigid-body modes removed.
    """
    P, n_rigid = _build_projector(positions, masses)
    H_proj = P @ H @ P
    # Enforce exact symmetry (floating-point noise)
    H_proj = 0.5 * (H_proj + H_proj.T)
    return H_proj, n_rigid


# ── per-snapshot diagonalisation and scalar computation ───────────────────────

def _parse_hess_line(values: np.ndarray, dof: int) -> np.ndarray:
    """Reconstruct a symmetric (dof×dof) Hessian from a flat log row.

    Accepts both the legacy full-matrix format (dof² values) and the current
    upper-triangle format (dof(dof+1)/2 values).
    """
    n_full  = dof * dof
    n_upper = dof * (dof + 1) // 2
    if len(values) == n_full:
        return values.reshape(dof, dof)
    if len(values) == n_upper:
        H = np.zeros((dof, dof))
        rows, cols = np.triu_indices(dof)
        H[rows, cols] = values
        H[cols, rows] = values
        return H
    raise ValueError(
        f"Cannot interpret Hessian row: expected {n_full} (full) or "
        f"{n_upper} (upper-triangle) values, got {len(values)}."
    )


def _diagonalise(H_proj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors of H_proj, sorted ascending."""
    w, v = np.linalg.eigh(H_proj)
    idx  = np.argsort(w)
    return w[idx], v[:, idx]


def _quasiharmonic_free_energy(U: float, w: np.ndarray, beta: float) -> float:
    """
    A = U + (1/2β) Σ_{k: λ_k > 0} ln(β λ_k / 2π)

    Returns NaN if no positive eigenvalues are present.
    """
    pos = w[w > 0]
    if pos.size == 0:
        return np.nan
    return U + 0.5 / beta * np.sum(np.log(beta * pos / (2 * np.pi)))


# ── public entry point ────────────────────────────────────────────────────────

def run(
    logfile_prefix: str,
    temperature: float,
    masses: Optional[np.ndarray] = None,
    n_eigvecs: int = 20,
    kB: float = _KB_KJ_MOL_K,
) -> Step1Result:
    """
    Execute Step 1 of the GADES CV workflow.

    Loads the four surface-analysis log files, projects rigid-body modes
    out of each Hessian snapshot, diagonalises, and computes the
    per-snapshot scalar invariants required for Steps 2–5.

    Args:
        logfile_prefix: Prefix passed to ``GADESBias`` (e.g. ``"run/gades"``).
        temperature:    Simulation temperature in Kelvin.
        masses:         (M,) atomic masses of the biased atoms in any consistent
                        unit (only ratios matter for the projector).  ``None``
                        uses uniform masses.
        n_eigvecs:      Number of soft-mode eigenvectors to return (K ≤ 3M − n_rigid).
                        Eigenvectors are ordered from softest to stiffest.
        kB:             Boltzmann constant in units matching the log files.
                        Default: 8.314e-3 kJ mol⁻¹ K⁻¹ (OpenMM).
                        For ASE logs use ``kB=8.617e-5`` (eV K⁻¹).

    Returns:
        :class:`Step1Result` with all per-snapshot arrays plus ``n_rigid``.

    Notes:
        - The first call determines ``n_rigid`` from the first snapshot and
          reuses it for all subsequent snapshots (consistent geometry assumed).
        - Snapshots where the Hessian has no positive eigenvalues get ``A = NaN``.
        - ``spectral_gap`` is set to ``NaN`` when fewer than two non-rigid modes
          exist (degenerate or very small subsystem).
        - ``_hess.log`` is streamed line-by-line so only one snapshot's Hessian
          is held in memory at a time.  The other log files (pos, epot, forces)
          are loaded eagerly — they are O(N·M) and negligibly small.
    """
    # ── load small files eagerly ──────────────────────────────────────────────
    steps_pos, pos_flat = _read_log(f"{logfile_prefix}_pos.log")
    steps_U,   U_data   = _read_log(f"{logfile_prefix}_epot.log")
    steps_F,   F_flat   = _read_log(f"{logfile_prefix}_forces.log")

    for steps, name in [(steps_U, "epot"), (steps_F, "forces")]:
        if not np.array_equal(steps_pos, steps):
            raise ValueError(
                f"Step mismatch between pos.log and {name}.log. "
                "All four log files must come from the same run."
            )

    N   = len(steps_pos)
    M   = pos_flat.shape[1] // 3
    dof = 3 * M
    beta = 1.0 / (kB * temperature)

    # Determine n_rigid from the first snapshot
    _, n_rigid = _build_projector(pos_flat[0].reshape(M, 3), masses)
    n_nontrivial = dof - n_rigid
    K = min(n_eigvecs, n_nontrivial)

    if n_nontrivial < 1:
        raise ValueError(
            f"After removing {n_rigid} rigid-body modes, no non-trivial "
            f"modes remain (dof={dof}). Cannot proceed."
        )

    # Allocate output arrays
    A            = np.empty(N)
    F_norm       = np.empty(N)
    f_parallel   = np.empty(N)
    eigenvalues  = np.empty((N, n_nontrivial))
    eigenvectors = np.empty((N, K, dof))
    morse_index  = np.empty(N, dtype=np.int64)
    spectral_gap = np.empty(N)

    # ── stream _hess.log — one snapshot at a time ─────────────────────────────
    hess_path = f"{logfile_prefix}_hess.log"
    i = 0
    with open(hess_path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            if i >= N:
                raise ValueError(
                    f"{hess_path} has more data rows than the {N} steps in pos.log."
                )
            parts = line.split()
            step  = int(parts[0])
            if step != steps_pos[i]:
                raise ValueError(
                    f"Step mismatch at row {i}: pos.log has step {steps_pos[i]}, "
                    f"hess.log has step {step}."
                )

            H = _parse_hess_line(np.array(parts[1:], dtype=np.float64), dof)

            pos_i    = pos_flat[i].reshape(M, 3)
            forces_i = F_flat[i].reshape(M, 3)

            P, _ = _build_projector(pos_i, masses)
            H_proj = P @ H @ P
            H_proj = 0.5 * (H_proj + H_proj.T)
            w, v   = _diagonalise(H_proj)

            # Identify non-rigid modes: eigenvectors in col(P) have ‖P v‖ ≈ 1;
            # rigid-body modes (null(P)) have ‖P v‖ ≈ 0.  Threshold 0.5 cleanly
            # separates them regardless of eigenvalue sign order.
            proj_norms     = np.linalg.norm(P @ v, axis=0)
            non_rigid_mask = proj_norms > 0.5
            w_nt = w[non_rigid_mask]
            v_nt = v[:, non_rigid_mask]

            if len(w_nt) != n_nontrivial:
                warnings.warn(
                    f"Snapshot {i}: expected {n_nontrivial} non-rigid modes, "
                    f"found {len(w_nt)}. Snapshot skipped/trimmed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pad  = n_nontrivial - len(w_nt)
                w_nt = np.pad(w_nt, (0, max(pad, 0)))[:n_nontrivial]
                v_nt = np.pad(v_nt, ((0, 0), (0, max(pad, 0))))[:, :n_nontrivial]

            eigenvalues[i]  = w_nt
            eigenvectors[i] = v_nt[:, :K].T   # (K, dof) — row k = k-th soft mode
            morse_index[i]  = int(np.sum(w_nt < 0))
            spectral_gap[i] = (w_nt[1] - w_nt[0]) if n_nontrivial >= 2 else np.nan
            A[i]            = _quasiharmonic_free_energy(U_data[i, 0], w_nt, beta)

            # f_∥ = F · v₁  (v₁ = softest non-rigid eigenvector)
            f_flat        = forces_i.flatten()
            f_parallel[i] = float(np.dot(f_flat, v_nt[:, 0]))
            F_norm[i]     = float(np.linalg.norm(f_flat))

            i += 1

    if i != N:
        raise ValueError(
            f"{hess_path} has {i} data rows but pos.log has {N}."
        )

    return Step1Result(
        steps        = steps_pos,
        U            = U_data[:, 0],
        A            = A,
        F_norm       = F_norm,
        f_parallel   = f_parallel,
        eigenvalues  = eigenvalues,
        eigenvectors = eigenvectors,
        morse_index  = morse_index,
        spectral_gap = spectral_gap,
        n_rigid      = n_rigid,
    )
