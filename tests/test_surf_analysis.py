"""
Tests for GADES.SurfAnalysis.preprocessing (Step 1).
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from GADES.SurfAnalysis import load_logs, project_hessian, run, Step1Result


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_logs(tmp_path, steps, pos, U, forces, hess):
    """Write synthetic log files in the format GADESBias produces."""
    prefix = str(tmp_path / "test")
    M = pos.shape[1]
    dof = 3 * M

    def _write(fname, header_lines, rows):
        with open(fname, "w") as f:
            for h in header_lines:
                f.write(f"# {h}\n")
            for step, row in zip(steps, rows):
                f.write(f"{step} " + " ".join(f"{v}" for v in row) + "\n")

    _write(f"{prefix}_pos.log",
           ["positions"],
           [pos[i].flatten() for i in range(len(steps))])
    _write(f"{prefix}_epot.log",
           ["energy"],
           [[U[i]] for i in range(len(steps))])
    _write(f"{prefix}_forces.log",
           ["forces"],
           [forces[i].flatten() for i in range(len(steps))])
    _write(f"{prefix}_hess.log",
           ["hessian"],
           [hess[i].flatten() for i in range(len(steps))])
    return prefix


def _make_diagonal_hess(eigenvalues, positions=None):
    """
    Build a Hessian (in the biased-atom space) that has the given eigenvalues.
    Returns a diagonal matrix for simplicity; no rigid-body projection applied.
    """
    dof = len(eigenvalues)
    return np.diag(eigenvalues.astype(float))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def single_atom_data(tmp_path):
    """One snapshot, one biased atom — simplest possible case."""
    N, M = 1, 1
    steps   = np.array([200])
    pos     = np.zeros((N, M, 3))
    U       = np.array([-10.0])
    forces  = np.zeros((N, M, 3));  forces[0, 0, 0] = 1.0
    hess    = np.eye(3)[None]        # (1, 3, 3)
    prefix  = _write_logs(tmp_path, steps, pos, U, forces, hess)
    return prefix, steps, pos, U, forces, hess


@pytest.fixture
def three_atom_snapshot(tmp_path):
    """
    Three biased atoms arranged in an equilateral triangle.
    Hessian is a known diagonal matrix with distinct eigenvalues.
    """
    N, M = 3, 3
    steps = np.array([200, 400, 600])
    pos = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.5, 0.866, 0.0]],
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.5, 0.866, 0.0]],
    ])
    U = np.array([-5.0, -4.5, -4.0])
    forces = np.zeros((N, M, 3));  forces[:, 0, 0] = 0.5
    # Eigenvalues: 3 zeros (rigid body from projection) + mix of pos/neg
    evals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    hess = np.stack([np.diag(evals)] * N)
    prefix = _write_logs(tmp_path, steps, pos, U, forces, hess)
    return prefix, steps, pos, U, forces, hess


# ── load_logs ─────────────────────────────────────────────────────────────────

class TestLoadLogs:
    def test_returns_snapshot_data(self, single_atom_data):
        prefix, steps, pos, U, forces, hess = single_atom_data
        data = load_logs(prefix)
        assert data.steps.shape == (1,)
        assert data.pos.shape   == (1, 1, 3)
        assert data.U.shape     == (1,)
        assert data.forces.shape == (1, 1, 3)
        assert data.hess.shape  == (1, 3, 3)

    def test_step_values_correct(self, three_atom_snapshot):
        prefix, steps, *_ = three_atom_snapshot
        data = load_logs(prefix)
        assert_allclose(data.steps, steps)

    def test_energy_values_correct(self, three_atom_snapshot):
        prefix, steps, pos, U, forces, hess = three_atom_snapshot
        data = load_logs(prefix)
        assert_allclose(data.U, U)

    def test_positions_reshaped_correctly(self, three_atom_snapshot):
        prefix, steps, pos, U, forces, hess = three_atom_snapshot
        data = load_logs(prefix)
        assert_allclose(data.pos, pos)

    def test_forces_reshaped_correctly(self, three_atom_snapshot):
        prefix, steps, pos, U, forces, hess = three_atom_snapshot
        data = load_logs(prefix)
        assert_allclose(data.forces, forces)

    def test_hessian_reshaped_correctly(self, three_atom_snapshot):
        prefix, steps, pos, U, forces, hess = three_atom_snapshot
        data = load_logs(prefix)
        assert_allclose(data.hess, hess)

    def test_step_mismatch_raises(self, tmp_path):
        N, M = 2, 1
        steps_a = np.array([100, 200])
        steps_b = np.array([100, 300])   # different
        pos    = np.zeros((N, M, 3))
        U      = np.zeros(N)
        forces = np.zeros((N, M, 3))
        hess   = np.tile(np.eye(3), (N, 1, 1))

        # Write pos/forces/hess with steps_a, epot with steps_b
        prefix = str(tmp_path / "mismatch")
        def _write(fname, steps, rows):
            with open(fname, "w") as f:
                for s, row in zip(steps, rows):
                    f.write(f"{s} " + " ".join(str(v) for v in row) + "\n")
        _write(f"{prefix}_pos.log",    steps_a, [pos[i].flatten()    for i in range(N)])
        _write(f"{prefix}_epot.log",   steps_b, [[U[i]]               for i in range(N)])
        _write(f"{prefix}_forces.log", steps_a, [forces[i].flatten() for i in range(N)])
        _write(f"{prefix}_hess.log",   steps_a, [hess[i].flatten()   for i in range(N)])

        with pytest.raises(ValueError, match="Step mismatch"):
            load_logs(prefix)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_logs(str(tmp_path / "nonexistent"))


# ── project_hessian ───────────────────────────────────────────────────────────

class TestProjectHessian:
    def test_returns_symmetric_matrix(self, three_atom_snapshot):
        prefix, steps, pos, *_ = three_atom_snapshot
        data = load_logs(prefix)
        H_proj, _ = project_hessian(data.hess[0], data.pos[0])
        assert_allclose(H_proj, H_proj.T, atol=1e-12)

    def test_removes_at_most_six_modes(self, three_atom_snapshot):
        prefix, steps, pos, *_ = three_atom_snapshot
        data = load_logs(prefix)
        _, n_rigid = project_hessian(data.hess[0], data.pos[0])
        assert 0 <= n_rigid <= 6

    def test_rigid_body_modes_become_zero(self):
        """After projection, translations/rotations should have zero eigenvalue."""
        M = 4
        pos = np.random.default_rng(42).standard_normal((M, 3))
        H = np.eye(3 * M)
        H_proj, n_rigid = project_hessian(H, pos)
        w = np.linalg.eigvalsh(H_proj)
        near_zero = np.sum(np.abs(w) < 1e-8)
        assert near_zero >= n_rigid

    def test_n_rigid_six_for_nonlinear_cluster(self):
        """A non-linear set of atoms should yield exactly 6 rigid-body modes."""
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        H = np.eye(12)
        _, n_rigid = project_hessian(H, pos)
        assert n_rigid == 6

    def test_n_rigid_five_for_linear_cluster(self):
        """A linear arrangement of atoms has only 5 rigid-body modes."""
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        H = np.eye(9)
        _, n_rigid = project_hessian(H, pos)
        assert n_rigid == 5

    def test_masses_affect_projector(self):
        """Providing explicit masses should still remove 6 modes for a 3D cluster."""
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        masses = np.array([12., 1., 1., 16.])
        H = np.eye(12)
        _, n_rigid = project_hessian(H, pos, masses=masses)
        assert n_rigid == 6

    def test_projected_hessian_has_correct_shape(self):
        M = 3
        pos = np.random.default_rng(0).standard_normal((M, 3))
        H = np.eye(3 * M)
        H_proj, _ = project_hessian(H, pos)
        assert H_proj.shape == (3 * M, 3 * M)


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def test_returns_step1result(self, three_atom_snapshot):
        prefix, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        assert isinstance(result, Step1Result)

    def test_output_lengths_match_n_snapshots(self, three_atom_snapshot):
        prefix, steps, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        N = len(steps)
        assert len(result.steps)      == N
        assert len(result.U)          == N
        assert len(result.A)          == N
        assert len(result.F_norm)     == N
        assert len(result.f_parallel) == N
        assert len(result.morse_index) == N
        assert len(result.spectral_gap) == N

    def test_steps_preserved(self, three_atom_snapshot):
        prefix, steps, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        assert_allclose(result.steps, steps)

    def test_U_preserved(self, three_atom_snapshot):
        prefix, steps, pos, U, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        assert_allclose(result.U, U)

    def test_F_norm_nonnegative(self, three_atom_snapshot):
        prefix, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        assert np.all(result.F_norm >= 0)

    def test_morse_index_nonnegative_integer(self, three_atom_snapshot):
        prefix, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        assert result.morse_index.dtype == np.int64
        assert np.all(result.morse_index >= 0)

    def test_eigenvalues_sorted_ascending(self, three_atom_snapshot):
        prefix, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        for i in range(len(result.steps)):
            assert np.all(np.diff(result.eigenvalues[i]) >= -1e-10)

    def test_eigenvectors_shape(self, three_atom_snapshot):
        prefix, steps, pos, U, forces, hess = three_atom_snapshot
        M = pos.shape[1]
        dof = 3 * M
        K_requested = 5
        result = run(prefix, temperature=300.0, n_eigvecs=K_requested)
        N = len(steps)
        n_nontrivial = dof - result.n_rigid
        K_actual = min(K_requested, n_nontrivial)
        assert result.eigenvectors.shape == (N, K_actual, dof)

    def test_eigenvectors_normalised(self, three_atom_snapshot):
        prefix, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0, n_eigvecs=3)
        for i in range(len(result.steps)):
            for k in range(result.eigenvectors.shape[1]):
                norm = np.linalg.norm(result.eigenvectors[i, k])
                assert_allclose(norm, 1.0, atol=1e-10)

    def test_quasiharmonic_free_energy_finite_at_minimum(self, tmp_path):
        """At a minimum all eigenvalues positive → A should be finite."""
        N, M = 1, 4
        steps  = np.array([200])
        pos    = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        U      = np.array([0.0])
        forces = np.zeros((N, M, 3))
        evals  = np.linspace(1.0, 9.0, 3 * M)
        hess   = np.stack([np.diag(evals)])
        prefix = _write_logs(tmp_path, steps, pos, U, forces, hess)
        result = run(prefix, temperature=300.0)
        assert np.isfinite(result.A[0])

    def test_quasiharmonic_free_energy_nan_when_no_positive_evals(self, tmp_path):
        """If all non-rigid eigenvalues are negative, A should be NaN."""
        N, M = 1, 4
        steps  = np.array([200])
        pos    = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        U      = np.array([0.0])
        forces = np.zeros((N, M, 3))
        # Build a Hessian whose non-rigid modes are all negative by constructing
        # H = P @ diag(neg_evals) @ P; rigid-body null space is unaffected.
        from GADES.SurfAnalysis.preprocessing import _build_projector
        p = pos[0]
        P, n_rigid = _build_projector(p)
        dof = 3 * M
        neg_evals = np.linspace(-9.0, -1.0, dof)
        H_base = P @ np.diag(neg_evals) @ P
        hess = np.stack([H_base])
        prefix = _write_logs(tmp_path, steps, pos, U, forces, hess)
        result = run(prefix, temperature=300.0)
        assert np.isnan(result.A[0])

    def test_morse_index_consistent_with_eigenvalues(self, three_atom_snapshot):
        """morse_index must equal the count of negative entries in result.eigenvalues."""
        prefix, *_ = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        for i in range(len(result.steps)):
            expected = int(np.sum(result.eigenvalues[i] < 0))
            assert result.morse_index[i] == expected

    def test_spectral_gap_positive_at_minimum(self, tmp_path):
        """At a minimum (all positive eigenvalues), spectral gap should be positive."""
        N, M = 1, 4
        steps  = np.array([200])
        pos    = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        U      = np.array([0.0])
        forces = np.zeros((N, M, 3))
        evals  = np.linspace(1.0, 12.0, 3 * M)
        hess   = np.stack([np.diag(evals)])
        prefix = _write_logs(tmp_path, steps, pos, U, forces, hess)
        result = run(prefix, temperature=300.0)
        assert result.spectral_gap[0] > 0

    def test_n_rigid_returned_correctly(self, tmp_path):
        """n_rigid should equal 6 for a non-linear 3D cluster."""
        N, M = 1, 4
        steps  = np.array([200])
        pos    = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        U      = np.array([0.0])
        forces = np.zeros((N, M, 3))
        hess   = np.stack([np.eye(3 * M)])
        prefix = _write_logs(tmp_path, steps, pos, U, forces, hess)
        result = run(prefix, temperature=300.0)
        assert result.n_rigid == 6

    def test_f_parallel_matches_manual_dot(self, three_atom_snapshot):
        """f_parallel = F · v₁ where v₁ is the softest eigenvector."""
        prefix, steps, pos, U, forces, hess = three_atom_snapshot
        result = run(prefix, temperature=300.0)
        # v₁ is the first row of eigenvectors[i]
        for i in range(len(steps)):
            v1 = result.eigenvectors[i, 0]
            f_flat = forces[i].flatten()
            expected = float(np.dot(f_flat, v1))
            assert_allclose(result.f_parallel[i], expected, atol=1e-10)

    def test_custom_kB_affects_A(self, tmp_path):
        """Changing kB (ASE vs OpenMM units) should change A."""
        N, M = 1, 4
        steps  = np.array([200])
        pos    = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        U      = np.array([1.0])
        forces = np.zeros((N, M, 3))
        evals  = np.linspace(1.0, 12.0, 3 * M)
        hess   = np.stack([np.diag(evals)])
        prefix = _write_logs(tmp_path, steps, pos, U, forces, hess)
        r1 = run(prefix, temperature=300.0, kB=8.314e-3)
        r2 = run(prefix, temperature=300.0, kB=8.617e-5)
        assert not np.isclose(r1.A[0], r2.A[0])
