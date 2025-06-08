import numpy as np
from numpy.linalg import norm, svd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pydiffmap import diffusion_map as dm
from openmm import CustomExternalForce, unit, CMMotionRemover

from utils import clamp_force_magnitudes as fclamp

def getGADESBiasForce(n_particles):
    """
    Function to create the custom force for GADES

    Returns:
        CustomExternalForce: OpenMM custom force class generated for GADES biasing
    """
    force = CustomExternalForce("fx*x+fy*y+fz*z")
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    for i in range(n_particles):
        force.addParticle(i, [0.0, 0.0, 0.0])
    force.setForceGroup(1)
    return force


class GADESForceUpdater(object):
    def __init__(self, biased_force, bias_atom_indices, hess_func, clamp_magnitude, kappa, interval, stability_interval=None, logfile_prefix=None):
        """
        Class to update biased forces in a molecular simulation using Gentlest Ascent Dynamics (GAD).
        
        This updater periodically recalculates the biasing force based on the softest mode
        (smallest eigenvalue direction) from the system's Hessian matrix and applies it to
        specified atoms.

        Parameters
        ----------
        biased_force : openmm.CustomExternalForce or similar
            The OpenMM force object where the biased forces will be applied.
        bias_atom_indices : array-like
            Indices of the atoms to which the biased forces are applied.
        hess_func : callable
            Function that computes the Hessian matrix given the system, positions, 
            selected atom indices, a displacement tolerance, and platform.
        clamp_magnitude : float
            Maximum allowed magnitude for each biased force component.
        kappa : float
            Scaling factor for the biased force.
        interval : int
            Number of simulation steps between force updates.
        """
        self.biased_force = biased_force
        self.bias_atom_indices = bias_atom_indices
        self.hess_func = hess_func
        self.clamp_magnitude = clamp_magnitude
        if interval < 100:
            print("\033[1;33m[GADES| WARNING] Bias update interval must be larger than 100 steps to ensure system stability. Changing the frequency to 110 steps internally...\033[0m")
            self.interval = 110
        else:
            self.interval = interval
        self.kappa = kappa
        self.hess_step_size = 1e-5
        self.check_stability = False
        self.is_biasing = False
        self.s_interval = stability_interval
        
        # post bias update check
        self.next_postbias_check_step = None

        self.logfile_prefix = logfile_prefix
        self._evec_log = None
        self._eval_log = None

        if logfile_prefix is not None:
            self._evec_log = open(f"{logfile_prefix}_evec.log", "w")
            self._eval_log = open(f"{logfile_prefix}_eval.log", "w")
            self._xyz_log  = open(f"{logfile_prefix}_biased_atoms.xyz", "w")

    def set_kappa(self, kappa):
        """
        Update the scaling factor for the biased force.

        Parameters
        ----------
        kappa : float
            New scaling factor.
        """
        self.kappa = kappa
        return None
    
    def set_hess_step_size(self, delta):
        """
        Update the displacement step size used in the Hessian calculation.

        Parameters
        ----------
        delta : float
            New displacement step size.
        """
        self.hess_step_size = delta
        return None
    
    def _is_stable(self, simulation):
        dof = 0
        system = simulation.system
        state = simulation.context.getState(getEnergy=True)
        for i in range(system.getNumParticles()):
            if system.getParticleMass(i) > 0*unit.dalton:
                dof += 3
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if system.getParticleMass(p1) > 0*unit.dalton or system.getParticleMass(p2) > 0*unit.dalton:
                dof -= 1
        if any(type(system.getForce(i)) == CMMotionRemover for i in range(system.getNumForces())):
            dof -= 3
        temperature = (2*state.getKineticEnergy()/(dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)
        target_temperature = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
        if abs(temperature - target_temperature) > 50:
            return False
        return True
    
    def _get_gad_force(self, simulation):
        """
        Compute the Gentlest Ascent Dynamics (GAD) force vector and direction for a molecular system.

        Parameters
        ----------
        sim : openmm.app.Simulation
            The OpenMM simulation object containing the current system state.
        bias_atom_indices : array-like
            Indices of the atoms to which the biasing force is applied.
        hess_func : callable
            Function that computes the Hessian matrix given the system, positions, 
            selected atom indices, a displacement tolerance, and platform.
        clamp_magnitude : float
            Maximum allowed magnitude for the biased force components; forces 
            exceeding this will be clamped.
        kappa : float, optional
            Scaling factor applied to the biased force (default is 0.9).

        Returns
        -------
        forces_b : np.ndarray
            The biased force vector, reshaped to match the system's position shape 
            (typically (N_atoms, 3)).

        Notes
        -----
        This function computes the Hessian of the system, extracts the eigenvector 
        associated with the smallest eigenvalue (softest mode), and constructs a 
        biased force vector aligned with this mode. The biased force is scaled, 
        clamped, and reshaped to match the atomic positions.

        """
        state = simulation.context.getState(getPositions=True, getForces=True)
        platform = simulation.context.getPlatform().getName()
        forces_u = state.getForces(asNumpy=True)[self.bias_atom_indices, :]
        positions = state.getPositions(asNumpy=True)
        hess = self.hess_func(simulation.system, positions, self.bias_atom_indices, self.hess_step_size, platform)
        w, v = np.linalg.eigh(hess)
        w_sorted = w.argsort()
        w_min = w[w_sorted[0]]
        n = v[:, w_sorted[0]]
        n /= np.linalg.norm(n)
        forces_b = -np.dot(n, forces_u.flatten()) * n * self.kappa
        # clamping biased forces so their abs value is never larger than `clamp_magnitude`
        forces_b = fclamp(forces_b, self.clamp_magnitude)

         # Logging
        if self._evec_log is not None:
            self._evec_log.write(" ".join(map(str, n)) + "\n")
            self._evec_log.flush()
        if self._eval_log is not None:
            self._eval_log.write(f"{w_min}\n")
            self._eval_log.flush()
        if self._xyz_log is not None:
            pos_nm = positions[self.bias_atom_indices, :].value_in_unit(unit.nanometer)
            self._xyz_log.write(f"{len(self.bias_atom_indices)}\n")
            self._xyz_log.write(f"Step {simulation.currentStep}\n")
            for coord in pos_nm:
                x, y, z = coord
                self._xyz_log.write(f"C {x:.6f} {y:.6f} {z:.6f}\n")  # Placeholder atom type
            self._xyz_log.flush()
        
        return forces_b.reshape(forces_u.shape)
        
    def describeNextReport(self, simulation):
        """
        Define the interval and required data for the next report.

        Parameters
        ----------
        simulation : openmm.app.Simulation

        Returns
        -------
        tuple
            (steps until next report, pos, vel, force, energy, volume)
        """
        step = simulation.currentStep

        # Compute time to each type of report
        steps_to_check = (
            self.s_interval - step % self.s_interval
            if self.s_interval else np.inf
        )
        steps_to_bias = self.interval - step % self.interval

        if self.next_postbias_check_step is not None:
            steps_to_postbias = max(self.next_postbias_check_step - step, 0)
        else:
            steps_to_postbias = np.inf

        # Choose the next event
        steps = min(steps_to_bias, steps_to_check, steps_to_postbias)

        # Set flags *before* return
        self.is_biasing = (steps == steps_to_bias)
        is_forced_check = (steps == steps_to_postbias)
        is_regular_check = (steps == steps_to_check)

        self.check_stability = is_forced_check or is_regular_check

        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        """
        Apply the computed biased forces at the current simulation step.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The OpenMM simulation object.
        state : openmm.State
            The current simulation state (unused here but required by interface).
        """
        step = simulation.currentStep

        def remove_bias():
            for idx in self.bias_atom_indices:
                self.biased_force.setParticleParameters(idx, idx, (0.0, 0.0, 0.0))

        def apply_bias():
            biased_forces = self._get_gad_force(simulation)
            for i, idx in enumerate(self.bias_atom_indices):
                self.biased_force.setParticleParameters(idx, idx, tuple(biased_forces[i]))

        if self.check_stability:
            is_stable = self._is_stable(simulation)
            if not is_stable:
                print(f"\033[1;31m[GADES | step {step}] System is unstable: Removing bias until next cycle...\033[0m", flush=True)
                remove_bias()
            elif self.is_biasing:
                print(f"\033[1;32m[GADES | step {step}] Updating bias forces...\033[0m", flush=True)
                apply_bias()
                self.next_postbias_check_step = step + 100

            self.biased_force.updateParametersInContext(simulation.context)
            self.check_stability = False
            self.is_biasing = False
            if step == self.next_postbias_check_step:
                self.next_postbias_check_step = None
            return None

        if self.is_biasing:
            print(f"\033[1;32m[GADES | step {step}] Updating bias forces...\033[0m", flush=True)
            apply_bias()
            self.biased_force.updateParametersInContext(simulation.context)
            self.is_biasing = False
            self.next_postbias_check_step = step + 100
            return None

        # If neither flag is True, do nothing
        return None
    
    def __del__(self):
        for f in [self._evec_log, self._eval_log, self._xyz_log]:
            if f is not None:
                f.close()

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2, hidden_dim=256, beta=1.0):
        super().__init__()
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def beta_vae_loss(x, x_recon, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl


class VAERCEstimator:
    def __init__(self, input_dim, latent_dim=2, hidden_dim=256, beta=1.0, epochs=100, batch_size=64, lr=1e-3):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BetaVAE(input_dim, latent_dim, hidden_dim, beta).to(self.device)

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch, in loader:
                optimizer.zero_grad()
                x_recon, mu, logvar = self.model(batch)
                loss = beta_vae_loss(batch, x_recon, mu, logvar, self.beta)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch:3d}/{self.epochs} - Loss: {avg_loss:.6f}")

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(X)
        return mu.cpu().numpy()


class GetRCs:
    def __init__(self, eigenvectors, structures, eigenvalues=None):
        self.V = np.copy(eigenvectors)
        self.X = np.copy(structures)
        self.L = np.copy(eigenvalues) if eigenvalues is not None else None
        self.method = None
        self.model = None

        self._check_dimensions()
        self._normalize_eigenvectors()
        self._align_to_reference()
        if self.L is not None:
            self._append_eigenvalues()

    def _check_dimensions(self):
        assert self.V.ndim == 2, "Eigenvectors must be a 2D array (n_samples, 3N)"
        assert self.X.ndim == 3, "Structures must be a 3D array (n_samples, N, 3)"
        n_samples, n_coords = self.V.shape
        assert self.X.shape[0] == n_samples, "Mismatch in number of frames"
        assert self.X.shape[1] * 3 == n_coords, "Mismatch between structures and eigenvectors"
        if self.L is not None:
            assert self.L.ndim == 1, "Eigenvalues must be a 1D array"
            assert self.L.shape[0] == n_samples, "Mismatch in number of eigenvalues"

    def _normalize_eigenvectors(self):
        norms = norm(self.V, axis=1)
        mask = np.abs(norms - 1.0) > 1e-4
        if np.any(mask):
            self.V[mask] /= norms[mask][:, np.newaxis]

    def _align_to_reference(self):
        ref_coords = self.X[0] - self.X[0].mean(axis=0)
        aligned_V = []

        for x, v in zip(self.X, self.V):
            x_centered = x - x.mean(axis=0)
            R = self._kabsch(x_centered, ref_coords)
            v_reshaped = v.reshape(-1, 3)
            v_aligned = np.dot(v_reshaped, R.T)
            aligned_V.append(v_aligned.flatten())

        self.V_aligned = np.array(aligned_V)

    def _kabsch(self, P, Q):
        C = np.dot(P.T, Q)
        V, S, Wt = svd(C)
        d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
        D = np.diag([1, 1, d])
        R = np.dot(Wt.T, np.dot(D, V.T))
        return R

    def _append_eigenvalues(self):
        scaler = StandardScaler()
        L_scaled = scaler.fit_transform(self.L.reshape(-1, 1))
        self.V_aligned = np.hstack([self.V_aligned, L_scaled])

    def fit(self, method='dmap', n_components=2, **kwargs):
        self.method = method

        if method == 'dmap':
            self.model = dm.DiffusionMap.from_sklearn(n_evecs=n_components, **kwargs)
            self.model.fit(self.V_aligned)

        elif method == 'vae':
            self.model = VAERCEstimator(input_dim=self.V_aligned.shape[1], latent_dim=n_components, **kwargs)
            self.model.fit(self.V_aligned)

        else:
            raise ValueError("Method must be 'dmap' or 'vae'")

    def transform(self):
        if self.model is None:
            raise RuntimeError("Model has not been fit. Call 'fit' first.")

        if self.method == 'dmap':
            return self.model.transform()

        elif self.method == 'vae':
            return self.model.transform(self.V_aligned)

        else:
            raise ValueError("Unknown method. Must be 'dmap' or 'vae'")


