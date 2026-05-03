# Run with `python path_metad.py`

import torch
import numpy as np
from tqdm import tqdm

torch.set_default_dtype(torch.float32)
device = "cuda"  # or "cpu"


# -------------------------
# Path collective variable (PLUMED-style)
# -------------------------
def path_cv(xy: torch.Tensor, path_points: torch.Tensor, lambda_: float = 10.0) -> tuple:
    """
    Compute path collective variables s(x) and z(x) using PLUMED's algebraic method.

    s = Σᵢ i·exp(-λ·R[X - Xᵢ]) / Σᵢ exp(-λ·R[X - Xᵢ])
    z = -(1/λ)·ln[Σᵢ exp(-λ·R[X - Xᵢ])]

    Where R[X - Xᵢ] is the squared distance to reference frame i.

    Reference: https://www.plumed.org/doc-v2.9/user-doc/html/_p_a_t_h.html

    Parameters
    ----------
    xy : tensor (2,) - current position in real space
    path_points : tensor (n_points, 2) - ordered points defining the path
    lambda_ : float - smoothing parameter (higher = sharper localization)

    Returns
    -------
    s : scalar tensor - progress along path (1 to n_points, can be normalized)
    z : scalar tensor - distance from path
    """
    n_points = path_points.shape[0]

    # Squared distances to each reference frame
    diff = xy.unsqueeze(0) - path_points  # (n_points, 2)
    R = torch.sum(diff ** 2, dim=1)       # (n_points,)

    # Log weights (for numerical stability)
    log_weights = -lambda_ * R  # (n_points,)

    # Use log-sum-exp trick to prevent underflow
    max_log_weight = torch.max(log_weights)
    # Shift weights by max for stability
    exp_weights_shifted = torch.exp(log_weights - max_log_weight)  # (n_points,)
    sum_exp_shifted = torch.sum(exp_weights_shifted)

    # Frame indices (1 to n_points)
    indices = torch.arange(1, n_points + 1, dtype=xy.dtype, device=xy.device)

    # s-path: weighted average of frame indices (shift cancels in ratio)
    s = torch.sum(indices * exp_weights_shifted) / sum_exp_shifted

    # z-path: log-sum-exp with stability correction
    # z = -(1/λ) * log(sum_exp) = -(1/λ) * (max_log_weight + log(sum_exp_shifted))
    z = -(1.0 / lambda_) * (max_log_weight + torch.log(sum_exp_shifted))

    return s, z

def path_cv_s_only(xy: torch.Tensor, path_points: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
    """
    Compute only the s-path variable (for backward compatibility).

    Returns s normalized to [0, 1] range.
    """
    s, _ = path_cv(xy, path_points, lambda_)
    n_points = path_points.shape[0]
    # Normalize from [1, n_points] to [0, 1]
    s_normalized = (s - 1.0) / (n_points - 1.0)
    return s_normalized

# -------------------------
# Müller–Brown potential
# -------------------------
A  = torch.tensor([-200.0, -100.0, -170.0,  15.0], device=device)
a  = torch.tensor([  -1.0,   -1.0,   -6.5,   0.7], device=device)
b  = torch.tensor([   0.0,    0.0,   11.0,   0.6], device=device)
c  = torch.tensor([ -10.0,  -10.0,   -6.5,   0.7], device=device)
x0 = torch.tensor([   1.0,    0.0,  -0.5,  -1.0], device=device)
y0 = torch.tensor([   0.0,    0.5,   1.5,   1.0], device=device)

def mb_potential(xy: torch.Tensor) -> torch.Tensor:
    """
    xy: shape (2,), returns scalar tensor V(x,y)
    """
    x, y = xy[0], xy[1]
    dx = x - x0
    dy = y - y0
    expo = a * dx**2 + b * dx * dy + c * dy**2
    return torch.sum(A * torch.exp(expo))

# -------------------------
# Forces: unbiased and bias
# -------------------------
def force_unbiased(xy: torch.Tensor) -> torch.Tensor:
    """
    Unbiased force: F_u = -∇V(x)
    """
    xy = xy.clone().detach().to(device)
    xy.requires_grad_(True)
    V = mb_potential(xy)
    (grad_xy,) = torch.autograd.grad(V, xy, create_graph=False)
    F_u = -grad_xy
    return F_u.detach()

# -------------------------
# Well-Tempered Metadynamics (1D path CV)
# -------------------------
class WellTemperedMetadynamics1D:
    """
    Well-tempered metadynamics in 1D CV space (path variable) with grid-based bias.

    V_bias(s,t) = \Sigma_i w_i * exp(-(s - s_center_i)^2 / (2\sigma^2))

    Well-tempered: w_i = w_0 * exp(-V_bias(s_i, t_i) / (k_B * 
    \Delta T))

    Optionally includes upper wall on z (distance from path) to keep system on-path.
    """

    def __init__(self, sigma, height, biasfactor, kBT, path_points, device,
                 grid_bounds, grid_bins, lambda_path=10.0,
                 z_wall_max=None, z_wall_kappa=100.0):
        """
        sigma: scalar - width of Gaussians in CV space
        height: w_0 - initial Gaussian height
        biasfactor: \gamma - bias factor (ΔT = (\gamma-1)*T)
        kBT: k_B * T - thermal energy
        path_points: (n_points, 2) tensor - points defining the path
        device: torch device
        grid_bounds: (s_min, s_max) - CV space bounds
        grid_bins: int - number of grid points
        lambda_path: float - smoothing parameter for path CV (higher = sharper)
        z_wall_max: float or None - upper wall position for z (None = no wall)
        z_wall_kappa: float - wall force constant (default 100.0)
        """
        self.sigma = torch.tensor(sigma, dtype=torch.get_default_dtype(), device=device)
        self.height = torch.tensor(height, dtype=torch.get_default_dtype(), device=device)
        self.biasfactor = biasfactor
        self.kBT = torch.tensor(kBT, dtype=torch.get_default_dtype(), device=device)
        self.deltaT = (biasfactor - 1.0) * self.kBT
        self.path_points = path_points.to(device)
        self.lambda_path = lambda_path
        self.device = device

        # Z-wall parameters
        self.z_wall_max = z_wall_max
        self.z_wall_kappa = z_wall_kappa

        # Storage for Gaussian hills
        self.hills_centers = []  # list of scalar tensors
        self.hills_heights = []  # list of scalars

        # Grid setup
        self.grid_bounds = grid_bounds
        self.grid_bins = grid_bins

        # Create grid
        self.s_grid = torch.linspace(grid_bounds[0], grid_bounds[1],
                                     grid_bins, device=device)

        # Grid spacing
        self.ds = (grid_bounds[1] - grid_bounds[0]) / (grid_bins - 1)

        # Bias grid: stores V_bias at each grid point
        self.bias_grid = torch.zeros(grid_bins, dtype=torch.get_default_dtype(),
                                     device=device)

    def add_hill(self, s):
        """
        Add a Gaussian hill at current CV position s.
        Height is scaled by well-tempered factor.
        Updates the bias grid.

        s: scalar tensor - current CV position
        """
        # Compute current bias at s
        V_bias_current = self.compute_bias(s)

        # Well-tempered scaling
        w = self.height * torch.exp(-V_bias_current / self.deltaT)

        self.hills_centers.append(s.detach().clone())
        self.hills_heights.append(w.detach().clone())

        # Update grid with new Gaussian
        self._update_grid(s, w)

    def _update_grid(self, s_center, height):
        """
        Add a Gaussian hill to the bias grid.

        s_center: scalar tensor - center of Gaussian in CV space
        height: scalar tensor - height of Gaussian
        """
        # Compute Gaussian on grid
        ds = (self.s_grid - s_center) / self.sigma
        gaussian = height * torch.exp(-0.5 * ds**2)

        # Add to bias grid
        self.bias_grid += gaussian

    def compute_bias(self, s):
        """
        Compute bias potential at CV position s via linear interpolation.

        s: scalar tensor - CV position
        Returns: scalar tensor
        """
        if len(self.hills_centers) == 0:
            return torch.tensor(0.0, dtype=torch.get_default_dtype(), device=self.device)

        # Linear interpolation from grid
        s = torch.clamp(s, self.grid_bounds[0], self.grid_bounds[1])

        # Find grid indices
        i_f = (s - self.grid_bounds[0]) / self.ds

        i0 = torch.floor(i_f).long()
        i1 = torch.clamp(i0 + 1, 0, self.grid_bins - 1)
        i0 = torch.clamp(i0, 0, self.grid_bins - 1)

        # Interpolation weight
        w = i_f - i0.float()

        # Linear interpolation
        V_bias = (1 - w) * self.bias_grid[i0] + w * self.bias_grid[i1]

        return V_bias

    def compute_bias_force(self, xy):
        """
        Compute bias force in xy space: 
        F_bias = -\Delta_x V_bias(s(x)) + F_wall(z(x))

        Includes optional upper wall on z to keep system on-path.

        xy: tensor (2,) - position in real space
        Returns: F_b (2,), s (scalar), V_bias (scalar), z (scalar)
        """
        xy = xy.clone().detach().to(self.device)
        xy.requires_grad_(True)

        # Compute CV (PLUMED-style path CV returns s, z)
        s, z = path_cv(xy, self.path_points, self.lambda_path)

        # Compute bias potential
        V_bias = self.compute_bias(s)

        # Compute z-wall potential if enabled
        V_wall = torch.tensor(0.0, dtype=torch.get_default_dtype(), device=self.device)
        if self.z_wall_max is not None and z > self.z_wall_max:
            dz = z - self.z_wall_max
            V_wall = 0.5 * self.z_wall_kappa * dz ** 2

        # Total bias potential (metadynamics + wall)
        V_total = V_bias + V_wall

        if V_total.requires_grad or len(self.hills_centers) > 0 or V_wall > 0:
            # Compute force via autodiff
            (grad_xy,) = torch.autograd.grad(V_total, xy, create_graph=False, allow_unused=True)
            if grad_xy is None:
                grad_xy = torch.zeros_like(xy)
            F_b = -grad_xy
        else:
            F_b = torch.zeros_like(xy)

        return F_b.detach(), s.detach(), V_bias.detach(), z.detach()

# -------------------------
# BAOAB Langevin integrator
# -------------------------
def baoab_langevin_integrator(positions,
                              velocities,
                              forces_u,
                              forces_b,
                              mass,
                              gamma,
                              dt,
                              kBT,
                              force_function_u,
                              force_function_b,
                              n_steps=1):
    """
    BAOAB Langevin integrator (Leimkuhler & Matthews 2013).
    """
    dim = positions.shape[0]

    c1 = torch.exp(-gamma * dt)
    c3 = torch.sqrt(kBT * (1.0 - c1**2))
    inv_mass = 1.0 / mass
    inv_mass_sqrt = 1.0 / torch.sqrt(mass)

    for _ in range(n_steps):

        # B: first half-step momentum update
        forces = forces_u + forces_b
        velocities = velocities + 0.5 * dt * inv_mass * forces

        # A: half-step position update
        positions = positions + 0.5 * dt * velocities

        # O: thermostat
        random_force = torch.randn(dim, dtype=positions.dtype, device=positions.device)
        velocities = c1 * velocities + c3 * inv_mass_sqrt * random_force

        # A: second half-step position update
        positions = positions + 0.5 * dt * velocities

        # B: second half-step momentum update
        forces_u = force_function_u(positions)
        forces_b = force_function_b(positions)
        forces = forces_u + forces_b
        velocities = velocities + 0.5 * dt * inv_mass * forces

    return positions, velocities, forces_u, forces_b

# -------------------------
# Load checkpoint for restart
# -------------------------
def load_checkpoint(checkpoint_dir, sigma, height, biasfactor, kBT, 
                    path_points, device, grid_bounds, grid_bins, 
                    lambda_path=10.0, z_wall_max=None, z_wall_kappa=100.0,
                    reinit_velocities=False):
    """
    Load checkpoint data for restarting simulation.

    Parameters:
    -----------
    checkpoint_dir: path to checkpoint directory
    sigma, height, biasfactor, kBT: metadynamics parameters
    path_points: tensor defining the path
    device: torch device
    grid_bounds, grid_bins: grid parameters
    lambda_path: smoothing parameter for path CV
    z_wall_max: upper wall position for z (None = no wall)
    z_wall_kappa: wall force constant
    reinit_velocities: if True, reinitialize velocities from Maxwell-Boltzmann

    Returns:
    --------
    metad: WellTemperedMetadynamics1D object with restored state
    xy0: starting position
    v0: starting velocities (or None if reinit_velocities=True)
    start_step: step number to continue from
    prev_traj_data: dictionary with previous trajectory data
    """
    import os

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"Loading checkpoint from {checkpoint_dir}...")

    # Read checkpoint info
    info_file = f"{checkpoint_dir}/checkpoint_info.txt"
    start_step = 0
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            for line in f:
                if line.startswith("step:"):
                    start_step = int(line.split(":")[1].strip())

    # Initialize metadynamics object
    metad = WellTemperedMetadynamics1D(
        sigma=sigma,
        height=height,
        biasfactor=biasfactor,
        kBT=kBT,
        path_points=path_points,
        device=device,
        grid_bounds=grid_bounds,
        grid_bins=grid_bins,
        lambda_path=lambda_path,
        z_wall_max=z_wall_max,
        z_wall_kappa=z_wall_kappa
    )

    # Load bias grid
    bias_grid_file = f"{checkpoint_dir}/bias_grid.npy"
    if os.path.exists(bias_grid_file):
        bias_grid_np = np.load(bias_grid_file)
        metad.bias_grid = torch.from_numpy(bias_grid_np).to(device)
        print(f"  Loaded bias grid")

    # Load hills
    hills_file = f"{checkpoint_dir}/hills.dat"
    if os.path.exists(hills_file):
        hills_data = np.loadtxt(hills_file)
        if hills_data.ndim == 1:  # Single hill
            hills_data = hills_data.reshape(1, -1)

        for hill in hills_data:
            s_center = torch.tensor(hill[0], dtype=torch.get_default_dtype(), device=device)
            h = torch.tensor(hill[1], dtype=torch.get_default_dtype(), device=device)
            metad.hills_centers.append(s_center)
            metad.hills_heights.append(h)
        print(f"  Loaded {len(metad.hills_centers)} hills")

    # Load current position
    xy_file = f"{checkpoint_dir}/current_xy.npy"
    if os.path.exists(xy_file):
        xy0 = np.load(xy_file)
        print(f"  Loaded position: xy = {xy0}")
    else:
        raise FileNotFoundError(f"Current position file not found: {xy_file}")

    # Load or reinitialize velocities
    v_file = f"{checkpoint_dir}/current_v.npy"
    if reinit_velocities or not os.path.exists(v_file):
        v0 = None
        print(f"  Will reinitialize velocities from Maxwell-Boltzmann")
    else:
        v0 = np.load(v_file)
        print(f"  Loaded velocities")

    # Load previous trajectory data
    prev_traj_data = {}
    traj_file = f"{checkpoint_dir}/metad_traj.dat"
    if os.path.exists(traj_file):
        prev_traj = np.loadtxt(traj_file)
        prev_traj_data['t'] = prev_traj[:, 0].tolist()
        prev_traj_data['s'] = prev_traj[:, 1].tolist()
        # Handle both old format (without z) and new format (with z)
        if prev_traj.shape[1] >= 6:
            # New format: t, s, z, V_bias, kBT, n_hills
            prev_traj_data['z'] = prev_traj[:, 2].tolist()
            prev_traj_data['bias'] = prev_traj[:, 3].tolist()
            prev_traj_data['n_hills'] = prev_traj[:, 5].tolist()
        else:
            # Old format: t, s, V_bias, kBT, n_hills
            prev_traj_data['z'] = []
            prev_traj_data['bias'] = prev_traj[:, 2].tolist()
            prev_traj_data['n_hills'] = prev_traj[:, 4].tolist()
        print(f"  Loaded previous trajectory ({len(prev_traj_data['t'])} frames)")

    traj_xy_file = f"{checkpoint_dir}/metad_traj_xy.dat"
    if os.path.exists(traj_xy_file):
        prev_traj_xy = np.loadtxt(traj_xy_file)
        prev_traj_data['xy'] = prev_traj_xy[:, 1:3]

    print(f"Checkpoint loaded successfully. Resuming from step {start_step}")

    return metad, xy0, v0, start_step, prev_traj_data

# -------------------------
# Run metadynamics simulation
# -------------------------
def run_metadynamics(xy0,
                     path_points,
                     sigma,
                     height,
                     biasfactor,
                     deposition_stride,
                     beta,
                     dt,
                     nsteps,
                     grid_bounds,
                     grid_bins=1000,
                     lambda_path=10.0,
                     z_wall_max=None,
                     z_wall_kappa=100.0,
                     mass=1.0,
                     gamma=1.0,
                     stride=10,
                     dump_stride=10000,
                     output_dir="path_metad_data",
                     restart_from=None,
                     reinit_velocities=False):
    """
    Run well-tempered metadynamics simulation with path CV.

    xy0: initial [x, y] (ignored if restart_from is provided)
    path_points: (n_points, 2) array - points defining the path
    sigma: scalar - Gaussian width in CV space
    height: w_0 - initial Gaussian height
    biasfactor: \gamma - bias factor
    deposition_stride: add Gaussian every N steps
    beta: 1/(k_B T)
    dt: time step
    nsteps: number of steps (total, including restart)
    grid_bounds: (s_min, s_max) - CV space bounds (s ranges from 1 to n_path_points)
    grid_bins: int - grid resolution
    lambda_path: float - smoothing parameter for path CV (higher = sharper)
    z_wall_max: float or None - upper wall position for z (None = no wall)
    z_wall_kappa: float - wall force constant (default 100.0)
    mass: scalar mass
    gamma: friction
    stride: sampling stride for trajectory output
    dump_stride: save checkpoint every N steps (0 = no periodic dumps)
    output_dir: directory for output files
    restart_from: path to checkpoint directory to restart from (None = fresh start)
    reinit_velocities: if True, reinitialize velocities when restarting
    """

    import os
    os.makedirs(output_dir, exist_ok=True)

    kBT = 1.0 / beta

    # Convert path_points to tensor
    path_points_t = torch.tensor(path_points, dtype=torch.get_default_dtype(), device=device)

    mass_t = torch.full((2,), float(mass), dtype=torch.get_default_dtype(), device=device)
    gamma_t = torch.tensor(float(gamma), dtype=torch.get_default_dtype(), device=device)
    dt_t = torch.tensor(float(dt), dtype=torch.get_default_dtype(), device=device)
    kBT_t = torch.tensor(float(kBT), dtype=torch.get_default_dtype(), device=device)

    # Initialize or load checkpoint
    if restart_from is not None:
        # Restart from checkpoint
        metad, xy0, v0, start_step, prev_traj_data = load_checkpoint(
            checkpoint_dir=restart_from,
            sigma=sigma,
            height=height,
            biasfactor=biasfactor,
            kBT=kBT,
            path_points=path_points_t,
            device=device,
            grid_bounds=grid_bounds,
            grid_bins=grid_bins,
            lambda_path=lambda_path,
            z_wall_max=z_wall_max,
            z_wall_kappa=z_wall_kappa,
            reinit_velocities=reinit_velocities
        )
        xy = torch.tensor(xy0, dtype=torch.get_default_dtype(), device=device)

        # Initialize velocities
        if v0 is None:
            sigma_v = torch.sqrt(kBT_t / mass_t)
            v = sigma_v * torch.randn_like(xy)
        else:
            v = torch.tensor(v0, dtype=torch.get_default_dtype(), device=device)

        # Restore previous trajectories
        traj_t = prev_traj_data.get('t', [])
        traj_s = prev_traj_data.get('s', [])
        traj_z = prev_traj_data.get('z', [])
        traj_xy = prev_traj_data.get('xy', np.array([])).tolist()
        traj_u_bias = prev_traj_data.get('bias', [])
        traj_n_hills = prev_traj_data.get('n_hills', [])

    else:
        # Fresh start
        start_step = 0
        xy = torch.tensor(xy0, dtype=torch.get_default_dtype(), device=device)

        # Initialize metadynamics
        metad = WellTemperedMetadynamics1D(
            sigma=sigma,
            height=height,
            biasfactor=biasfactor,
            kBT=kBT,
            path_points=path_points_t,
            device=device,
            grid_bounds=grid_bounds,
            grid_bins=grid_bins,
            lambda_path=lambda_path,
            z_wall_max=z_wall_max,
            z_wall_kappa=z_wall_kappa
        )

        # Initialize velocities
        sigma_v = torch.sqrt(kBT_t / mass_t)
        v = sigma_v * torch.randn_like(xy)

        # Empty trajectories
        traj_t = []
        traj_xy = []
        traj_s = []
        traj_z = []
        traj_u_bias = []
        traj_n_hills = []

    # Helper function for periodic dumps
    def save_checkpoint(step, traj_t, traj_xy, traj_s, traj_z, traj_u_bias, traj_n_hills,
                       current_xy, current_v):
        """Save checkpoint during simulation."""
        checkpoint_dir = f"./{output_dir}/checkpoint_step_{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save trajectory so far
        np.savetxt(
            f"{checkpoint_dir}/metad_traj.dat",
            np.column_stack([
                traj_t,
                traj_s,
                traj_z,
                traj_u_bias,
                np.ones_like(traj_u_bias) * kBT,
                traj_n_hills
            ]),
            header="t  s  z  V_bias  kBT  n_hills"
        )

        # Save xy trajectory
        np.savetxt(
            f"{checkpoint_dir}/metad_traj_xy.dat",
            np.column_stack([
                traj_t,
                traj_xy[:, 0],
                traj_xy[:, 1],
                traj_u_bias
            ]),
            header="t  x  y  V_bias"
        )

        # Save current state for restart
        np.save(f"{checkpoint_dir}/current_xy.npy", current_xy.cpu().numpy())
        np.save(f"{checkpoint_dir}/current_v.npy", current_v.cpu().numpy())

        # Save bias grid
        np.save(f"{checkpoint_dir}/bias_grid.npy", metad.bias_grid.cpu().numpy())

        # Save hills
        hills_data = [[c.cpu().numpy(), h.cpu().numpy()]
                      for c, h in zip(metad.hills_centers, metad.hills_heights)]
        np.savetxt(f"{checkpoint_dir}/hills.dat", np.array(hills_data),
                   header="s_center  height")

        # Save checkpoint info
        with open(f"{checkpoint_dir}/checkpoint_info.txt", "w") as f:
            f.write(f"step: {step}\n")
            f.write(f"n_hills: {len(metad.hills_centers)}\n")
            f.write(f"n_frames: {len(traj_t)}\n")

        print(f"  -> Checkpoint saved at step {step} ({len(metad.hills_centers)} hills)")


    # Initial forces
    F_u = force_unbiased(xy)
    F_b, s_init, Ubias_init, z_init = metad.compute_bias_force(xy)

    # Functions for BAOAB
    def force_function_u(pos):
        return force_unbiased(pos)

    def force_function_b(pos):
        Fb, _, _, _ = metad.compute_bias_force(pos)
        return Fb

    # Main simulation loop
    for step in tqdm(range(start_step, nsteps), desc="Path Metadynamics", initial=start_step, total=nsteps):
        # Add Gaussian hill
        if step % deposition_stride == 0 and step > 0:
            _, s_current, _, _ = metad.compute_bias_force(xy)
            metad.add_hill(s_current)

        # Integrate
        xy, v, F_u, F_b = baoab_langevin_integrator(
            positions=xy,
            velocities=v,
            forces_u=F_u,
            forces_b=F_b,
            mass=mass_t,
            gamma=gamma_t,
            dt=dt_t,
            kBT=kBT_t,
            force_function_u=force_function_u,
            force_function_b=force_function_b,
            n_steps=1,
        )

        # Sample trajectory
        if step % stride == 0:
            _, s, Ubias, z = metad.compute_bias_force(xy)
            traj_t.append(step * dt)
            traj_xy.append(xy.cpu().numpy().copy())
            traj_s.append(float(s.cpu()))
            traj_z.append(float(z.cpu()))
            traj_u_bias.append(float(Ubias.cpu()))
            traj_n_hills.append(len(metad.hills_centers))

        # Periodic checkpoint
        if dump_stride > 0 and step > 0 and step % dump_stride == 0:
            save_checkpoint(
                step,
                np.array(traj_t),
                np.array(traj_xy),
                np.array(traj_s),
                np.array(traj_z),
                np.array(traj_u_bias),
                np.array(traj_n_hills),
                xy,
                v
            )

    traj_xy = np.array(traj_xy)      # (N, 2)
    traj_s = np.array(traj_s)        # (N,)
    traj_z = np.array(traj_z)        # (N,)
    traj_u_bias = np.array(traj_u_bias)  # (N,)
    traj_n_hills = np.array(traj_n_hills)  # (N,)

    # Save final checkpoint for restarting (at completion)
    print("\nSaving final checkpoint for restart...")
    save_checkpoint(
        nsteps - 1,  # Final step
        traj_t,
        traj_xy,
        traj_s,
        traj_z,
        traj_u_bias,
        traj_n_hills,
        xy,
        v
    )

    return traj_t, traj_xy, traj_s, traj_z, traj_u_bias, traj_n_hills, metad

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # Metadynamics parameters
    beta = 0.5              # 1/(k_B T)
    kBT = 1.0 / beta

    # Gaussian parameters in CV space
    height = 5.0             # Initial height - adjust for exploration rate
    biasfactor = 20.0       # Bias factor γ (typical: 5-20)
    deposition_stride = 500 # Add Gaussian every N steps

    # Define path points along the reaction coordinate

    n_interp = 20  # points per segment
    # From find_min.ipynb
    minA = np.array([-0.56762427,  1.43228364])
    minC = np.array([0.18218053,  0.34740155])
    minB = np.array([0.53938401,  0.03642641])

    # Interpolate A -> C -> B
    path_A_to_C = np.linspace(minA, minC, n_interp, endpoint=False)
    path_C_to_B = np.linspace(minC, minB, n_interp + 1)
    path_points = np.vstack([path_A_to_C, path_C_to_B])

    # Grid parameters for 1D path CV
    # PLUMED-style s ranges from 1 to n_path_points
    n_path_points = len(path_points)
    grid_bounds = (1.0, float(n_path_points))  # s in [1, n_path_points]
    grid_bins = 1000          # Grid resolution

    # Lambda controls sharpness of path localization
    lambda_path = 100.0

    # Sigma for Gaussian hills in s-space
    sigma = 0.5

    # Z-wall parameters to keep system on-path
    # z is the distance from the path; wall activates when z > z_wall_max
    z_wall_max = 0.05        # Maximum allowed distance from path
    z_wall_kappa = 100.0    # Wall force constant

    # Simulation parameters
    nsteps = 5_000_000      # Simulation length
    dt = 1e-4
    stride = 100            # Save every N steps
    dump_stride = 1_000_000   # Save checkpoint every N steps (set to 0 to disable)
    output_dir = "path_metad_data"

    # Initial position (start near path)
    xy0 = [-0.558, 1.442]   # Near minimum A

    # Run metadynamics
    print("Running well-tempered metadynamics with path CV...")
    print(f"Grid: {grid_bins} points covering {grid_bounds}")
    print(f"Path defined by {len(path_points)} points")
    if z_wall_max is not None:
        print(f"Z-wall: z_max={z_wall_max}, kappa={z_wall_kappa}")
    if dump_stride > 0:
        print(f"Checkpoints will be saved every {dump_stride} steps to {output_dir}/")

    traj_t, traj_xy, traj_s, traj_z, traj_u_bias, traj_n_hills, metad = run_metadynamics(
        xy0=xy0,
        path_points=path_points,
        sigma=sigma,
        height=height,
        biasfactor=biasfactor,
        deposition_stride=deposition_stride,
        beta=beta,
        dt=dt,
        nsteps=nsteps,
        grid_bounds=grid_bounds,
        grid_bins=grid_bins,
        lambda_path=lambda_path,
        z_wall_max=z_wall_max,
        z_wall_kappa=z_wall_kappa,
        mass=1.0,
        gamma=1.0,
        stride=stride,
        dump_stride=dump_stride,
        output_dir=output_dir
    )

    print(f"Simulation complete. Total hills deposited: {len(metad.hills_centers)}")

    # Save final trajectory (for reweighting)
    np.savetxt(
        f"./{output_dir}/metad_traj.dat",
        np.column_stack([
            traj_t,
            traj_s,
            traj_z,
            traj_u_bias,
            np.ones_like(traj_u_bias) * kBT,
            traj_n_hills
        ]),
        header="t  s  z  V_bias  kBT  n_hills"
    )

    # Save xy trajectory
    np.savetxt(
        f"./{output_dir}/metad_traj_xy.dat",
        np.column_stack([
            traj_t,
            traj_xy[:, 0],
            traj_xy[:, 1],
            traj_u_bias
        ]),
        header="t  x  y  V_bias"
    )

    # Save hills for analysis/restart
    hills_data = []
    for center, h in zip(metad.hills_centers, metad.hills_heights):
        hills_data.append([
            center.cpu().numpy(),
            h.cpu().numpy()
        ])

    np.savetxt(
        f"./{output_dir}/hills.dat",
        np.array(hills_data),
        header="s_center  height"
    )

    # Save bias grid for visualization
    np.save(f"./{output_dir}/bias_grid.npy", metad.bias_grid.cpu().numpy())
    np.save(f"./{output_dir}/s_grid.npy", metad.s_grid.cpu().numpy())

    # Save path points
    np.save(f"./{output_dir}/path_points.npy", path_points)

    # Save grid info
    with open(f"./{output_dir}/grid_info.txt", "w") as f:
        f.write(f"grid_bounds: {grid_bounds}\n")
        f.write(f"grid_bins: {grid_bins}\n")
        f.write(f"sigma: {sigma}\n")
        f.write(f"biasfactor: {biasfactor}\n")
        f.write(f"lambda_path: {lambda_path}\n")
        f.write(f"n_path_points: {n_path_points}\n")
        f.write(f"z_wall_max: {z_wall_max}\n")
        f.write(f"z_wall_kappa: {z_wall_kappa}\n")

    print(f"\n{'='*60}")
    print(f"Final data saved to {output_dir}/")
    print(f"{'='*60}")
    print("\nFor reweighting, use V_bias column with standard metadynamics reweighting:")
    print("  P(s) ~ Sum_t exp(beta * V_bias(s, t)) * delta(s - s(t))")
    print("\nThe final bias grid approximates -F(s) (negative free energy)")
    print(f"Load with: bias_grid = np.load('{output_dir}/bias_grid.npy')")

    # Print restart information
    final_checkpoint = f"{output_dir}/checkpoint_step_{nsteps-1}"
    print(f"\nRestart files saved to: {final_checkpoint}/")
    print(f"  - current_xy.npy, current_v.npy (for continuing simulation)")
    print(f"  - bias_grid.npy, hills.dat (full metadynamics state)")

    if dump_stride > 0:
        print(f"\nIntermediate checkpoints available in {output_dir}/checkpoint_step_*/")

    print(f"\nTo restart/extend this simulation, use:")
    print(f"  restart_from='{final_checkpoint}'")
