import numpy as np
from tqdm.notebook import tqdm

# Mean shift: start from each data point, shift toward density maxima
def mean_shift(points, kde, tol=1e-5, max_iter=300, bandwidth=None, min_density=None):
    """Find local maxima of a KDE via mean shift.

    Parameters
    ----------
    points : (n, d) array - starting points for mean shift trajectories
    kde : scipy.stats.gaussian_kde - fitted density estimate
    tol : convergence tolerance on shift magnitude
    max_iter : maximum iterations per trajectory
    bandwidth : kernel bandwidth for mean shift weights (uses KDE bandwidth if None)
    min_density : minimum KDE density for returned modes (None keeps all)
    """
    if bandwidth is None:
        # Use mean bandwidth across all dimensions for anisotropic KDEs
        bandwidth = np.sqrt(np.mean(np.diag(kde.covariance)))
    
    dataset = kde.dataset.T  # (n_data, d)
    shifted = points.copy()
    
    pbar = tqdm(range(max_iter), desc="Mean shift")
    for _ in pbar:
        prev = shifted.copy()
        # Pairwise squared distances: (n_starts, n_data)
        diff = shifted[:, None, :] - dataset[None, :, :]  # (n_starts, n_data, d)
        dists_sq = np.sum(diff**2, axis=2)                 # (n_starts, n_data)
        # Gaussian weights
        weights = np.exp(-0.5 * dists_sq / bandwidth**2)   # (n_starts, n_data)
        # Weighted mean for all points at once
        w_sum = weights.sum(axis=1, keepdims=True)          # (n_starts, 1)
        shifted = (weights @ dataset) / w_sum               # (n_starts, d)
        # Check convergence
        max_shift = np.max(np.linalg.norm(shifted - prev, axis=1))
        pbar.set_postfix(max_shift=f"{max_shift:.2e}")
        if max_shift < tol:
            break

    if min_density is not None:
        densities = kde(shifted.T)
        shifted = shifted[densities >= min_density]

    return shifted

# Deduplicate converged modes
def deduplicate(modes, threshold=0.1):
    if len(modes) == 0:
        return modes
    unique = [modes[0]]
    for m in modes[1:]:
        if all(np.linalg.norm(m - u) > threshold for u in unique):
            unique.append(m)
    return np.array(unique)