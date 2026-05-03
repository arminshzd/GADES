import numpy as np
from scipy.stats import gaussian_kde

def _autocorr_fft(y):
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    n = len(y)

    if n < 2:
        return np.array([1.0])

    c0 = np.dot(y, y)
    if not np.isfinite(c0) or c0 <= 1e-30:
        # effectively zero variance series
        return np.array([1.0])

    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    acf /= c0
    return acf

def _integrated_autocorr_time(y, c=5.0):
    acf = _autocorr_fft(y)

    if len(acf) == 1:
        return 0.5

    tau = 0.5
    for m in range(1, len(acf)):
        if not np.isfinite(acf[m]) or acf[m] <= 0:
            break
        tau += acf[m]
        if m > c * tau:
            break

    return max(float(tau), 0.5)

def _mean_cov_se(a, b):
    """
    Autocorrelation-corrected covariance matrix of sample means of
    correlated series a_t and b_t.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if len(b) != n:
        raise ValueError("a and b must have same length")

    am = a.mean()
    bm = b.mean()
    ac = a - am
    bc = b - bm

    var_a = np.mean(ac * ac)
    var_b = np.mean(bc * bc)
    cov_ab = np.mean(ac * bc)

    # Use tau from sum and difference to estimate cross-correlation time
    tau_a = _integrated_autocorr_time(a)
    tau_b = _integrated_autocorr_time(b)
    tau_p = _integrated_autocorr_time(a + b)
    tau_m = _integrated_autocorr_time(a - b)

    # Since Var(a±b) = Var(a)+Var(b)±2Cov(a,b), same logic for tau-corrected means
    gamma_p = (var_a + var_b + 2 * cov_ab) * (2 * tau_p)
    gamma_m = (var_a + var_b - 2 * cov_ab) * (2 * tau_m)

    var_mean_a = var_a * (2 * tau_a) / n
    var_mean_b = var_b * (2 * tau_b) / n
    cov_mean_ab = (gamma_p - gamma_m) / (4 * n)

    return am, bm, var_mean_a, var_mean_b, cov_mean_ab

def weighted_kde_with_se(
    x,
    weights,
    grid,
    bw_method="scott",
    normalize_weights=True,
):
    """
    Approximate pointwise SE band for a weighted KDE from correlated samples.

    Parameters
    ----------
    x : (N,) array
        Samples.
    weights : (N,) array
        Boltzmann/reweighting factors per sample.
    grid : (M,) array
        Points where KDE and SE are evaluated.
    bw_method : str, scalar, or callable
        Passed to scipy.stats.gaussian_kde.
    normalize_weights : bool
        If True, rescales weights to mean 1 for numerical stability.

    Returns
    -------
    density : (M,) array
    se : (M,) array
        Approximate pointwise standard error.
    info : dict
        Diagnostics.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()

    if len(x) != len(w):
        raise ValueError("x and weights must have the same length")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if np.sum(w) <= 0:
        raise ValueError("sum of weights must be positive")

    if normalize_weights:
        w = w / np.mean(w)

    kde = gaussian_kde(x, weights=w, bw_method=bw_method)
    density = kde(grid)

    # gaussian_kde stores covariance of kernel, so in 1D:
    h = float(np.sqrt(kde.covariance.squeeze()))
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * h)

    se = np.empty_like(grid)

    for i, gx in enumerate(grid):
        K = norm * np.exp(-0.5 * ((gx - x) / h) ** 2)

        # numerator/denominator series
        Y = w * K
        W = w

        muY, muW, var_muY, var_muW, cov_muYW = _mean_cov_se(Y, W)

        # delta method for ratio muY / muW
        # Var(f) ≈ (df/dmuY)^2 Var(muY) + (df/dmuW)^2 Var(muW)
        #         + 2 (df/dmuY)(df/dmuW) Cov(muY,muW)
        # with f = muY / muW
        var_f = (
            var_muY / (muW ** 2)
            + (muY ** 2) * var_muW / (muW ** 4)
            - 2 * muY * cov_muYW / (muW ** 3)
        )

        se[i] = np.sqrt(max(var_f, 0.0))

        # if var_f > 0:
        #     se[i] = np.sqrt(var_f)
        # else:
        #     # fallback: ignore denominator fluctuations
        #     se[i] = np.sqrt(max(var_muY, 0.0)) / abs(muW)

    info = {
        "bandwidth": h,
        "raw_weight_ess_iid": (w.sum() ** 2) / np.sum(w ** 2),
        "tau_int_weights": _integrated_autocorr_time(w),
        "kde": kde,
    }
    return density, se, info