"""
Day 3 — Wasserstein-distance kernel for GP inputs (Bachoc et al., 2018 style).

For densities on the real line, the 2-Wasserstein distance has a closed form
via quantile functions:

    W2(F, G)^2 = integral_0^1 ( F^{-1}(u) - G^{-1}(u) )^2 du

This is exactly the L2 distance between the two quantile functions viewed as
vectors — i.e. 1D distributions embed *isometrically* into L2([0,1]) via their
quantile functions. Consequence (the key fact this file relies on, and the
special case of the Bachoc et al. Hilbertian-embedding construction used
here): any positive-definite kernel of Euclidean distance, e.g. the Gaussian
kernel

    k(F, G) = exp( -gamma * W2(F, G)^2 )

is automatically a valid (positive semi-definite) kernel on the space of 1D
distributions, because it is literally a Gaussian kernel applied to the
quantile-function embedding. This sidesteps the general-metric-space PD
issues that plague naive "kernelize any distance" approaches, which is why
Day 3's test is a PSD check on synthetic data rather than a proof.
"""
from __future__ import annotations
import numpy as np


def density_to_quantile(x_grid: np.ndarray, density: np.ndarray, n_quantiles: int = 200) -> np.ndarray:
    """
    Convert a density evaluated on x_grid into a quantile function evaluated
    on a uniform probability grid u_1, ..., u_{n_quantiles} in (0, 1).
    """
    density = np.maximum(density, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x_grid))])
    cdf = cdf / cdf[-1]  # normalize to exactly [0, 1]

    # de-duplicate for monotone interpolation
    cdf_unique, idx = np.unique(cdf, return_index=True)
    x_unique = x_grid[idx]

    u = np.linspace(1e-4, 1 - 1e-4, n_quantiles)
    q = np.interp(u, cdf_unique, x_unique)
    return q


def wasserstein2_from_quantiles(q1: np.ndarray, q2: np.ndarray) -> float:
    """W2 distance between two distributions given their quantile functions
    on the SAME uniform grid (trapezoidal approximation of the integral)."""
    u = np.linspace(1e-4, 1 - 1e-4, len(q1))
    return float(np.sqrt(np.trapezoid((q1 - q2) ** 2, u)))


def wasserstein_gram_matrix(quantile_matrix: np.ndarray) -> np.ndarray:
    """
    quantile_matrix: (n_obs, n_quantiles) — each row is one distribution's
    quantile function on a shared grid.
    Returns the (n_obs, n_obs) matrix of pairwise W2 distances.
    """
    n = quantile_matrix.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein2_from_quantiles(quantile_matrix[i], quantile_matrix[j])
            W[i, j] = W[j, i] = d
    return W


def gaussian_wasserstein_kernel(W2_matrix: np.ndarray, gamma: float) -> np.ndarray:
    """k(F_i, F_j) = exp(-gamma * W2(F_i, F_j)^2). PSD for 1D dists (see module docstring)."""
    return np.exp(-gamma * W2_matrix**2)


def build_kernel_from_densities(
    grids: list[np.ndarray], densities: list[np.ndarray], gamma: float, n_quantiles: int = 200
):
    """
    Convenience wrapper: raw list of (x_grid, density) pairs (e.g. one per
    day's RND from Day 2) -> (quantile_matrix, W2_matrix, kernel_matrix).
    """
    Q = np.stack(
        [density_to_quantile(g, d, n_quantiles) for g, d in zip(grids, densities)]
    )
    W2 = wasserstein_gram_matrix(Q)
    K = gaussian_wasserstein_kernel(W2, gamma)
    return Q, W2, K
