from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# 1. DENSITY -> QUANTILE
# ============================================================

def density_to_quantile(
    x_grid: np.ndarray,
    density: np.ndarray,
    n_quantiles: int = 200
) -> np.ndarray:
    """
    Convert a 1D density into its quantile function.

    x_grid must be increasing.
    """

    x_grid = np.asarray(x_grid, dtype=float)
    density = np.asarray(density, dtype=float)

    # Remove invalid values
    mask = (
        np.isfinite(x_grid)
        & np.isfinite(density)
    )

    x_grid = x_grid[mask]
    density = density[mask]

    # Ensure non-negative density
    density = np.maximum(
        density,
        0.0
    )

    # Normalize density
    mass = np.trapezoid(
        density,
        x_grid
    )

    if mass <= 0:
        raise ValueError(
            "Density has zero probability mass."
        )

    density = density / mass

    # CDF using trapezoidal integration
    dx = np.diff(x_grid)

    cdf = np.concatenate([
        [0.0],
        np.cumsum(
            0.5
            * (
                density[1:]
                + density[:-1]
            )
            * dx
        )
    ])

    # Numerical normalization
    cdf = cdf / cdf[-1]

    # Remove duplicate CDF values
    cdf_unique, idx = np.unique(
        cdf,
        return_index=True
    )

    x_unique = x_grid[idx]

    # Common probability grid
    u = np.linspace(
        1e-4,
        1 - 1e-4,
        n_quantiles
    )

    q = np.interp(
        u,
        cdf_unique,
        x_unique
    )

    return q


# ============================================================
# 2. WASSERSTEIN DISTANCE
# ============================================================

def wasserstein2_from_quantiles(
    q1: np.ndarray,
    q2: np.ndarray
) -> float:
    """
    1D 2-Wasserstein distance.

        W2² = integral (Q1(u)-Q2(u))² du
    """

    if len(q1) != len(q2):
        raise ValueError(
            "Quantile vectors must have equal length."
        )

    u = np.linspace(
        1e-4,
        1 - 1e-4,
        len(q1)
    )

    w2_squared = np.trapezoid(
        (q1 - q2) ** 2,
        u
    )

    return float(
        np.sqrt(
            max(w2_squared, 0.0)
        )
    )


def wasserstein_gram_matrix(
    quantile_matrix: np.ndarray
) -> np.ndarray:
    """
    Pairwise W2 distance matrix.
    """

    n = quantile_matrix.shape[0]

    W = np.zeros(
        (n, n),
        dtype=float
    )

    for i in range(n):

        for j in range(i + 1, n):

            distance = (
                wasserstein2_from_quantiles(
                    quantile_matrix[i],
                    quantile_matrix[j]
                )
            )

            W[i, j] = distance
            W[j, i] = distance

    return W


# ============================================================
# 3. MEDIAN-HEURISTIC GAMMA
# ============================================================

def median_heuristic_gamma(
    W2_matrix: np.ndarray,
    min_gamma: float = 1e-8
) -> float:
    """
    Stable median heuristic for

        k(F,G) = exp(-gamma * W2(F,G)^2)

    gamma = 1 / median(W2²)

    The diagonal zeros are excluded.
    """

    W2 = np.asarray(
        W2_matrix,
        dtype=float
    )

    n = W2.shape[0]

    values = []

    for i in range(n):

        for j in range(i + 1, n):

            value = W2[i, j] ** 2

            if np.isfinite(value) and value > 0:
                values.append(value)

    if len(values) == 0:
        return min_gamma

    median_sq = float(
        np.median(values)
    )

    if median_sq <= 0:
        return min_gamma

    gamma = 1.0 / median_sq

    return max(
        gamma,
        min_gamma
    )


# ============================================================
# 4. GAUSSIAN WASSERSTEIN KERNEL
# ============================================================

def gaussian_wasserstein_kernel(
    W2_matrix: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    k(F_i,F_j) =
        exp(-gamma * W2(F_i,F_j)^2)
    """

    W2_matrix = np.asarray(
        W2_matrix,
        dtype=float
    )

    K = np.exp(
        -gamma * W2_matrix**2
    )

    # Numerical symmetry
    K = 0.5 * (
        K + K.T
    )

    # Exact diagonal
    np.fill_diagonal(
        K,
        1.0
    )

    return K


# ============================================================
# 5. STABLE KERNEL WEIGHTS
# ============================================================

def stable_kernel_weights(
    similarities: np.ndarray,
    nearest_index: int | None = None
) -> np.ndarray:
    """
    Convert kernel similarities into stable weights.

    Required stabilization from Sven's review:

    - weights must always sum to 1
    - if numerical underflow/collapse occurs,
      use an explicit nearest-neighbour fallback
    """

    similarities = np.asarray(
        similarities,
        dtype=float
    )

    similarities = np.maximum(
        similarities,
        0.0
    )

    total = similarities.sum()

    # Normal case
    if np.isfinite(total) and total > 1e-14:

        weights = (
            similarities / total
        )

        # Final normalization
        weights = (
            weights
            / weights.sum()
        )

        return weights

    # --------------------------------------------------------
    # Explicit nearest-neighbour fallback
    # --------------------------------------------------------

    weights = np.zeros_like(
        similarities
    )

    if nearest_index is None:

        # If no neighbour supplied,
        # use the largest similarity.
        nearest_index = int(
            np.argmax(similarities)
        )

    weights[nearest_index] = 1.0

    return weights


# ============================================================
# 6. BUILD KERNEL FROM DENSITIES
# ============================================================

def build_kernel_from_densities(
    grids: list[np.ndarray],
    densities: list[np.ndarray],
    gamma: float | None = None,
    n_quantiles: int = 200
):
    """
    Convert densities to quantiles, calculate W2 distances,
    and construct the Gaussian Wasserstein kernel.

    If gamma is None, use the median heuristic.
    """

    Q = np.stack([
        density_to_quantile(
            g,
            d,
            n_quantiles
        )
        for g, d in zip(
            grids,
            densities
        )
    ])

    W2 = wasserstein_gram_matrix(
        Q
    )

    # Median-heuristic stabilization
    if gamma is None:

        gamma = median_heuristic_gamma(
            W2
        )

    K = gaussian_wasserstein_kernel(
        W2,
        gamma
    )

    return Q, W2, K, gamma


# ============================================================
# 7. LOAD RND DATA AND NORMALIZE BY FORWARD
# ============================================================

def load_forward_normalized_rnds(
    path: str,
    n_quantiles: int = 200
):
    """
    Load RND curves and transform

        S_T -> S_T / F_t

    so Wasserstein distance measures distribution shape
    rather than absolute SPX level.
    """

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    grids = []
    densities = []
    dates = []
    forwards = []

    for date, g in df.groupby("date"):

        g = g.sort_values(
            "strike"
        )

        strike = g[
            "strike"
        ].to_numpy(
            dtype=float
        )

        density = g[
            "density"
        ].to_numpy(
            dtype=float
        )

        forward = float(
            g["forward"].iloc[0]
        )

        # Forward-normalized coordinate
        x = strike / forward

        grids.append(
            x
        )

        densities.append(
            density
        )

        dates.append(
            date
        )

        forwards.append(
            forward
        )

    Q, W2, K, gamma = (
        build_kernel_from_densities(
            grids,
            densities,
            gamma=None,
            n_quantiles=n_quantiles
        )
    )

    return (
        dates,
        forwards,
        Q,
        W2,
        K,
        gamma
    )


# ============================================================
# 8. MAIN TEST
# ============================================================

if __name__ == "__main__":

    INPUT = (
        r"02_rnd_extraction\data\rnd_30d.csv"
    )

    print("=" * 70)
    print("FORWARD-NORMALIZED WASSERSTEIN KERNEL")
    print("=" * 70)

    (
        dates,
        forwards,
        Q,
        W2,
        K,
        gamma
    ) = load_forward_normalized_rnds(
        INPUT,
        n_quantiles=200
    )

    print()
    print(
        f"RND days       : {len(dates)}"
    )

    print(
        f"Quantile shape : {Q.shape}"
    )

    print(
        f"W2 min         : {W2[W2 > 0].min():.6f}"
    )

    print(
        f"W2 max         : {W2.max():.6f}"
    )

    print(
        f"Median gamma   : {gamma:.6f}"
    )

    print()
    print("Kernel diagnostics")
    print("-" * 70)

    eigenvalues = np.linalg.eigvalsh(
        K
    )

    print(
        f"Kernel min eigenvalue : "
        f"{eigenvalues.min():.10e}"
    )

    print(
        f"Kernel max eigenvalue : "
        f"{eigenvalues.max():.10e}"
    )

    print(
        f"Kernel diagonal       : "
        f"{np.diag(K).min():.6f} "
        f"to "
        f"{np.diag(K).max():.6f}"
    )

    print()
    print("First W2 row:")
    print(
        np.round(
            W2[0],
            6
        )
    )

    print()
    print("=" * 70)
    print("WASSERSTEIN KERNEL COMPLETE")
    print("=" * 70)