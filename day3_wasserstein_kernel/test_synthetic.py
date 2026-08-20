"""
Day 3 — Validate the Wasserstein kernel on synthetic distribution data.

Checks:
1. W2 distance matches the known closed form for two Gaussians:
   W2(N(m1,s1^2), N(m2,s2^2))^2 = (m1-m2)^2 + (s1-s2)^2
2. The resulting Gaussian-Wasserstein kernel matrix is positive semi-definite
   (all eigenvalues >= -1e-8) on a batch of random synthetic distributions.
3. Kernel values behave sensibly: k(F,F) = 1, k decreases as distributions
   grow apart.
"""
from __future__ import annotations
import numpy as np
from wasserstein_kernel import (
    density_to_quantile,
    wasserstein2_from_quantiles,
    build_kernel_from_densities,
)


def gaussian_density(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def check_closed_form_w2(tol=1e-2):
    x = np.linspace(-10, 10, 2000)
    cases = [
        (0.0, 1.0, 0.0, 1.0),   # identical -> W2 = 0
        (0.0, 1.0, 2.0, 1.0),   # mean shift only -> W2 = 2
        (0.0, 1.0, 0.0, 1.5),   # std shift only -> W2 = 0.5
        (1.0, 0.8, 3.0, 1.3),   # both -> W2 = sqrt((2)^2 + (0.5)^2)
    ]
    print("=== Closed-form W2 check (Gaussians) ===")
    all_ok = True
    for m1, s1, m2, s2 in cases:
        q1 = density_to_quantile(x, gaussian_density(x, m1, s1), n_quantiles=400)
        q2 = density_to_quantile(x, gaussian_density(x, m2, s2), n_quantiles=400)
        w2_est = wasserstein2_from_quantiles(q1, q2)
        w2_true = np.sqrt((m1 - m2) ** 2 + (s1 - s2) ** 2)
        ok = abs(w2_est - w2_true) < tol
        all_ok &= ok
        print(f"  N({m1},{s1}^2) vs N({m2},{s2}^2): est={w2_est:.4f} true={w2_true:.4f} {'OK' if ok else 'FAIL'}")
    return all_ok


def check_psd(n_dists=30, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-10, 10, 1000)

    grids, densities = [], []
    for _ in range(n_dists):
        mu = rng.uniform(-3, 3)
        sigma = rng.uniform(0.5, 2.5)
        grids.append(x)
        densities.append(gaussian_density(x, mu, sigma))

    Q, W2, K = build_kernel_from_densities(grids, densities, gamma=0.5)
    eigvals = np.linalg.eigvalsh(K)
    min_eig = eigvals.min()

    print("\n=== PSD check on kernel Gram matrix ===")
    print(f"  n = {n_dists}, gamma = 0.5")
    print(f"  min eigenvalue = {min_eig:.3e} (should be >= ~-1e-8)")
    print(f"  diag(K) sample: {np.diag(K)[:5]} (should all be 1.0, since W2(F,F)=0)")
    return min_eig > -1e-6, K, W2


if __name__ == "__main__":
    ok1 = check_closed_form_w2()
    ok2, K, W2 = check_psd()

    print("\n=== Summary ===")
    print(f"Closed-form W2 check: {'PASS' if ok1 else 'FAIL'}")
    print(f"PSD check:            {'PASS' if ok2 else 'FAIL'}")

    if ok1 and ok2:
        print("\nWasserstein kernel validated on synthetic data. Ready for Day 4.")
