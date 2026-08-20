"""
Day 4 — Fit GP regression: RND_t -> target_{t+1} using the Wasserstein kernel.

Target: rather than forecasting the full infinite-dimensional next-day density
(left for a richer output-side extension later), Day 4 forecasts a scalar
functional of tomorrow's RND — its variance (a standard, economically
meaningful summary: the market-implied forward variance) — from today's whole
RND as a distribution-valued input. This is the simplest well-posed
scalar-output instance of the Bachoc et al. (2018) distribution-input GP
framework, and matches "RNDt -> RNDt+1 (or a scalar functional, e.g. implied
variance)" in the plan.

Model:
    y_{t+1} = f(RND_t) + eps,   f ~ GP(0, k_W(RND_t, RND_t'))
    k_W(F, F') = sigma_f^2 * exp( -gamma * W2(F, F')^2 )

Hyperparameters (sigma_f^2, gamma, sigma_n^2) are fit by maximizing the log
marginal likelihood via Cholesky decomposition (standard GP MLE, same
approach used in the WTGP work).
"""
from __future__ import annotations
import sys
import os
import numpy as np
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day3_wasserstein_kernel"))
from wasserstein_kernel import density_to_quantile, wasserstein_gram_matrix  # noqa: E402


def rnd_variance(strike: np.ndarray, density: np.ndarray) -> float:
    """Variance functional of a density given on a strike grid (used as the
    scalar target y_{t+1})."""
    mean = np.trapezoid(strike * density, strike)
    var = np.trapezoid((strike - mean) ** 2 * density, strike)
    return float(var)


def neg_log_marginal_likelihood(log_params: np.ndarray, W2: np.ndarray, y: np.ndarray) -> float:
    log_sigma_f2, log_gamma, log_sigma_n2 = log_params
    sigma_f2, gamma, sigma_n2 = np.exp(log_sigma_f2), np.exp(log_gamma), np.exp(log_sigma_n2)

    n = len(y)
    K = sigma_f2 * np.exp(-gamma * W2**2) + sigma_n2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    fit_term = 0.5 * y @ alpha
    complexity_term = np.sum(np.log(np.diag(L)))
    const_term = 0.5 * n * np.log(2 * np.pi)
    return float(fit_term + complexity_term + const_term)


def median_heuristic_gamma(W2: np.ndarray) -> float:
    """Median heuristic for the kernel lengthscale: pick gamma so the
    *median* off-diagonal pairwise W2 distance sits at kernel value ~1/e,
    i.e. gamma = 1 / median(W2)^2. This is scale-aware by construction --
    it gives gamma ~ 1e-4 for real BTC-price-level W2 distances (hundreds to
    thousands) and gamma ~ 1 for synthetic near-100-scale distances, instead
    of a single fixed constant that only happens to work at one scale."""
    off_diag = W2[np.triu_indices_from(W2, k=1)]
    off_diag = off_diag[off_diag > 0]
    if off_diag.size == 0:
        return 1.0
    med = np.median(off_diag)
    return 1.0 / (med ** 2)


def fit_hyperparameters(W2: np.ndarray, y: np.ndarray, x0=None, gamma_bound_decades: float = 1.5):
    """Fit (sigma_f2, gamma, sigma_n2) by maximizing log marginal likelihood.

    FIX (results_memo.md Next steps #1): gamma used to be unconstrained in
    log-space. On 38 real-price-scale training points that let the optimizer
    drive gamma to a value where exp(-gamma * W2^2) underflows to ~0 for
    almost every pair -- a degenerate, near-one-hot kernel that happened to
    still lower the (scalar-target) marginal likelihood slightly, but made
    the full-density barycenter forecaster (which reuses this gamma) produce
    near-meaningless output, a >100x W2 blowup vs. persistence. Bounding
    log(gamma) to within `gamma_bound_decades` orders of magnitude of the
    median-heuristic value keeps the search in the region where the
    Wasserstein kernel is actually acting as a smooth similarity measure,
    while still letting MLE pick the best gamma *within* that range from the
    data, rather than fixing it outright.
    """
    gamma_med = median_heuristic_gamma(W2)
    if x0 is None:
        x0 = np.log([np.var(y) + 1e-6, gamma_med, 0.01 * np.var(y) + 1e-8])

    log_gamma_lo = np.log(gamma_med) - gamma_bound_decades * np.log(10)
    log_gamma_hi = np.log(gamma_med) + gamma_bound_decades * np.log(10)
    bounds = [(None, None), (log_gamma_lo, log_gamma_hi), (None, None)]

    res = minimize(
        neg_log_marginal_likelihood,
        x0,
        args=(W2, y),
        method="L-BFGS-B",
        bounds=bounds,
    )
    sigma_f2, gamma, sigma_n2 = np.exp(res.x)
    return {
        "sigma_f2": sigma_f2,
        "gamma": gamma,
        "sigma_n2": sigma_n2,
        "neg_log_lik": res.fun,
        "gamma_median_heuristic": gamma_med,
    }


class WassersteinGP:
    """Distribution-input, scalar-output GP with a Wasserstein-Gaussian kernel."""

    def __init__(self):
        self.sigma_f2 = None
        self.gamma = None
        self.sigma_n2 = None
        self.Q_train = None  # quantile functions of training inputs
        self.alpha = None
        self.L = None
        self.y_mean = 0.0

    def fit(self, quantile_matrix: np.ndarray, y: np.ndarray):
        self.y_mean = float(np.mean(y))
        y_centered = y - self.y_mean

        W2_train = wasserstein_gram_matrix(quantile_matrix)
        params = fit_hyperparameters(W2_train, y_centered)
        self.sigma_f2, self.gamma, self.sigma_n2 = (
            params["sigma_f2"],
            params["gamma"],
            params["sigma_n2"],
        )

        n = len(y)
        K = self.sigma_f2 * np.exp(-self.gamma * W2_train**2) + self.sigma_n2 * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_centered))
        self.Q_train = quantile_matrix
        self.train_log_lik = -params["neg_log_lik"]
        return self

    def _w2_to_train(self, q_star: np.ndarray) -> np.ndarray:
        u = np.linspace(1e-4, 1 - 1e-4, q_star.shape[0])
        diffs = self.Q_train - q_star[None, :]
        return np.sqrt(np.trapezoid(diffs**2, u, axis=1))

    def predict(self, quantile_matrix_test: np.ndarray):
        """Returns (posterior_mean, posterior_std) for each test input.

        NOTE (fixed per Sven's review, 2026): sigma_f2 - v@v is the posterior
        variance of the latent function f, i.e. Var[f(x*) | data]. Coverage
        and NLPD in day6/day4 are scored against *realized observations* y,
        which carry the additional observation-noise term sigma_n2. The
        predictive variance for y* is Var[f(x*)|data] + sigma_n2, not
        Var[f(x*)|data] alone. Omitting sigma_n2 here understated predictive
        std (mean predictive std 2.56 vs. sqrt(sigma_n2)=3.84) and dragged
        95% coverage down to 25%; adding it restores calibration (~87.5%,
        matching the baseline) and drops NLPD from 8.34 to 3.92.
        """
        means, stds = [], []
        for q_star in quantile_matrix_test:
            w2_star = self._w2_to_train(q_star)
            k_star = self.sigma_f2 * np.exp(-self.gamma * w2_star**2)

            mean = self.y_mean + k_star @ self.alpha
            v = np.linalg.solve(self.L, k_star)
            f_var = self.sigma_f2 - v @ v
            var = f_var + self.sigma_n2  # predictive (observation-level) variance
            var = max(var, 1e-12)

            means.append(mean)
            stds.append(np.sqrt(var))
        return np.array(means), np.array(stds)


class WassersteinBarycenterForecaster:
    """
    Full-density (not just scalar) next-day RND forecast.

    Because 1D distributions embed isometrically into L2 via their quantile
    functions (see day3), the Wasserstein barycenter of a weighted set of
    distributions is *exactly* the weighted average of their quantile
    functions. So a Nadaraya-Watson kernel regression in quantile space:

        Q_pred(u) = sum_i w_i(x*) * Q_target_train_i(u),
        w_i(x*) = k(x*, x_i) / sum_j k(x*, x_j)

    is a principled, kernel-consistent way to forecast tomorrow's whole RND
    from today's RND, using the same Wasserstein-Gaussian kernel as the
    scalar-target GP. This is what Day 6 compares against the baseline's
    moment-matched Gaussian density forecast via W2 error.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.Q_input_train = None   # quantile functions of RND_t
        self.Q_target_train = None  # quantile functions of RND_{t+1}

    def fit(self, Q_input_train: np.ndarray, Q_target_train: np.ndarray):
        self.Q_input_train = Q_input_train
        self.Q_target_train = Q_target_train
        return self

    def predict_quantile(self, q_star: np.ndarray) -> np.ndarray:
        u = np.linspace(1e-4, 1 - 1e-4, q_star.shape[0])
        diffs = self.Q_input_train - q_star[None, :]
        w2 = np.sqrt(np.trapezoid(diffs**2, u, axis=1))
        k = np.exp(-self.gamma * w2**2)
        weights = k / (k.sum() + 1e-12)
        return weights @ self.Q_target_train

    def predict(self, Q_input_test: np.ndarray) -> np.ndarray:
        return np.stack([self.predict_quantile(q) for q in Q_input_test])


if __name__ == "__main__":
    import pandas as pd

    rnd = pd.read_csv("../day2_rnd_extraction/data/rnd_curves.csv", parse_dates=["date"])
    dates = sorted(rnd["date"].unique())

    grids, densities = [], []
    for d in dates:
        g = rnd[rnd["date"] == d].sort_values("strike")
        grids.append(g["strike"].to_numpy())
        densities.append(g["density"].to_numpy())

    Q = np.stack([density_to_quantile(g, dens, n_quantiles=200) for g, dens in zip(grids, densities)])
    y_all = np.array([rnd_variance(g, dens) for g, dens in zip(grids, densities)])

    # Inputs: RND_t (t = 0..T-2). Targets: variance functional of RND_{t+1}.
    X_Q = Q[:-1]
    y = y_all[1:]

    n = len(y)
    n_train = int(n * 0.8)
    Q_train, Q_test = X_Q[:n_train], X_Q[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    gp = WassersteinGP().fit(Q_train, y_train)
    mean_pred, std_pred = gp.predict(Q_test)

    print("=== Fitted hyperparameters ===")
    print(f"  sigma_f^2 = {gp.sigma_f2:.5f}")
    print(f"  gamma     = {gp.gamma:.5f}")
    print(f"  sigma_n^2 = {gp.sigma_n2:.5f}")
    print(f"  train log marginal likelihood = {gp.train_log_lik:.3f}")

    rmse = np.sqrt(np.mean((mean_pred - y_test) ** 2))
    z = (y_test - mean_pred) / std_pred
    coverage_95 = np.mean(np.abs(z) < 1.96)

    print("\n=== Held-out performance (n_test={}) ===".format(len(y_test)))
    print(f"  RMSE            = {rmse:.6f}")
    print(f"  95% coverage    = {coverage_95:.2%}  (predictive var now includes sigma_n2; full comparison vs baseline/persistence is Day 6)")

    out = pd.DataFrame(
        {
            "date": dates[n_train + 1 : n + 1],
            "y_true": y_test,
            "y_pred_mean": mean_pred,
            "y_pred_std": std_pred,
        }
    )
    out.to_csv("results/predictions.csv", index=False)
    with open("results/summary.txt", "w") as f:
        f.write(f"sigma_f2={gp.sigma_f2}\ngamma={gp.gamma}\nsigma_n2={gp.sigma_n2}\n")
        f.write(f"train_log_lik={gp.train_log_lik}\nrmse={rmse}\ncoverage_95={coverage_95}\n")
    print("\nWrote results/predictions.csv and results/summary.txt")
