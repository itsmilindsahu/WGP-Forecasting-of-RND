from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# IMPORT WASSERSTEIN FUNCTIONS
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "03_wasserstein_kernel"
    )
)

from wasserstein_kernel import (
    density_to_quantile,
    wasserstein_gram_matrix,
    median_heuristic_gamma,
    stable_kernel_weights
)


# ============================================================
# FILES
# ============================================================

RND_FILE = (
    r"02_rnd_extraction\data\rnd_30d.csv"
)

RESULT_DIR = (
    r"04_gp_model\results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# RND VARIANCE
# ============================================================

def rnd_variance(
    x: np.ndarray,
    density: np.ndarray
) -> float:

    density = np.maximum(
        density,
        0.0
    )

    mass = np.trapezoid(
        density,
        x
    )

    if mass <= 0:
        raise ValueError(
            "Density has zero mass."
        )

    density = density / mass

    mean = np.trapezoid(
        x * density,
        x
    )

    variance = np.trapezoid(
        (x - mean) ** 2 * density,
        x
    )

    return float(variance)


# ============================================================
# GP NEGATIVE LOG MARGINAL LIKELIHOOD
# ============================================================

def neg_log_marginal_likelihood(
    log_params: np.ndarray,
    W2: np.ndarray,
    y: np.ndarray
) -> float:

    log_sigma_f2, log_gamma, log_sigma_n2 = (
        log_params
    )

    sigma_f2 = np.exp(
        log_sigma_f2
    )

    gamma = np.exp(
        log_gamma
    )

    sigma_n2 = np.exp(
        log_sigma_n2
    )

    n = len(y)

    K = (
        sigma_f2
        * np.exp(
            -gamma * W2**2
        )
        + sigma_n2 * np.eye(n)
    )

    try:

        L = np.linalg.cholesky(
            K
        )

    except np.linalg.LinAlgError:

        return 1e10

    alpha = np.linalg.solve(
        L.T,
        np.linalg.solve(
            L,
            y
        )
    )

    fit_term = (
        0.5 * y @ alpha
    )

    complexity_term = np.sum(
        np.log(
            np.diag(L)
        )
    )

    constant_term = (
        0.5
        * n
        * np.log(2 * np.pi)
    )

    return float(
        fit_term
        + complexity_term
        + constant_term
    )


# ============================================================
# FIT GP HYPERPARAMETERS
# ============================================================

def fit_hyperparameters(
    W2: np.ndarray,
    y: np.ndarray,
    gamma_bound_decades: float = 1.5
):

    gamma_med = median_heuristic_gamma(
        W2
    )

    y_var = max(
        float(np.var(y)),
        1e-8
    )

    x0 = np.log([
        y_var,
        gamma_med,
        0.01 * y_var + 1e-8
    ])

    log_gamma_lo = (
        np.log(gamma_med)
        - gamma_bound_decades * np.log(10)
    )

    log_gamma_hi = (
        np.log(gamma_med)
        + gamma_bound_decades * np.log(10)
    )

    bounds = [
        (None, None),
        (
            log_gamma_lo,
            log_gamma_hi
        ),
        (None, None)
    ]

    result = minimize(
        neg_log_marginal_likelihood,
        x0,
        args=(W2, y),
        method="L-BFGS-B",
        bounds=bounds
    )

    sigma_f2, gamma, sigma_n2 = np.exp(
        result.x
    )

    return {
        "sigma_f2": sigma_f2,
        "gamma": gamma,
        "sigma_n2": sigma_n2,
        "neg_log_lik": result.fun,
        "gamma_median_heuristic": gamma_med
    }


# ============================================================
# WASSERSTEIN GP
# ============================================================

class WassersteinGP:

    def __init__(self):

        self.sigma_f2 = None
        self.gamma = None
        self.sigma_n2 = None

        self.Q_train = None
        self.alpha = None
        self.L = None

        self.y_mean = 0.0

    def fit(
        self,
        quantile_matrix: np.ndarray,
        y: np.ndarray
    ):

        self.y_mean = float(
            np.mean(y)
        )

        y_centered = (
            y - self.y_mean
        )

        W2_train = (
            wasserstein_gram_matrix(
                quantile_matrix
            )
        )

        params = fit_hyperparameters(
            W2_train,
            y_centered
        )

        self.sigma_f2 = (
            params["sigma_f2"]
        )

        self.gamma = (
            params["gamma"]
        )

        self.sigma_n2 = (
            params["sigma_n2"]
        )

        K = (
            self.sigma_f2
            * np.exp(
                -self.gamma
                * W2_train**2
            )
            + self.sigma_n2
            * np.eye(
                len(y)
            )
        )

        self.L = np.linalg.cholesky(
            K
        )

        self.alpha = np.linalg.solve(
            self.L.T,
            np.linalg.solve(
                self.L,
                y_centered
            )
        )

        self.Q_train = (
            quantile_matrix
        )

        self.train_log_lik = (
            -params["neg_log_lik"]
        )

        return self

    def _w2_to_train(
        self,
        q_star: np.ndarray
    ):

        u = np.linspace(
            1e-4,
            1 - 1e-4,
            len(q_star)
        )

        differences = (
            self.Q_train
            - q_star[None, :]
        )

        return np.sqrt(
            np.trapezoid(
                differences**2,
                u,
                axis=1
            )
        )

    def predict(
        self,
        quantile_matrix_test
    ):

        means = []
        stds = []

        for q_star in (
            quantile_matrix_test
        ):

            w2_star = (
                self._w2_to_train(
                    q_star
                )
            )

            k_star = (
                self.sigma_f2
                * np.exp(
                    -self.gamma
                    * w2_star**2
                )
            )

            mean = (
                self.y_mean
                + k_star @ self.alpha
            )

            v = np.linalg.solve(
                self.L,
                k_star
            )

            f_var = (
                self.sigma_f2
                - v @ v
            )

            predictive_var = (
                f_var
                + self.sigma_n2
            )

            predictive_var = max(
                predictive_var,
                1e-12
            )

            means.append(
                mean
            )

            stds.append(
                np.sqrt(
                    predictive_var
                )
            )

        return (
            np.asarray(means),
            np.asarray(stds)
        )


# ============================================================
# FULL RND WASSERSTEIN BARYCENTER FORECASTER
# ============================================================

class WassersteinBarycenterForecaster:

    def __init__(
        self,
        gamma: float
    ):

        self.gamma = gamma

        self.Q_input_train = None
        self.Q_target_train = None

    def fit(
        self,
        Q_input_train,
        Q_target_train
    ):

        self.Q_input_train = (
            Q_input_train
        )

        self.Q_target_train = (
            Q_target_train
        )

        return self

    def predict_quantile(
        self,
        q_star
    ):

        u = np.linspace(
            1e-4,
            1 - 1e-4,
            len(q_star)
        )

        differences = (
            self.Q_input_train
            - q_star[None, :]
        )

        w2 = np.sqrt(
            np.trapezoid(
                differences**2,
                u,
                axis=1
            )
        )

        similarities = np.exp(
            -self.gamma * w2**2
        )

        # Nearest neighbour
        nearest_index = int(
            np.argmin(w2)
        )

        # STABLE NORMALIZATION
        weights = stable_kernel_weights(
            similarities,
            nearest_index=nearest_index
        )

        return (
            weights
            @ self.Q_target_train
        )

    def predict(
        self,
        Q_input_test
    ):

        return np.stack([
            self.predict_quantile(q)
            for q in Q_input_test
        ])


# ============================================================
# LOAD RND DATA
# ============================================================

def load_rnds(
    path: str,
    n_quantiles: int = 200
):

    rnd = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    dates = sorted(
        rnd["date"].unique()
    )

    grids = []
    densities = []
    forwards = []

    for date in dates:

        g = (
            rnd[
                rnd["date"] == date
            ]
            .sort_values("strike")
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

        # IMPORTANT:
        # Forward-normalized coordinate
        x = strike / forward

        grids.append(
            x
        )

        densities.append(
            density
        )

        forwards.append(
            forward
        )

    Q = np.stack([
        density_to_quantile(
            x,
            d,
            n_quantiles
        )
        for x, d in zip(
            grids,
            densities
        )
    ])

    return (
        dates,
        Q,
        forwards,
        grids,
        densities
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("WASSERSTEIN GP — SPX 30D RND")
    print("=" * 70)

    (
        dates,
        Q,
        forwards,
        grids,
        densities
    ) = load_rnds(
        RND_FILE,
        n_quantiles=200
    )

    print(
        f"RND days : {len(dates)}"
    )

    print(
        f"Quantiles: {Q.shape}"
    )

    # --------------------------------------------------------
    # INPUT / TARGET
    # --------------------------------------------------------

    # Input = RND_t
    # Target = RND_{t+1}

    X_Q = Q[:-1]

    Q_target = Q[1:]

    target_dates = dates[1:]

    n = len(X_Q)

    if n < 5:

        raise ValueError(
            "Too few observations for GP."
        )

    # 80/20 chronological split
    n_train = max(
        3,
        int(
            n * 0.8
        )
    )

    if n_train >= n:

        n_train = n - 1

    Q_train = X_Q[
        :n_train
    ]

    Q_test = X_Q[
        n_train:
    ]

    Q_target_train = (
        Q_target[
            :n_train
        ]
    )

    Q_target_test = (
        Q_target[
            n_train:
        ]
    )

    test_dates = target_dates[
        n_train:
    ]

    print(
        f"Training days : {n_train}"
    )

    print(
        f"Test days     : {len(Q_test)}"
    )

    # --------------------------------------------------------
    # WASSERSTEIN GP — SCALAR VARIANCE TARGET
    # --------------------------------------------------------

    y_all = np.array([
        rnd_variance(
            grids[i],
            densities[i]
        )
        for i in range(
            len(dates)
        )
    ])

    y_input = y_all[:-1]
    y_target = y_all[1:]

    y_train = y_target[
        :n_train
    ]

    y_test = y_target[
        n_train:
    ]

    gp = (
        WassersteinGP()
        .fit(
            Q_train,
            y_train
        )
    )

    mean_pred, std_pred = (
        gp.predict(
            Q_test
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                mean_pred
                - y_test
            ) ** 2
        )
    )

    z = (
        y_test
        - mean_pred
    ) / std_pred

    coverage_95 = np.mean(
        np.abs(z) <= 1.96
    )

    print()
    print("SCALAR GP")
    print("-" * 70)

    print(
        f"sigma_f^2 : "
        f"{gp.sigma_f2:.6g}"
    )

    print(
        f"gamma     : "
        f"{gp.gamma:.6g}"
    )

    print(
        f"sigma_n^2 : "
        f"{gp.sigma_n2:.6g}"
    )

    print(
        f"RMSE      : "
        f"{rmse:.6g}"
    )

    print(
        f"95% cover : "
        f"{coverage_95:.2%}"
    )

    # --------------------------------------------------------
    # FULL DISTRIBUTION FORECAST
    # --------------------------------------------------------

    # Use median-heuristic gamma
    W2_train = (
        wasserstein_gram_matrix(
            Q_train
        )
    )

    bary_gamma = (
        median_heuristic_gamma(
            W2_train
        )
    )

    bary = (
        WassersteinBarycenterForecaster(
            bary_gamma
        )
    )

    bary.fit(
        Q_train,
        Q_target_train
    )

    Q_pred = bary.predict(
        Q_test
    )

    # W2 forecast errors
    bary_errors = []

    for pred, actual in zip(
        Q_pred,
        Q_target_test
    ):

        error = (
            np.sqrt(
                np.trapezoid(
                    (pred - actual) ** 2,
                    np.linspace(
                        1e-4,
                        1 - 1e-4,
                        len(pred)
                    )
                )
            )
        )

        bary_errors.append(
            error
        )

    bary_errors = np.asarray(
        bary_errors
    )

    # --------------------------------------------------------
    # PERSISTENCE BASELINE
    # --------------------------------------------------------

    persistence_errors = []

    for q_input, q_actual in zip(
        Q_test,
        Q_target_test
    ):

        error = (
            np.sqrt(
                np.trapezoid(
                    (q_input - q_actual) ** 2,
                    np.linspace(
                        1e-4,
                        1 - 1e-4,
                        len(q_input)
                    )
                )
            )
        )

        persistence_errors.append(
            error
        )

    persistence_errors = np.asarray(
        persistence_errors
    )

    # --------------------------------------------------------
    # SAVE FULL DISTRIBUTION RESULTS
    # --------------------------------------------------------

    rows = []

    for i, date in enumerate(
        test_dates
    ):

        rows.append({
            "date": date,
            "wasserstein_gp_w2":
                bary_errors[i],
            "persistence_w2":
                persistence_errors[i]
        })

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        os.path.join(
            RESULT_DIR,
            "wasserstein_forecast_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("FULL RND FORECAST")
    print("-" * 70)

    print(
        f"Barycenter gamma : "
        f"{bary_gamma:.6g}"
    )

    print(
        f"Wasserstein GP W2 "
        f"mean error       : "
        f"{bary_errors.mean():.6f}"
    )

    print(
        f"Persistence W2 "
        f"mean error       : "
        f"{persistence_errors.mean():.6f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        os.path.join(
            RESULT_DIR,
            "wasserstein_forecast_comparison.csv"
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    main()