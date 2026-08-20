"""
Day 5 — Moment-based linear baseline.

Instead of treating each day's RND as a distribution-valued GP input, this
baseline collapses it to its first few moments (mean, variance, skewness) and
fits a linear AR-type model on those moments to forecast next-day variance
(the same scalar target used by the Day 4 Wasserstein-GP, so the two are
directly comparable in Day 6).

Two things are produced, both needed for Day 6:
1. A scalar forecast of Var(RND_{t+1}) with a homoscedastic Gaussian
   predictive (mean = linear prediction, std = training-residual std) --
   this is what NLPD / coverage are computed against.
2. A full baseline *density* forecast: a moment-matched Gaussian using the
   predicted next-day mean and variance. This stands in for "the baseline's
   guess at tomorrow's whole RND" and is what the Wasserstein forecast error
   in Day 6 is measured against, as the counterpart to the GP's quantile-
   barycenter density forecast.
"""
from __future__ import annotations
import numpy as np


def rnd_moments(strike: np.ndarray, density: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, variance, skewness) of a density given on a strike grid."""
    mean = np.trapezoid(strike * density, strike)
    var = np.trapezoid((strike - mean) ** 2 * density, strike)
    std = np.sqrt(max(var, 1e-12))
    skew = np.trapezoid(((strike - mean) / std) ** 3 * density, strike)
    return float(mean), float(var), float(skew)


class MomentLinearBaseline:
    """
    Linear regression: [mean_t, var_t, skew_t, 1] -> var_{t+1}.
    Predictive std is the (homoscedastic) residual std on the training set,
    which is the natural, minimal-assumption analogue of the GP's posterior
    std for a model that isn't Bayesian by construction.
    """

    def __init__(self):
        self.beta = None
        self.resid_std = None
        self.feat_mean = None
        self.feat_std = None

    @staticmethod
    def _features(moments: np.ndarray) -> np.ndarray:
        # moments: (n, 3) = [mean, var, skew]
        return moments  # keep raw; standardization handled separately

    def fit(self, moments_t: np.ndarray, y_next_var: np.ndarray):
        X = self._features(moments_t)
        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0) + 1e-8
        Xs = (X - self.feat_mean) / self.feat_std
        Xs = np.hstack([Xs, np.ones((len(Xs), 1))])  # intercept

        # ordinary least squares via lstsq (stable, no external deps)
        self.beta, *_ = np.linalg.lstsq(Xs, y_next_var, rcond=None)

        resid = y_next_var - Xs @ self.beta
        self.resid_std = float(np.std(resid, ddof=Xs.shape[1]))
        return self

    def predict(self, moments_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._features(moments_t)
        Xs = (X - self.feat_mean) / self.feat_std
        Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
        mean_pred = Xs @ self.beta
        std_pred = np.full(len(mean_pred), self.resid_std)
        return mean_pred, std_pred


def gaussian_density_on_grid(mean: float, var: float, x_grid: np.ndarray) -> np.ndarray:
    """Moment-matched Gaussian density forecast, evaluated on x_grid."""
    std = np.sqrt(max(var, 1e-8))
    return np.exp(-0.5 * ((x_grid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


if __name__ == "__main__":
    import pandas as pd
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day4_gp_model"))
    from fit_gp import rnd_variance  # noqa: E402

    rnd = pd.read_csv("../day2_rnd_extraction/data/rnd_curves.csv", parse_dates=["date"])
    dates = sorted(rnd["date"].unique())

    grids, densities = [], []
    for d in dates:
        g = rnd[rnd["date"] == d].sort_values("strike")
        grids.append(g["strike"].to_numpy())
        densities.append(g["density"].to_numpy())

    moments = np.array([rnd_moments(g, d) for g, d in zip(grids, densities)])
    y_all = np.array([rnd_variance(g, d) for g, d in zip(grids, densities)])

    X_moments = moments[:-1]
    y = y_all[1:]

    n = len(y)
    n_train = int(n * 0.8)
    model = MomentLinearBaseline().fit(X_moments[:n_train], y[:n_train])
    mean_pred, std_pred = model.predict(X_moments[n_train:])
    y_test = y[n_train:]

    rmse = np.sqrt(np.mean((mean_pred - y_test) ** 2))
    print(f"Baseline held-out RMSE: {rmse:.6f}")

    out = pd.DataFrame(
        {"date": dates[n_train + 1 : n + 1], "y_true": y_test, "y_pred_mean": mean_pred, "y_pred_std": std_pred}
    )
    out.to_csv("results/baseline_predictions.csv", index=False)
    print("Wrote results/baseline_predictions.csv")
