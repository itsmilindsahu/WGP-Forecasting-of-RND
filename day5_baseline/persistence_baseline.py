"""
Day 5b — Persistence benchmark (per Sven's review, 2026).

The natural null model for any daily forecasting problem is the random walk:
predict tomorrow = today, unchanged. Neither the Wasserstein-GP nor the
moment-based linear baseline means anything until both are shown to beat
this. Two versions, matching the two targets used elsewhere in the pipeline:

1. Scalar target (Var(RND_{t+1})): y_hat_{t+1} = Var(RND_t). Predictive std
   is the (homoscedastic) training-residual std, exactly as for the linear
   baseline, so coverage/NLPD are computed the same way for all three models.
2. Full-density target (RND_{t+1}): Q_hat_{t+1} = Q_t, i.e. today's quantile
   function is used verbatim as tomorrow's forecast. No fitting is involved.
"""
from __future__ import annotations
import numpy as np


class PersistenceForecaster:
    """Scalar persistence: y_hat_{t+1} = y_t. Std is the homoscedastic
    training-residual std of the persistence forecast, so it plugs into the
    same coverage/NLPD machinery as the GP and the linear baseline."""

    def __init__(self):
        self.resid_std = None

    def fit(self, y_persistence_train: np.ndarray, y_train: np.ndarray):
        resid = y_train - y_persistence_train
        ddof = 1 if len(resid) > 1 else 0
        self.resid_std = float(np.std(resid, ddof=ddof)) + 1e-12
        return self

    def predict(self, y_persistence_test: np.ndarray):
        mean_pred = np.asarray(y_persistence_test, dtype=float)
        std_pred = np.full(len(mean_pred), self.resid_std)
        return mean_pred, std_pred


def persistence_quantile_forecast(Q_input_test: np.ndarray) -> np.ndarray:
    """Full-density persistence forecast: tomorrow's quantile function is
    predicted to be identical to today's. No fitting needed."""
    return np.asarray(Q_input_test, dtype=float).copy()


if __name__ == "__main__":
    import os
    import sys
    import pandas as pd

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day4_gp_model"))
    from fit_gp import rnd_variance  # noqa: E402

    rnd = pd.read_csv("../day2_rnd_extraction/data/rnd_curves.csv", parse_dates=["date"])
    dates = sorted(rnd["date"].unique())

    grids, densities = [], []
    for d in dates:
        g = rnd[rnd["date"] == d].sort_values("strike")
        grids.append(g["strike"].to_numpy())
        densities.append(g["density"].to_numpy())

    y_all = np.array([rnd_variance(g, d) for g, d in zip(grids, densities)])
    y_persistence_all = y_all[:-1]
    y = y_all[1:]

    n = len(y)
    n_train = int(n * 0.8)
    model = PersistenceForecaster().fit(y_persistence_all[:n_train], y[:n_train])
    mean_pred, std_pred = model.predict(y_persistence_all[n_train:])
    y_test = y[n_train:]

    rmse = np.sqrt(np.mean((mean_pred - y_test) ** 2))
    print(f"Persistence held-out RMSE: {rmse:.6f}")

    out = pd.DataFrame(
        {"date": dates[n_train + 1 : n + 1], "y_true": y_test, "y_pred_mean": mean_pred, "y_pred_std": std_pred}
    )
    out.to_csv("results/persistence_predictions.csv", index=False)
    print("Wrote results/persistence_predictions.csv")
