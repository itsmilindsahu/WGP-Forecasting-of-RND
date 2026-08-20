"""
Day 6 — Evaluate: empirical coverage, NLPD, Wasserstein forecast error,
Wasserstein-GP vs the Day 5 moment-based linear baseline.

Three metrics, each computed for both models on the same held-out days:

1. Empirical coverage of the (nominal 95%) prediction interval for the
   scalar target Var(RND_{t+1}).
2. Negative log-predictive density (NLPD) of the realized scalar target
   under each model's Gaussian predictive.
3. Wasserstein-2 forecast error: W2 distance between the *full* predicted
   next-day RND and the *full* realized next-day RND.
   - GP forecast: Nadaraya-Watson quantile barycenter (day4/fit_gp.py,
     WassersteinBarycenterForecaster), using the same kernel/gamma as the
     scalar-target GP.
   - Baseline forecast: moment-matched Gaussian using the linear baseline's
     predicted next-day mean & variance (mean held fixed at today's realized
     mean, since Day 5's baseline only forecasts variance — this is stated
     explicitly rather than silently assumed).

This file assumes day1-5 have already been run (run_all.py does that).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "day2_rnd_extraction"))
sys.path.append(os.path.join(ROOT, "day3_wasserstein_kernel"))
sys.path.append(os.path.join(ROOT, "day4_gp_model"))
sys.path.append(os.path.join(ROOT, "day5_baseline"))

from wasserstein_kernel import density_to_quantile  # noqa: E402
from fit_gp import WassersteinGP, WassersteinBarycenterForecaster, rnd_variance  # noqa: E402
from linear_baseline import MomentLinearBaseline, rnd_moments, gaussian_density_on_grid  # noqa: E402
from persistence_baseline import PersistenceForecaster, persistence_quantile_forecast  # noqa: E402


def gaussian_nlpd(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-point negative log density under N(mean, std^2)."""
    return 0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((y_true - mean) / std) ** 2


def empirical_coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.96) -> float:
    return float(np.mean(np.abs(y_true - mean) <= z * std))


def run_evaluation(rnd_path: str, n_quantiles: int = 200):
    rnd = pd.read_csv(rnd_path, parse_dates=["date"])
    dates = sorted(rnd["date"].unique())

    grids, densities = [], []
    for d in dates:
        g = rnd[rnd["date"] == d].sort_values("strike")
        grids.append(g["strike"].to_numpy())
        densities.append(g["density"].to_numpy())

    Q = np.stack([density_to_quantile(g, d, n_quantiles) for g, d in zip(grids, densities)])
    y_all = np.array([rnd_variance(g, d) for g, d in zip(grids, densities)])
    moments_all = np.array([rnd_moments(g, d) for g, d in zip(grids, densities)])

    n = len(dates) - 1
    n_train = int(n * 0.8)

    # ---- scalar target split (RND_t -> Var(RND_{t+1})) ----
    X_Q, y = Q[:-1], y_all[1:]
    X_moments = moments_all[:-1]
    y_test = y[n_train:]

    # ---- GP (Day 4) ----
    gp = WassersteinGP().fit(X_Q[:n_train], y[:n_train])
    gp_mean, gp_std = gp.predict(X_Q[n_train:])

    # ---- Baseline (Day 5) ----
    baseline = MomentLinearBaseline().fit(X_moments[:n_train], y[:n_train])
    bl_mean, bl_std = baseline.predict(X_moments[n_train:])

    # ---- Persistence (Day 5b) ---- the model to beat, per Sven's review:
    # y_hat_{t+1} = y_t, i.e. today's realized variance, unchanged.
    y_persistence = y_all[:-1]  # aligned with y = y_all[1:], same as X_Q/X_moments
    persistence = PersistenceForecaster().fit(y_persistence[:n_train], y[:n_train])
    pers_mean, pers_std = persistence.predict(y_persistence[n_train:])

    # ---- scalar metrics ----
    metrics = {
        "gp_rmse": float(np.sqrt(np.mean((gp_mean - y_test) ** 2))),
        "baseline_rmse": float(np.sqrt(np.mean((bl_mean - y_test) ** 2))),
        "persistence_rmse": float(np.sqrt(np.mean((pers_mean - y_test) ** 2))),
        "gp_coverage_95": empirical_coverage(y_test, gp_mean, gp_std),
        "baseline_coverage_95": empirical_coverage(y_test, bl_mean, bl_std),
        "persistence_coverage_95": empirical_coverage(y_test, pers_mean, pers_std),
        "gp_nlpd_mean": float(np.mean(gaussian_nlpd(y_test, gp_mean, gp_std))),
        "baseline_nlpd_mean": float(np.mean(gaussian_nlpd(y_test, bl_mean, bl_std))),
        "persistence_nlpd_mean": float(np.mean(gaussian_nlpd(y_test, pers_mean, pers_std))),
    }

    # ---- full-density (Wasserstein) forecast error ----
    Q_input_train = Q[:-1][:n_train]     # RND_t,  t = 0..n_train-1
    Q_target_train = Q[1:][:n_train]     # RND_{t+1} for the same t's
    Q_input_test = Q[:-1][n_train:]      # RND_t for held-out t's
    Q_target_test = Q[1:][n_train:]      # realized RND_{t+1}, held out

    bary = WassersteinBarycenterForecaster(gamma=gp.gamma).fit(Q_input_train, Q_target_train)
    Q_gp_forecast = bary.predict(Q_input_test)

    # persistence full-density forecast (Day 5b): tomorrow's quantile
    # function predicted to equal today's, verbatim. No fitting.
    Q_persistence_forecast = persistence_quantile_forecast(Q_input_test)

    # baseline full-density forecast: moment-matched Gaussian using baseline's
    # predicted next-day variance and *today's realized mean* (Day 5 doesn't
    # forecast the mean, so this is the honest, explicitly-stated choice)
    u = np.linspace(1e-4, 1 - 1e-4, n_quantiles)
    Q_baseline_forecast = []
    # no-forecast diagnostic (per Sven's review): Gaussian fitted to *today's*
    # own realized mean/variance, scored against *tomorrow's* true density.
    # This involves zero forecasting -- it isolates the cost of imposing a
    # Gaussian shape from any actual forecast skill, and is what the
    # GP-vs-baseline full-density gap must be checked against before it's
    # interpreted as a modeling win.
    Q_noforecast_gaussian = []
    test_idx = np.arange(n_train, n)  # index into moments_all / grids for t (0-based over n = len(dates)-1)
    for i, t in enumerate(test_idx):
        mean_t, var_t = moments_all[t][0], moments_all[t][1]  # today's realized mean & var, held fixed
        var_pred = bl_mean[i]
        x_grid = grids[t + 1]  # reuse next day's strike support for evaluation grid

        dens_bl = gaussian_density_on_grid(mean_t, var_pred, x_grid)
        Q_baseline_forecast.append(density_to_quantile(x_grid, dens_bl, n_quantiles))

        dens_nf = gaussian_density_on_grid(mean_t, var_t, x_grid)
        Q_noforecast_gaussian.append(density_to_quantile(x_grid, dens_nf, n_quantiles))

    Q_baseline_forecast = np.stack(Q_baseline_forecast)
    Q_noforecast_gaussian = np.stack(Q_noforecast_gaussian)

    def w2_rowwise(A, B):
        return np.sqrt(np.trapezoid((A - B) ** 2, u, axis=1))

    def w2_rowwise_shape_only(A, B):
        """W2 after removing each side's own mean -- isolates shape error
        from the mean-shift component (per Sven's review: on this data W2 is
        ~95% mean shift, so the raw number barely tests the Wasserstein
        kernel's actual selling point). Reported alongside the raw W2, not
        as a replacement -- a full re-derivation needs real data (Day 4 of
        Sven's list) and is left as a next step."""
        A_c = A - A.mean(axis=1, keepdims=True)
        B_c = B - B.mean(axis=1, keepdims=True)
        return np.sqrt(np.trapezoid((A_c - B_c) ** 2, u, axis=1))

    w2_gp = w2_rowwise(Q_gp_forecast, Q_target_test)
    w2_baseline = w2_rowwise(Q_baseline_forecast, Q_target_test)
    w2_persistence = w2_rowwise(Q_persistence_forecast, Q_target_test)
    w2_noforecast = w2_rowwise(Q_noforecast_gaussian, Q_target_test)

    w2_gp_shape = w2_rowwise_shape_only(Q_gp_forecast, Q_target_test)
    w2_baseline_shape = w2_rowwise_shape_only(Q_baseline_forecast, Q_target_test)
    w2_persistence_shape = w2_rowwise_shape_only(Q_persistence_forecast, Q_target_test)

    metrics["gp_w2_forecast_error_mean"] = float(np.mean(w2_gp))
    metrics["baseline_w2_forecast_error_mean"] = float(np.mean(w2_baseline))
    metrics["persistence_w2_forecast_error_mean"] = float(np.mean(w2_persistence))
    metrics["noforecast_gaussian_w2_mean"] = float(np.mean(w2_noforecast))
    metrics["gp_w2_shape_only_mean"] = float(np.mean(w2_gp_shape))
    metrics["baseline_w2_shape_only_mean"] = float(np.mean(w2_baseline_shape))
    metrics["persistence_w2_shape_only_mean"] = float(np.mean(w2_persistence_shape))

    per_day = pd.DataFrame(
        {
            "date": [dates[t + 1] for t in test_idx],  # date of the forecasted (t+1) RND
            "y_true_var": y_test,
            "gp_mean": gp_mean,
            "gp_std": gp_std,
            "baseline_mean": bl_mean,
            "baseline_std": bl_std,
            "persistence_mean": pers_mean,
            "persistence_std": pers_std,
            "gp_w2_error": w2_gp,
            "baseline_w2_error": w2_baseline,
            "persistence_w2_error": w2_persistence,
            "noforecast_gaussian_w2_error": w2_noforecast,
        }
    )

    return metrics, per_day


def print_comparison_table(metrics: dict):
    print(f"{'Metric':<28}{'Wasserstein-GP':>18}{'Linear baseline':>18}{'Persistence':>16}")
    print("-" * 80)
    rows = [
        ("RMSE (variance target)", metrics["gp_rmse"], metrics["baseline_rmse"], metrics["persistence_rmse"]),
        (
            "95% coverage",
            metrics["gp_coverage_95"],
            metrics["baseline_coverage_95"],
            metrics["persistence_coverage_95"],
        ),
        (
            "NLPD (mean, lower=better)",
            metrics["gp_nlpd_mean"],
            metrics["baseline_nlpd_mean"],
            metrics["persistence_nlpd_mean"],
        ),
        (
            "W2 forecast error (mean)",
            metrics["gp_w2_forecast_error_mean"],
            metrics["baseline_w2_forecast_error_mean"],
            metrics["persistence_w2_forecast_error_mean"],
        ),
    ]
    for name, gp_v, bl_v, pers_v in rows:
        print(f"{name:<28}{gp_v:>18.4f}{bl_v:>18.4f}{pers_v:>16.4f}")

    print()
    print(
        f"  {'Gaussian fit to today (no forecast at all), W2 vs tomorrow':<58}"
        f"{metrics['noforecast_gaussian_w2_mean']:>10.4f}"
    )
    print(
        "  ^ This uses zero forecast information (today's own realized mean/var, scored\n"
        "    against tomorrow). If GP/baseline W2 are close to this number, the model isn't\n"
        "    doing forecasting -- it's just the cost of assuming a Gaussian shape."
    )
    print()
    print("  Shape-only W2 (each side's own mean removed before comparing):")
    print(
        f"    {'Wasserstein-GP':<20}{metrics['gp_w2_shape_only_mean']:>10.4f}   "
        f"{'Linear baseline':<20}{metrics['baseline_w2_shape_only_mean']:>10.4f}   "
        f"{'Persistence':<14}{metrics['persistence_w2_shape_only_mean']:>10.4f}"
    )


def make_comparison_plot(metrics: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Coverage (95%)", "NLPD (mean)", "W2 forecast error"]
    gp_vals = [metrics["gp_coverage_95"], metrics["gp_nlpd_mean"], metrics["gp_w2_forecast_error_mean"]]
    bl_vals = [
        metrics["baseline_coverage_95"],
        metrics["baseline_nlpd_mean"],
        metrics["baseline_w2_forecast_error_mean"],
    ]
    pers_vals = [
        metrics["persistence_coverage_95"],
        metrics["persistence_nlpd_mean"],
        metrics["persistence_w2_forecast_error_mean"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, label, gv, bv, pv in zip(axes, labels, gp_vals, bl_vals, pers_vals):
        bars = ax.bar(
            ["Wasserstein-GP", "Linear baseline", "Persistence"],
            [gv, bv, pv],
            color=["#3b6fb6", "#b6653b", "#5a5a5a"],
        )
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, fmt="%.3f")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Day 6: Wasserstein-GP vs moment-based linear baseline vs persistence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    metrics, per_day = run_evaluation(os.path.join(ROOT, "day2_rnd_extraction/data/rnd_curves.csv"))
    print_comparison_table(metrics)

    per_day.to_csv("results/comparison_per_day.csv", index=False)
    pd.Series(metrics).to_csv("results/comparison_summary.csv")
    make_comparison_plot(metrics, "results/comparison_plot.png")
    print("\nWrote results/comparison_per_day.csv, results/comparison_summary.csv, results/comparison_plot.png")
