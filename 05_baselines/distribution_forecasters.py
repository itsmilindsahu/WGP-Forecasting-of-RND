from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_gp_model"))
from fit_gp import load_rnds

RND_FILE = ROOT / "02_rnd_extraction" / "data" / "rnd_30d.csv"
RESULT_DIR = ROOT / "05_baselines" / "results"


def quantile_moments(quantiles: np.ndarray) -> np.ndarray:
    u = np.linspace(1e-4, 1 - 1e-4, quantiles.shape[1])
    means = np.trapezoid(quantiles, u, axis=1) / (u[-1] - u[0])
    centered = quantiles - means[:, None]
    variances = np.trapezoid(centered**2, u, axis=1) / (u[-1] - u[0])
    stds = np.sqrt(np.maximum(variances, 1e-16))
    skewness = np.trapezoid(centered**3, u, axis=1) / (u[-1] - u[0]) / stds**3
    return np.column_stack((means, variances, skewness))


class WassersteinAutoregression:
    """Pooled first-order linear autoregression on quantile coordinates."""

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> WassersteinAutoregression:
        input_values = np.asarray(inputs, dtype=float).ravel()
        target_values = np.asarray(targets, dtype=float).ravel()
        input_mean = float(input_values.mean())
        target_mean = float(target_values.mean())
        denominator = float(np.sum((input_values - input_mean) ** 2))
        slope = (
            float(np.sum((input_values - input_mean) * (target_values - target_mean)) / denominator)
            if denominator > 0
            else 0.0
        )
        self.slope = max(slope, 0.0)
        self.intercept = target_mean - self.slope * input_mean
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        predicted = self.intercept + self.slope * np.asarray(inputs, dtype=float)
        return np.maximum.accumulate(np.maximum(predicted, 0.0), axis=1)


class MomentGaussianProcess:
    """Exact multi-output RBF GP from distribution moments to target quantiles."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> MomentGaussianProcess:
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        self.x_mean = features.mean(axis=0)
        self.x_scale = features.std(axis=0)
        self.x_scale[self.x_scale < 1e-12] = 1.0
        self.x_train = (features - self.x_mean) / self.x_scale

        self.y_mean = targets.mean(axis=0)
        self.y_scale = targets.std(axis=0)
        self.y_scale[self.y_scale < 1e-10] = 1.0
        self.y_train = (targets - self.y_mean) / self.y_scale

        def kernel(left: np.ndarray, right: np.ndarray, params: np.ndarray) -> np.ndarray:
            signal_variance = np.exp(params[0])
            length_scales = np.exp(params[1:4])
            differences = (left[:, None, :] - right[None, :, :]) / length_scales
            return signal_variance * np.exp(-0.5 * np.sum(differences**2, axis=2))

        def objective(params: np.ndarray) -> float:
            covariance = kernel(self.x_train, self.x_train, params)
            covariance += np.exp(params[4]) * np.eye(len(self.x_train))
            try:
                cholesky = np.linalg.cholesky(covariance)
                alpha = np.linalg.solve(
                    cholesky.T,
                    np.linalg.solve(cholesky, self.y_train),
                )
            except np.linalg.LinAlgError:
                return 1e12
            output_count = self.y_train.shape[1]
            log_determinant = output_count * np.log(np.diag(cholesky)).sum()
            return float(
                0.5 * np.sum(self.y_train * alpha)
                + log_determinant
                + 0.5 * len(self.x_train) * output_count * np.log(2 * np.pi)
            )

        bounds = [(-9.0, 5.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-12.0, 0.0)]
        result = minimize(
            objective,
            np.array([0.0, 0.0, 0.0, 0.0, -3.0]),
            method="L-BFGS-B",
            bounds=bounds,
        )
        self.params = result.x
        self.signal_variance = float(np.exp(self.params[0]))
        self.length_scales = np.exp(self.params[1:4])
        self.noise_variance = float(np.exp(self.params[4]))

        covariance = kernel(self.x_train, self.x_train, self.params)
        covariance += self.noise_variance * np.eye(len(self.x_train))
        self.cholesky = np.linalg.cholesky(covariance)
        self.alpha = np.linalg.solve(
            self.cholesky.T,
            np.linalg.solve(self.cholesky, self.y_train),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=float) - self.x_mean) / self.x_scale
        differences = (standardized[:, None, :] - self.x_train[None, :, :]) / self.length_scales
        cross_covariance = self.signal_variance * np.exp(-0.5 * np.sum(differences**2, axis=2))
        predicted = self.y_mean + (cross_covariance @ self.alpha) * self.y_scale
        return np.maximum.accumulate(np.maximum(predicted, 0.0), axis=1)


def w2_errors(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    u = np.linspace(1e-4, 1 - 1e-4, predictions.shape[1])
    return np.sqrt(np.trapezoid((predictions - targets) ** 2, u, axis=1))


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates, quantiles, _, _, _ = load_rnds(str(RND_FILE), n_quantiles=200)
    inputs = quantiles[:-1]
    targets = quantiles[1:]
    test_dates = dates[1:]
    train_count = max(3, int(len(inputs) * 0.8))
    if train_count >= len(inputs):
        train_count = len(inputs) - 1

    ar = WassersteinAutoregression().fit(inputs[:train_count], targets[:train_count])
    ar_predictions = ar.predict(inputs[train_count:])

    features = quantile_moments(inputs)
    moment_gp = MomentGaussianProcess().fit(features[:train_count], targets[:train_count])
    gp_predictions = moment_gp.predict(features[train_count:])
    actual = targets[train_count:]
    evaluation_dates = test_dates[train_count:]

    per_day = pd.DataFrame({
        "date": evaluation_dates,
        "wasserstein_ar_w2": w2_errors(ar_predictions, actual),
        "moment_gp_w2": w2_errors(gp_predictions, actual),
    })
    per_day["wasserstein_ar_slope"] = ar.slope
    per_day["moment_gp_signal_variance"] = moment_gp.signal_variance
    per_day["moment_gp_noise_variance"] = moment_gp.noise_variance

    summary = per_day[["wasserstein_ar_w2", "moment_gp_w2"]].agg(["mean", "median", "std"]).T
    summary = summary.rename(columns={"mean": "mean_w2", "median": "median_w2", "std": "std_w2"})
    summary.index.name = "model"
    summary = summary.reset_index()
    summary["model"] = summary["model"].map({
        "wasserstein_ar_w2": "Wasserstein autoregression",
        "moment_gp_w2": "Conventional GP (moments)",
    })
    return per_day, summary


if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    per_day, summary = run()
    per_day.to_csv(RESULT_DIR / "distribution_forecast_predictions.csv", index=False)
    summary.to_csv(RESULT_DIR / "distribution_forecast_summary.csv", index=False)
    print(per_day.to_string(index=False))
    print(summary.to_string(index=False))