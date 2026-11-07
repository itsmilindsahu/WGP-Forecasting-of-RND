from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_gp_model"))
sys.path.insert(0, str(ROOT / "05_baselines"))

from fit_gp import (  # noqa: E402
    WassersteinBarycenterForecaster,
    WassersteinGP,
    load_rnds,
    rnd_variance,
)
from distribution_forecasters import (  # noqa: E402
    MomentGaussianProcess,
    WassersteinAutoregression,
    quantile_moments,
    w2_errors,
)
from trading_calendar_pairs import build_trading_pairs, nyse_sessions  # noqa: E402


RND_FILE = ROOT / "02_rnd_extraction" / "data" / "rnd_30d.csv"
RESULT_DIR = ROOT / "06_evaluation" / "results"
PAIRS_FILE = RESULT_DIR / "trading_calendar_pairs.csv"
PREDICTIONS_FILE = RESULT_DIR / "horizon_forecast_predictions.csv"
SUMMARY_FILE = RESULT_DIR / "horizon_forecast_summary.csv"
REPORT_FILE = RESULT_DIR / "pipeline_failure_report.txt"
DATA_DIR = ROOT / "01_data_collection" / "data"
HORIZONS = (1, 5)
TRAIN_FRACTION = 0.8
N_QUANTILES = 200
GAUSSIAN_MOMENT_BASELINE = "Gaussian-shape moment baseline (reference)"
GAUSSIAN_TRUE_MOMENTS_REFERENCE = "Gaussian with true moments (diagnostic only)"


def _w2_errors(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    u = np.linspace(1e-4, 1 - 1e-4, predictions.shape[1])
    return np.sqrt(np.trapezoid((predictions - targets) ** 2, u, axis=1))


def _linear_moment_forecast(
    inputs: np.ndarray,
    targets: np.ndarray,
    test_inputs: np.ndarray,
) -> np.ndarray:
    input_features = quantile_moments(inputs)
    target_variances = quantile_moments(targets)[:, 1]
    design = np.column_stack((np.ones(len(input_features)), input_features))
    coefficients = np.linalg.lstsq(design, target_variances, rcond=None)[0]

    test_features = quantile_moments(test_inputs)
    test_design = np.column_stack((np.ones(len(test_features)), test_features))
    predicted_variance = np.maximum(test_design @ coefficients, 1e-12)
    u = np.linspace(1e-4, 1 - 1e-4, inputs.shape[1])
    return (
        test_features[:, 0, None]
        + np.sqrt(predicted_variance)[:, None] * norm.ppf(u)[None, :]
    )


def _fit_predict_models(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    train_target_variances: np.ndarray,
    test_inputs: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[str]]:
    predictions: dict[str, np.ndarray] = {
        "Persistence": test_inputs.copy(),
    }
    diagnostics = []

    gp = WassersteinGP().fit(train_inputs, train_target_variances)
    if not gp.hyperparameter_success:
        diagnostics.append(
            f"OPTIMIZER WARNING model=Wasserstein GP: "
            f"{gp.hyperparameter_message}"
        )
    gp.predict(test_inputs)
    barycenter = WassersteinBarycenterForecaster(gp.gamma).fit(
        train_inputs,
        train_targets,
    )
    predictions["Wasserstein GP"] = barycenter.predict(test_inputs)

    predictions[GAUSSIAN_MOMENT_BASELINE] = _linear_moment_forecast(
        train_inputs,
        train_targets,
        test_inputs,
    )

    autoregression = WassersteinAutoregression().fit(
        train_inputs,
        train_targets,
    )
    predictions["Wasserstein autoregression"] = autoregression.predict(
        test_inputs
    )

    moment_gp = MomentGaussianProcess().fit(
        quantile_moments(train_inputs),
        train_targets,
    )
    if not moment_gp.optimization_success:
        diagnostics.append(
            "OPTIMIZER WARNING model=Conventional GP (moments): "
            f"{moment_gp.optimization_message}"
        )
    predictions["Conventional GP (moments)"] = moment_gp.predict(
        quantile_moments(test_inputs)
    )
    return predictions, diagnostics


def _append_stage_diagnostics(report_lines: list[str]) -> None:
    report_lines.append("\nUPSTREAM PIPELINE DIAGNOSTICS")
    diagnostic_files = (
        (
            DATA_DIR / "spx_collection_diagnostics.csv",
            lambda row: str(row.get("status", "")) == "error",
            "collection error",
        ),
        (
            DATA_DIR / "spx_collection_retry_diagnostics.csv",
            lambda row: True,
            "collection retry",
        ),
        (
            DATA_DIR / "spx_iv_diagnostics.csv",
            lambda row: str(row.get("status", "")) in {"error", "warning"},
            "IV diagnostic",
        ),
        (
            DATA_DIR / "spx_30d_smile_diagnostics.csv",
            lambda row: not str(row.get("status", "")).startswith("ok"),
            "30D smile failure",
        ),
        (
            DATA_DIR / "svi_30d_params.csv",
            lambda row: str(row.get("success", "")).lower() in {"false", "0"},
            "SVI fit failure",
        ),
        (
            ROOT / "02_rnd_extraction" / "data" / "rnd_30d_diagnostics.csv",
            lambda row: (
                str(row.get("status", "")) == "error"
                or any(
                    str(row.get(column, "")).lower() in {"false", "0"}
                    for column in (
                        "density_nonnegative",
                        "mass_pass",
                        "martingale_pass",
                        "call_convexity_pass",
                        "call_convexity_pre_projection_pass",
                    )
                )
            ),
            "RND numerical issue",
        ),
    )

    for path, is_issue, label in diagnostic_files:
        if not path.exists():
            report_lines.append(f"NOT RUN: {label} source missing ({path.name})")
            continue
        try:
            diagnostics = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            report_lines.append(
                f"OK: no {label} entries in empty {path.name}"
            )
            continue
        issues = [row for row in diagnostics.to_dict("records") if is_issue(row)]
        if not issues:
            report_lines.append(f"OK: no {label} entries in {path.name}")
            continue
        for issue in issues:
            details = "; ".join(
                f"{key}={value}"
                for key, value in issue.items()
                if pd.notna(value) and str(value).strip()
            )
            report_lines.append(f"{label.upper()}: {details}")


def run() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dates, quantiles, _, grids, densities = load_rnds(
        str(RND_FILE),
        n_quantiles=N_QUANTILES,
    )
    normalized_dates = [
        pd.Timestamp(value).tz_localize(None).normalize()
        for value in dates
    ]
    date_to_index = {value: index for index, value in enumerate(normalized_dates)}
    observed_dates = set(normalized_dates)
    sessions = nyse_sessions(min(normalized_dates).date(), max(normalized_dates).date())
    split_date = sessions[int(np.floor(TRAIN_FRACTION * len(sessions)))]
    non_monotone_observed = np.flatnonzero(
        np.any(np.diff(quantiles, axis=1) < -1e-10, axis=1)
    )
    report_lines = [
        f"RND observed dates: {len(observed_dates)}",
        f"NYSE calendar sessions: {len(sessions)}",
        f"Shared chronological split date: {split_date.date()}",
        "Each target uses the RND observed on the exact h-th subsequent NYSE session.",
        "Missing endpoint dates remove pairs; pairs are never bridged across gaps.",
    ]
    for index in non_monotone_observed:
        report_lines.append(
            f"NUMERICAL ISSUE: observed non-monotone quantiles for "
            f"{normalized_dates[index].date()}"
        )

    all_pairs = []
    all_predictions = []
    all_summaries = []

    for horizon in HORIZONS:
        pairs = build_trading_pairs(sessions, observed_dates, horizon)
        if pairs.empty:
            report_lines.append(f"h={horizon}: no complete observed pairs")
            continue
        report_lines.append(
            f"h={horizon}: {len(sessions) - horizon} calendar candidate pairs; "
            f"{len(pairs)} valid; {len(sessions) - horizon - len(pairs)} removed "
            "because an endpoint RND was missing"
        )

        pairs = pairs.copy()
        pairs["horizon_sessions"] = horizon
        pairs["split"] = np.where(
            pairs["target_date"] < split_date,
            "train",
            "test",
        )
        pair_count = len(pairs)
        train_count = int((pairs["split"] == "train").sum())
        if train_count < 3 or train_count >= pair_count:
            raise ValueError(
                f"Not enough h={horizon} pairs for an 80/20 split: {pair_count}"
            )
        all_pairs.append(pairs)

        input_indices = np.array(
            [date_to_index[value] for value in pairs["input_date"]],
            dtype=int,
        )
        target_indices = np.array(
            [date_to_index[value] for value in pairs["target_date"]],
            dtype=int,
        )
        inputs = quantiles[input_indices]
        targets = quantiles[target_indices]
        train_inputs = inputs[:train_count]
        train_targets = targets[:train_count]
        test_inputs = inputs[train_count:]
        test_targets = targets[train_count:]
        test_pairs = pairs.iloc[train_count:].reset_index(drop=True)

        train_variances = np.array(
            [
                rnd_variance(grids[index], densities[index])
                for index in input_indices[:train_count]
            ],
            dtype=float,
        )
        target_variances = np.array(
            [
                rnd_variance(grids[index], densities[index])
                for index in target_indices[:train_count]
            ],
            dtype=float,
        )

        model_predictions, optimizer_diagnostics = _fit_predict_models(
            train_inputs,
            train_targets,
            target_variances,
            test_inputs,
        )
        target_moments = quantile_moments(test_targets)
        quantile_grid = np.linspace(1e-4, 1 - 1e-4, test_targets.shape[1])
        true_moment_gaussian = (
            target_moments[:, 0, None]
            + np.sqrt(target_moments[:, 1])[:, None]
            * norm.ppf(quantile_grid)[None, :]
        )
        true_moment_errors = _w2_errors(
            true_moment_gaussian,
            test_targets,
        )
        all_summaries.append(
            {
                "horizon_sessions": horizon,
                "model": GAUSSIAN_TRUE_MOMENTS_REFERENCE,
                "test_pairs": len(test_pairs),
                "mean_w2": float(true_moment_errors.mean()),
                "mean_squared_w2": float(np.mean(true_moment_errors**2)),
                "skill_vs_persistence_w2": np.nan,
                "skill_vs_persistence_mean_squared_w2": np.nan,
            }
        )
        report_lines.extend(
            f"{message} h={horizon}"
            for message in optimizer_diagnostics
        )

        persistence_errors = _w2_errors(
            model_predictions["Persistence"],
            test_targets,
        )
        for model_name, predictions in model_predictions.items():
            non_monotone_rows = np.flatnonzero(
                np.any(np.diff(predictions, axis=1) < -1e-10, axis=1)
            )
            if len(non_monotone_rows):
                dates_for_issue = test_pairs.iloc[non_monotone_rows]["target_date"]
                for issue_date in dates_for_issue:
                    report_lines.append(
                        f"NUMERICAL ISSUE h={horizon} model={model_name}: "
                        f"non-monotone forecast quantiles for {issue_date.date()}"
                    )

            errors = _w2_errors(predictions, test_targets)
            result = test_pairs[["input_date", "target_date"]].copy()
            result = result.rename(columns={"target_date": "date"})
            result["horizon_sessions"] = horizon
            result["model"] = model_name
            result["w2"] = errors
            result["w2_squared"] = errors**2
            all_predictions.append(result)

            mean_w2 = float(np.mean(errors))
            mean_w2_squared = float(np.mean(errors**2))
            persistence_mean = float(np.mean(persistence_errors))
            persistence_mean_squared = float(np.mean(persistence_errors**2))
            all_summaries.append(
                {
                    "horizon_sessions": horizon,
                    "model": model_name,
                    "test_pairs": len(errors),
                    "mean_w2": mean_w2,
                    "mean_squared_w2": mean_w2_squared,
                    "skill_vs_persistence_w2": (
                        1.0 - mean_w2 / persistence_mean
                        if persistence_mean > 0
                        else np.nan
                    ),
                    "skill_vs_persistence_mean_squared_w2": (
                        1.0 - mean_w2_squared / persistence_mean_squared
                        if persistence_mean_squared > 0
                        else np.nan
                    ),
                }
            )

        report_lines.append(
            f"h={horizon}: {pair_count} valid pairs; "
            f"{pair_count - train_count} held-out pairs; "
            f"shared split for all models"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_result = pd.concat(all_pairs, ignore_index=True)
    predictions_result = pd.concat(all_predictions, ignore_index=True)
    summary_result = pd.DataFrame(all_summaries)
    pairs_result.to_csv(PAIRS_FILE, index=False)
    predictions_result.to_csv(PREDICTIONS_FILE, index=False)
    summary_result.to_csv(SUMMARY_FILE, index=False)
    return predictions_result, summary_result, report_lines


if __name__ == "__main__":
    _, summary, diagnostics = run()
    _append_stage_diagnostics(diagnostics)
    existing_report = REPORT_FILE.read_text(encoding="utf-8") if REPORT_FILE.exists() else ""
    with REPORT_FILE.open("w", encoding="utf-8") as report:
        if existing_report:
            report.write(existing_report.rstrip() + "\n\n")
        report.write("TRADING-CALENDAR FORECAST EVALUATION\n")
        report.write("\n".join(diagnostics))
        report.write("\n\nSUMMARY METRICS\n")
        report.write(summary.to_string(index=False))
        report.write("\n")
    print(summary.to_string(index=False))
    print(f"Failure and numerical report: {REPORT_FILE}")
