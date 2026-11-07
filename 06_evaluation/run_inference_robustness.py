from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_gp_model"))
sys.path.insert(0, str(ROOT / "06_evaluation"))

from fit_gp import load_rnds, rnd_variance  # noqa: E402
from run_horizon_evaluation import (  # noqa: E402
    HORIZONS,
    N_QUANTILES,
    TRAIN_FRACTION,
    GAUSSIAN_MOMENT_BASELINE,
    _fit_predict_models,
    _w2_errors,
)
from trading_calendar_pairs import build_trading_pairs, nyse_sessions  # noqa: E402


RND_FILE = ROOT / "02_rnd_extraction" / "data" / "rnd_30d.csv"
SMILE_FILE = ROOT / "01_data_collection" / "data" / "spx_30d_smiles.csv"
RESULT_DIR = ROOT / "06_evaluation" / "results"
PREDICTIONS_FILE = RESULT_DIR / "horizon_forecast_predictions.csv"
INFERENCE_FILE = RESULT_DIR / "forecast_inference_summary.csv"
DIFFERENTIALS_FILE = RESULT_DIR / "forecast_loss_differentials.csv"
EXPANDING_PREDICTIONS_FILE = RESULT_DIR / "expanding_window_predictions.csv"
EXPANDING_SUMMARY_FILE = RESULT_DIR / "expanding_window_summary.csv"
REGIME_SUMMARY_FILE = RESULT_DIR / "expanding_window_regime_summary.csv"
STABILITY_FILE = RESULT_DIR / "expanding_window_skill_stability.csv"
REPORT_FILE = RESULT_DIR / "pipeline_failure_report.txt"
RND_DIAGNOSTICS_FILE = ROOT / "02_rnd_extraction" / "data" / "rnd_30d_diagnostics.csv"
MARTINGALE_DIAGNOSTICS_FILE = RESULT_DIR / "martingale_miss_diagnostics.csv"
MARTINGALE_SENSITIVITY_FILE = RESULT_DIR / "martingale_exclusion_sensitivity.csv"
HISTORY_PROBE_FILE = RESULT_DIR / "thetadata_history_probe.csv"
FAILED_SMILE_QUOTE_FILE = RESULT_DIR / "failed_smile_quote_quality.csv"

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20261106
SPLIT_FRACTIONS = (0.50, 0.65, 0.80)
LOSSES = {
    "W2": "w2",
    "Squared W2": "w2_squared",
}


def _hac_mean_variance(values: np.ndarray, lags: int) -> float:
    values = np.asarray(values, dtype=float)
    sample_count = len(values)
    if sample_count < 2:
        raise ValueError("At least two observations are required for HAC variance")
    lags = min(max(int(lags), 0), sample_count - 1)
    centered = values - values.mean()
    long_run_variance = float(centered @ centered) / sample_count
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag]) / sample_count
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance
    return max(long_run_variance / sample_count, 0.0)


def _moving_block_bootstrap_ci(
    persistence_losses: np.ndarray,
    model_losses: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    sample_count = len(persistence_losses)
    if block_length < 1 or block_length > sample_count:
        raise ValueError(
            f"Block length {block_length} is invalid for n={sample_count}"
        )
    blocks_per_sample = int(np.ceil(sample_count / block_length))
    starts = rng.integers(
        0,
        sample_count - block_length + 1,
        size=(BOOTSTRAP_RESAMPLES, blocks_per_sample),
    )
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets).reshape(BOOTSTRAP_RESAMPLES, -1)
    indices = indices[:, :sample_count]
    skill = 1.0 - (
        model_losses[indices].mean(axis=1)
        / persistence_losses[indices].mean(axis=1)
    )
    lower, upper = np.quantile(skill, [0.025, 0.975])
    return float(lower), float(upper)


def _paired_inference(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["horizon_sessions", "input_date", "date"]
    persistence = predictions.loc[
        predictions["model"] == "Persistence",
        keys + list(LOSSES.values()),
    ].rename(
        columns={
            "w2": "persistence_w2",
            "w2_squared": "persistence_w2_squared",
        }
    )
    models = predictions.loc[predictions["model"] != "Persistence"].merge(
        persistence,
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if models.empty:
        raise ValueError("No paired model and persistence forecasts were found")

    differential_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_rows = []
    for (horizon, model), group in models.groupby(
        ["horizon_sessions", "model"],
        sort=True,
    ):
        group = group.sort_values("input_date").reset_index(drop=True)
        for label, column in LOSSES.items():
            persistence_losses = group[f"persistence_{column}"].to_numpy(float)
            model_losses = group[column].to_numpy(float)
            differential = persistence_losses - model_losses
            mean_persistence = float(persistence_losses.mean())
            mean_differential = float(differential.mean())
            mean_skill = (
                mean_differential / mean_persistence
                if mean_persistence > 0
                else np.nan
            )

            primary_lags = max(int(horizon) - 1, 0)
            sensitivity_lags = 5 if horizon == 1 else 10
            nw_se_d = np.sqrt(_hac_mean_variance(differential, primary_lags))
            nw_t = mean_differential / nw_se_d if nw_se_d > 0 else np.nan
            if np.isfinite(nw_t):
                log_nw_p = float(np.log(2.0) + norm.logsf(abs(nw_t)))
                nw_p = (
                    float(np.exp(log_nw_p))
                    if log_nw_p > np.log(np.finfo(float).tiny)
                    else 0.0
                )
                log10_nw_p = log_nw_p / np.log(10.0)
            else:
                nw_p = np.nan
                log10_nw_p = np.nan
            influence = (
                differential - mean_skill * persistence_losses
            ) / mean_persistence
            nw_se_skill = np.sqrt(_hac_mean_variance(influence, primary_lags))
            nw_se_skill_sensitivity = np.sqrt(
                _hac_mean_variance(influence, sensitivity_lags)
            )
            bootstrap_primary = _moving_block_bootstrap_ci(
                persistence_losses,
                model_losses,
                int(horizon),
                rng,
            )
            bootstrap_10 = _moving_block_bootstrap_ci(
                persistence_losses,
                model_losses,
                10,
                rng,
            )

            inference_rows.append(
                {
                    "horizon_sessions": horizon,
                    "model": model,
                    "loss": label,
                    "n_pairs": len(group),
                    "mean_persistence_loss": mean_persistence,
                    "mean_model_loss": float(model_losses.mean()),
                    "mean_differential_persistence_minus_model": mean_differential,
                    "nw_lags_primary": primary_lags,
                    "nw_se_mean_differential": nw_se_d,
                    "nw_t_stat_mean_differential": nw_t,
                    "nw_p_value_mean_differential": nw_p,
                    "nw_log10_p_value_mean_differential": log10_nw_p,
                    "mean_skill": mean_skill,
                    "nw_se_mean_skill": nw_se_skill,
                    "nw_95_ci_skill_lower": mean_skill - 1.96 * nw_se_skill,
                    "nw_95_ci_skill_upper": mean_skill + 1.96 * nw_se_skill,
                    "nw_lags_sensitivity": sensitivity_lags,
                    "nw_se_skill_sensitivity": nw_se_skill_sensitivity,
                    "nw_95_ci_skill_sensitivity_lower": (
                        mean_skill - 1.96 * nw_se_skill_sensitivity
                    ),
                    "nw_95_ci_skill_sensitivity_upper": (
                        mean_skill + 1.96 * nw_se_skill_sensitivity
                    ),
                    "bootstrap_block_primary": int(horizon),
                    "bootstrap_95_ci_primary_lower": bootstrap_primary[0],
                    "bootstrap_95_ci_primary_upper": bootstrap_primary[1],
                    "bootstrap_block_10": 10,
                    "bootstrap_95_ci_block10_lower": bootstrap_10[0],
                    "bootstrap_95_ci_block10_upper": bootstrap_10[1],
                }
            )

            for row in group.itertuples(index=False):
                differential_rows.append(
                    {
                        "horizon_sessions": horizon,
                        "input_date": row.input_date,
                        "date": row.date,
                        "model": model,
                        "loss": label,
                        "persistence_loss": getattr(
                            row,
                            f"persistence_{column}",
                        ),
                        "model_loss": getattr(row, column),
                        "loss_differential_persistence_minus_model": (
                            getattr(row, f"persistence_{column}")
                            - getattr(row, column)
                        ),
                    }
                )
    return pd.DataFrame(inference_rows), pd.DataFrame(differential_rows)


def _atm_iv_by_date() -> dict[pd.Timestamp, float]:
    smiles = pd.read_csv(
        SMILE_FILE,
        usecols=["date", "k", "iv_30d"],
        parse_dates=["date"],
    )
    smiles["date"] = smiles["date"].dt.tz_localize(None).dt.normalize()
    smiles["k"] = pd.to_numeric(smiles["k"], errors="coerce")
    smiles["iv_30d"] = pd.to_numeric(smiles["iv_30d"], errors="coerce")
    smiles = smiles.dropna(subset=["date", "k", "iv_30d"])
    smiles["abs_k"] = smiles["k"].abs()
    atm = (
        smiles.sort_values(["date", "abs_k"])
        .drop_duplicates("date")
        .set_index("date")["iv_30d"]
    )
    return atm.astype(float).to_dict()


def _expanding_window_evaluation() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
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
    sessions = nyse_sessions(
        min(normalized_dates).date(),
        max(normalized_dates).date(),
    )
    atm_iv = _atm_iv_by_date()
    prediction_rows = []
    regime_rows = []
    optimizer_warnings = []

    for split_fraction in SPLIT_FRACTIONS:
        cutoff = sessions[int(np.floor(split_fraction * len(sessions)))]
        cutoff = pd.Timestamp(cutoff).tz_localize(None).normalize()
        for horizon in HORIZONS:
            pairs = build_trading_pairs(sessions, observed_dates, horizon)
            train_pairs = pairs.loc[pairs["target_date"] < cutoff].reset_index(
                drop=True
            )
            test_pairs = pairs.loc[pairs["target_date"] >= cutoff].reset_index(
                drop=True
            )
            if len(train_pairs) < 3 or test_pairs.empty:
                raise ValueError(
                    f"Insufficient pairs for split={split_fraction}, h={horizon}"
                )

            train_input_indices = np.array(
                [date_to_index[value] for value in train_pairs["input_date"]],
                dtype=int,
            )
            train_target_indices = np.array(
                [date_to_index[value] for value in train_pairs["target_date"]],
                dtype=int,
            )
            test_input_indices = np.array(
                [date_to_index[value] for value in test_pairs["input_date"]],
                dtype=int,
            )
            test_target_indices = np.array(
                [date_to_index[value] for value in test_pairs["target_date"]],
                dtype=int,
            )
            train_inputs = quantiles[train_input_indices]
            train_targets = quantiles[train_target_indices]
            test_inputs = quantiles[test_input_indices]
            test_targets = quantiles[test_target_indices]
            train_target_variances = np.array(
                [
                    rnd_variance(grids[index], densities[index])
                    for index in train_target_indices
                ],
                dtype=float,
            )
            model_predictions, warnings = _fit_predict_models(
                train_inputs,
                train_targets,
                train_target_variances,
                test_inputs,
            )
            optimizer_warnings.extend(
                f"split={split_fraction:.2f} cutoff={cutoff.date()} "
                f"h={horizon}: {warning}"
                for warning in warnings
            )
            persistence_losses = _w2_errors(
                model_predictions["Persistence"],
                test_targets,
            )
            persistence_squared_losses = persistence_losses**2

            train_vol = np.array(
                [atm_iv.get(value, np.nan) for value in train_pairs["input_date"]],
                dtype=float,
            )
            test_vol = np.array(
                [atm_iv.get(value, np.nan) for value in test_pairs["input_date"]],
                dtype=float,
            )
            valid_train_vol = train_vol[np.isfinite(train_vol)]
            if len(valid_train_vol) < 3:
                raise ValueError(
                    f"Insufficient training ATM IV data for cutoff {cutoff.date()}"
                )
            low_cutoff, high_cutoff = np.quantile(valid_train_vol, [1 / 3, 2 / 3])
            regimes = np.full(len(test_pairs), "middle", dtype=object)
            regimes[np.isfinite(test_vol) & (test_vol <= low_cutoff)] = "low_vol"
            regimes[np.isfinite(test_vol) & (test_vol >= high_cutoff)] = "high_vol"

            for model, forecast in model_predictions.items():
                errors = _w2_errors(forecast, test_targets)
                for index, pair in enumerate(test_pairs.itertuples(index=False)):
                    prediction_rows.append(
                        {
                            "split_fraction": split_fraction,
                            "cutoff_date": cutoff,
                            "horizon_sessions": horizon,
                            "input_date": pair.input_date,
                            "target_date": pair.target_date,
                            "model": model,
                            "w2": errors[index],
                            "w2_squared": errors[index] ** 2,
                            "terminal_date_30d_atm_iv": test_vol[index],
                            "volatility_regime": regimes[index],
                        }
                    )

                for regime in ("low_vol", "high_vol"):
                    mask = regimes == regime
                    if not mask.any():
                        continue
                    regime_persistence = persistence_losses[mask]
                    regime_persistence_sq = persistence_squared_losses[mask]
                    regime_errors = errors[mask]
                    regime_errors_sq = regime_errors**2
                    regime_rows.append(
                        {
                            "split_fraction": split_fraction,
                            "cutoff_date": cutoff,
                            "horizon_sessions": horizon,
                            "model": model,
                            "regime": regime,
                            "n_pairs": int(mask.sum()),
                            "training_atm_iv_tercile_1": float(low_cutoff),
                            "training_atm_iv_tercile_2": float(high_cutoff),
                            "mean_w2": float(regime_errors.mean()),
                            "mean_squared_w2": float(regime_errors_sq.mean()),
                            "mean_skill_w2": (
                                1.0 - float(regime_errors.mean())
                                / float(regime_persistence.mean())
                                if regime_persistence.mean() > 0
                                else np.nan
                            ),
                            "mean_skill_squared_w2": (
                                1.0 - float(regime_errors_sq.mean())
                                / float(regime_persistence_sq.mean())
                                if regime_persistence_sq.mean() > 0
                                else np.nan
                            ),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    regimes = pd.DataFrame(regime_rows)
    split_summary_rows = []
    for (split_fraction, cutoff, horizon, model), group in predictions.groupby(
        ["split_fraction", "cutoff_date", "horizon_sessions", "model"],
        sort=True,
    ):
        persistence = predictions.loc[
            (predictions["split_fraction"] == split_fraction)
            & (predictions["horizon_sessions"] == horizon)
            & (predictions["model"] == "Persistence")
        ].sort_values("target_date")
        group = group.sort_values("target_date")
        if not group["target_date"].reset_index(drop=True).equals(
            persistence["target_date"].reset_index(drop=True)
        ):
            raise ValueError("Models do not share identical expanding test pairs")
        split_summary_rows.append(
            {
                "split_fraction": split_fraction,
                "cutoff_date": cutoff,
                "horizon_sessions": horizon,
                "model": model,
                "n_pairs": len(group),
                "mean_w2": float(group["w2"].mean()),
                "mean_squared_w2": float(group["w2_squared"].mean()),
                "mean_skill_w2": (
                    1.0 - float(group["w2"].mean())
                    / float(persistence["w2"].mean())
                    if persistence["w2"].mean() > 0
                    else np.nan
                ),
                "mean_skill_squared_w2": (
                    1.0 - float(group["w2_squared"].mean())
                    / float(persistence["w2_squared"].mean())
                    if persistence["w2_squared"].mean() > 0
                    else np.nan
                ),
            }
        )

    split_summary = pd.DataFrame(split_summary_rows)
    stability_rows = []
    for (horizon, model), group in split_summary.loc[
        split_summary["model"] != "Persistence"
    ].groupby(["horizon_sessions", "model"], sort=True):
        for metric, column in (
            ("W2", "mean_skill_w2"),
            ("Squared W2", "mean_skill_squared_w2"),
        ):
            values = group.sort_values("split_fraction")[column].to_numpy(float)
            signs = np.sign(values)
            stability_rows.append(
                {
                    "horizon_sessions": horizon,
                    "model": model,
                    "loss": metric,
                    "split_skills": "; ".join(f"{value:.8g}" for value in values),
                    "sign_stable_across_splits": bool(
                        np.all(signs > 0) or np.all(signs < 0)
                    ),
                    "sign_pattern": (
                        "all_positive"
                        if np.all(signs > 0)
                        else "all_negative"
                        if np.all(signs < 0)
                        else "mixed_or_zero"
                    ),
                }
            )
    return (
        predictions,
        split_summary,
        regimes,
        pd.DataFrame(stability_rows),
        optimizer_warnings,
    )


def _martingale_exclusion_sensitivity(
    original_inference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    diagnostics = pd.read_csv(RND_DIAGNOSTICS_FILE)
    failed = diagnostics.loc[
        ~diagnostics["martingale_pass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1"})
    ].copy()
    failed["date"] = pd.to_datetime(failed["date"]).dt.tz_localize(None).dt.normalize()
    failed["absolute_miss_points"] = (
        failed["extracted_mean"] - failed["forward"]
    ).abs()
    failed["signed_miss_points"] = failed["extracted_mean"] - failed["forward"]
    failed["martingale_miss_percent"] = 100.0 * failed["martingale_rel_error"]
    failed.to_csv(MARTINGALE_DIAGNOSTICS_FILE, index=False)
    excluded_dates = set(failed["date"])

    dates, quantiles, _, grids, densities = load_rnds(
        str(RND_FILE),
        n_quantiles=N_QUANTILES,
    )
    normalized_dates = [
        pd.Timestamp(value).tz_localize(None).normalize()
        for value in dates
    ]
    date_to_index = {
        value: index for index, value in enumerate(normalized_dates)
    }
    observed_dates = set(normalized_dates) - excluded_dates
    sessions = nyse_sessions(
        min(normalized_dates).date(),
        max(normalized_dates).date(),
    )
    split_date = sessions[int(np.floor(TRAIN_FRACTION * len(sessions)))]
    prediction_rows = []
    train_counts = []
    optimizer_warnings = []

    for horizon in HORIZONS:
        pairs = build_trading_pairs(sessions, observed_dates, horizon)
        train_pairs = pairs.loc[pairs["target_date"] < split_date].reset_index(
            drop=True
        )
        test_pairs = pairs.loc[pairs["target_date"] >= split_date].reset_index(
            drop=True
        )
        if len(train_pairs) < 3 or test_pairs.empty:
            raise ValueError(
                f"Insufficient pairs after excluding martingale misses for h={horizon}"
            )
        train_input_indices = np.array(
            [date_to_index[value] for value in train_pairs["input_date"]],
            dtype=int,
        )
        train_target_indices = np.array(
            [date_to_index[value] for value in train_pairs["target_date"]],
            dtype=int,
        )
        test_input_indices = np.array(
            [date_to_index[value] for value in test_pairs["input_date"]],
            dtype=int,
        )
        test_target_indices = np.array(
            [date_to_index[value] for value in test_pairs["target_date"]],
            dtype=int,
        )
        predictions, warnings = _fit_predict_models(
            quantiles[train_input_indices],
            quantiles[train_target_indices],
            np.array(
                [
                    rnd_variance(grids[index], densities[index])
                    for index in train_target_indices
                ],
                dtype=float,
            ),
            quantiles[test_input_indices],
        )
        optimizer_warnings.extend(
            f"martingale-dates-excluded h={horizon}: {warning}"
            for warning in warnings
        )
        train_counts.append(
            {
                "horizon_sessions": horizon,
                "n_train_pairs": len(train_pairs),
                "n_test_pairs": len(test_pairs),
                "split_date": split_date,
            }
        )
        targets = quantiles[test_target_indices]
        for model, forecast in predictions.items():
            errors = _w2_errors(forecast, targets)
            for index, pair in enumerate(test_pairs.itertuples(index=False)):
                prediction_rows.append(
                    {
                        "horizon_sessions": horizon,
                        "input_date": pair.input_date,
                        "date": pair.target_date,
                        "model": model,
                        "w2": errors[index],
                        "w2_squared": errors[index] ** 2,
                    }
                )

    refit_predictions = pd.DataFrame(prediction_rows)
    refit_inference, _ = _paired_inference(refit_predictions)
    count_lookup = {
        row["horizon_sessions"]: row for row in train_counts
    }
    original = original_inference.set_index(
        ["horizon_sessions", "model", "loss"]
    )
    comparison_rows = []
    for row in refit_inference.to_dict("records"):
        key = (row["horizon_sessions"], row["model"], row["loss"])
        baseline = original.loc[key]
        row.update(
            {
                "n_train_pairs": count_lookup[row["horizon_sessions"]][
                    "n_train_pairs"
                ],
                "split_date": count_lookup[row["horizon_sessions"]]["split_date"],
                "original_mean_skill": baseline["mean_skill"],
                "skill_sign_changed": (
                    np.sign(row["mean_skill"])
                    != np.sign(baseline["mean_skill"])
                ),
                "nw_primary_ci_zero_inclusion_changed": (
                    (row["nw_95_ci_skill_lower"] <= 0 <= row["nw_95_ci_skill_upper"])
                    != (
                        baseline["nw_95_ci_skill_lower"]
                        <= 0
                        <= baseline["nw_95_ci_skill_upper"]
                    )
                ),
                "nw_sensitivity_ci_zero_inclusion_changed": (
                    (
                        row["nw_95_ci_skill_sensitivity_lower"]
                        <= 0
                        <= row["nw_95_ci_skill_sensitivity_upper"]
                    )
                    != (
                        baseline["nw_95_ci_skill_sensitivity_lower"]
                        <= 0
                        <= baseline["nw_95_ci_skill_sensitivity_upper"]
                    )
                ),
                "bootstrap_h_ci_zero_inclusion_changed": (
                    (
                        row["bootstrap_95_ci_primary_lower"]
                        <= 0
                        <= row["bootstrap_95_ci_primary_upper"]
                    )
                    != (
                        baseline["bootstrap_95_ci_primary_lower"]
                        <= 0
                        <= baseline["bootstrap_95_ci_primary_upper"]
                    )
                ),
                "bootstrap_10_ci_zero_inclusion_changed": (
                    (
                        row["bootstrap_95_ci_block10_lower"]
                        <= 0
                        <= row["bootstrap_95_ci_block10_upper"]
                    )
                    != (
                        baseline["bootstrap_95_ci_block10_lower"]
                        <= 0
                        <= baseline["bootstrap_95_ci_block10_upper"]
                    )
                ),
            }
        )
        comparison_rows.append(row)
    sensitivity = pd.DataFrame(comparison_rows)
    sensitivity.to_csv(MARTINGALE_SENSITIVITY_FILE, index=False)
    return failed, sensitivity, optimizer_warnings


def _write_report_appendix(
    inference: pd.DataFrame,
    expanding_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    stability: pd.DataFrame,
    optimizer_warnings: list[str],
    martingale_diagnostics: pd.DataFrame,
    martingale_sensitivity: pd.DataFrame,
    martingale_optimizer_warnings: list[str],
) -> None:
    marker = "\nINFERENCE AND SPLIT ROBUSTNESS\n"
    previous = REPORT_FILE.read_text(encoding="utf-8") if REPORT_FILE.exists() else ""
    if marker in previous:
        previous = previous.split(marker, maxsplit=1)[0]
    previous = previous.replace(
        "Linear moment baseline",
        GAUSSIAN_MOMENT_BASELINE,
    )
    with REPORT_FILE.open("w", encoding="utf-8") as report:
        report.write(previous.rstrip())
        report.write(marker)
        if HISTORY_PROBE_FILE.exists():
            probe = pd.read_csv(HISTORY_PROBE_FILE)
            probe["date"] = probe["date"].astype(str)
            status_table = (
                probe.pivot_table(
                    index="date",
                    columns="root",
                    values="status",
                    aggfunc="first",
                )
                .reset_index()
                .sort_values("date")
            )
            returned_rows = (
                probe.loc[
                    probe["status"] == "data_returned",
                    ["date", "root", "rows"],
                ]
                .assign(rows=lambda frame: frame["rows"].astype(int))
                .pivot_table(
                    index="date",
                    columns="root",
                    values="rows",
                    aggfunc="first",
                )
                .add_suffix("_rows")
                .reset_index()
            )
            status_table = status_table.merge(
                returned_rows,
                on="date",
                how="left",
            )
            report.write("\nTHETADATA HISTORY PROBE (no raw quotes saved)\n")
            report.write(
                "Requests: "
                f"{len(probe)}; data returned: "
                f"{int((probe['status'] == 'data_returned').sum())}; "
                f"permission denied: "
                f"{int((probe['status'] == 'permission_denied').sum())}; "
                f"rate-limited: {int((probe['status'] == 'rate_limited').sum())}. "
                "No rate limiting was observed at a one-second request pace. "
                "Earliest tested accessible date for each root: 2023-06-01; "
                "2023-05-31 was denied for both.\n"
            )
            report.write(status_table.to_string(index=False))
            report.write("\n")
        report.write(
            "\nDATA AUDIT (read-only summary of local 2024-2026 outputs)\n"
            "The earliest tested accessible SPX and SPXW history date was "
            "2023-06-01; 2023-05-31 was denied for both roots. All sampled "
            "dates in 2022 and earlier, plus 2023-03-01, were permission "
            "denied. The collected chain spans 2024-01-02 through "
            "2026-11-04 and contains both SPX (1,194,080 rows) and SPXW "
            "(6,803,711 rows); both roots occur on 701 of 714 sessions.\n"
            "\nSeven 30-day smile failures (all selected expiry groups are "
            "SPXW; SPX has no rows for these expiries):\n"
            "date | below expiry (DTE; SPXW rows/usable IV) | above expiry "
            "(DTE; SPXW rows/usable IV) | diagnostic\n"
            "2024-07-03 | 2024-08-01 (29; 178/0) | 2024-08-05 (33; 154/0) | "
            "fewer than 5 usable points on both sides\n"
            "2024-11-29 | 2024-12-27 (28; 402/0) | 2024-12-30 (31; 170/0) | "
            "fewer than 5 usable points on both sides\n"
            "2024-12-02 | 2024-12-31 (29; 970/707) | 2025-01-02 (31; 180/0) | "
            "fewer than 5 usable points above 30D\n"
            "2024-12-24 | 2025-01-22 (29; 200/0) | 2025-01-24 (31; 386/0) | "
            "fewer than 5 usable points on both sides\n"
            "2025-07-03 | 2025-08-01 (29; 312/0) | 2025-08-04 (32; 190/0) | "
            "fewer than 5 usable points on both sides\n"
            "2025-11-28 | 2025-12-26 (28; 466/0) | 2025-12-29 (31; 224/0) | "
            "fewer than 5 usable points on both sides\n"
            "2025-12-24 | 2026-01-22 (29; 230/0) | 2026-01-26 (33; 212/0) | "
            "fewer than 5 usable points on both sides\n"
            "SPXW was collected for these dates. Additional SPXW collection "
            "alone would not restore these smiles; the selected expiries have "
            "many raw rows but no usable midpoint IVs (except the usable "
            "lower expiry on 2024-12-02).\n"
            "\nThe 19,694 IV-warning expiry groups contain 711,523 rows with "
            "missing midpoint IV. Signed log(K/F) buckets for those rows: "
            "K/F <= exp(-0.10): 386,589; (-0.10,-0.02]: 38,373; "
            "(-0.02,0.02]: 22,792; (0.02,0.10]: 98,825; "
            "K/F > exp(0.10): 164,944. These are row counts, not group counts.\n"
            "Among missing-mid-IV rows, 202,611 have bid=0 (overlapping "
            "classification); 711,138 midpoints are at/below discounted "
            "intrinsic value; 10 are at/above the Black-76 upper bound; "
            "375 are strictly inside theoretical price bounds. Of those 375, "
            "372 cannot be bracketed by the implemented sigma interval "
            "[1e-6, 5] (mostly extreme-moneyness precision cases). Three "
            "additional rows were re-evaluated to finite midpoint IVs and "
            "their saved local IV rows were repaired; a multi-expiry adapter "
            "regression test now covers the persistence path. The raw "
            "ThetaData response includes a `created` timestamp, preserved "
            "through the chain and IV files. The separate `timestamp` alias "
            "is empty; the previous missing-timestamp count checked that "
            "empty alias rather than `created`. `created` is a common EOD "
            "snapshot time, not an update time per contract, so quote staleness "
            "cannot be inferred from it. Last-trade time is not quote freshness.\n"
            "Among the 1,016 expiry groups selected to form exact or "
            "bracketed 30D smiles, 999 have at least one unsolved midpoint "
            "IV (31,989 such rows); most still have enough other usable "
            "moneyness points. The seven failed smiles above are the cases "
            "where the selected bracket(s) lack five usable points.\n"
            "\nIssue overlap uses 'many unsolved IVs' = at least 1,000 "
            "missing midpoint-IV rows on a date:\n"
            "2024-04-12: SVI non-convergence + many unsolved IVs (1,134); "
            "no lost pairs.\n"
            "2024-07-03: failed smile + many unsolved IVs (10,518); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2024-11-29: failed smile + many unsolved IVs (9,989); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2024-12-02: failed smile + many unsolved IVs (8,906); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2024-12-24: failed smile + many unsolved IVs (11,202); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2025-05-12: SVI non-convergence + many unsolved IVs (1,084); "
            "no lost pairs.\n"
            "2025-07-03: failed smile + many unsolved IVs (11,489); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2025-11-28: failed smile + many unsolved IVs (11,662); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2025-12-24: failed smile + many unsolved IVs (11,610); "
            "2 h=1 endpoint incidences and 2 h=5 incidences.\n"
            "2026-09-08: SVI non-convergence + many unsolved IVs (1,192); "
            "no lost pairs.\n"
            "The other SVI non-convergence dates are 2025-02-26 and "
            "2025-03-27. The nine martingale misses are 2024-08-05, "
            "2025-04-04, 2025-04-07, 2025-04-08, 2025-04-09, 2025-04-10, "
            "2025-04-11, 2025-04-16, and 2025-04-21; these did not overlap "
            "another listed issue under the stated 1,000-row threshold.\n"
            "Only the seven failed-smile dates lack RNDs. They remove 13 "
            "unique h=1 pairs (14 endpoint incidences; one pair has two "
            "missing endpoints) and 14 unique h=5 pairs. Candidate/retained "
            "counts are 713/700 for h=1 and 709/695 for h=5. Non-converged "
            "SVI and martingale-miss dates still have RNDs and remove no pairs.\n"
            "On the five-case scalar/vectorized IV regression sample, the "
            "maximum absolute IV difference was 3.358980e-13.\n"
            "\nGAUSSIAN-SHAPE MOMENT BASELINE SANITY\n"
            "No leakage or arithmetic bug was found. Standardizing moment "
            "features reduced the training design condition number from "
            "5.325e3 to 4.508 but produced the same forecasts to numerical "
            "precision; no test forecast variance was clipped. Predicted "
            "test variance averaged 1.022 times realized variance (correlation "
            "0.745). The baseline assumes a Gaussian quantile shape and carries "
            "the input mean forward. Test target absolute skewness had median "
            "0.791. A Gaussian using the target's true mean and variance still "
            "had mean W2 8.031003e-03 versus 8.330430e-03 for this baseline, "
            "while persistence was 2.719857e-03. The poor result is consistent "
            "with Gaussian shape misspecification for skewed RNDs, not feature "
            "scaling or leakage; no baseline code change was made.\n"
        )
        if FAILED_SMILE_QUOTE_FILE.exists():
            quote_quality = pd.read_csv(FAILED_SMILE_QUOTE_FILE)
            report.write(
                "\nFAILED-SMILE RAW QUOTE QUALITY AND SESSION CLOSES\n"
            )
            report.write(quote_quality.to_string(index=False))
            report.write("\n")
        report.write(
            "\nAll inference uses paired per-origin loss differentials against "
            "persistence. Overlapping h=5 pairs are retained. Newey-West "
            "confidence intervals for skill use the delta method; t statistics "
            "and p-values test the mean loss differential against zero.\n"
        )
        base_summary = pd.read_csv(
            RESULT_DIR / "horizon_forecast_summary.csv"
        )
        report.write("\nBASELINE TEST METRICS (scientific notation)\n")
        report.write(
            base_summary.to_string(
                index=False,
                formatters={
                    "mean_w2": lambda value: f"{value:.5e}",
                    "mean_squared_w2": lambda value: f"{value:.5e}",
                },
            )
        )
        report.write("\n")
        report.write(
            f"Moving-block bootstrap resamples: {BOOTSTRAP_RESAMPLES}; "
            f"seed: {BOOTSTRAP_SEED}; blocks: horizon length and 10.\n"
        )
        report.write("\nPAIRED INFERENCE\n")
        report.write(inference.to_string(index=False))
        report.write("\n\nEXPANDING-WINDOW RESULTS\n")
        report.write(expanding_summary.to_string(index=False))
        report.write("\n\nHIGH/LOW VOLATILITY REGIME RESULTS\n")
        report.write(regime_summary.to_string(index=False))
        report.write("\n\nSIGN STABILITY ACROSS EXPANDING SPLITS\n")
        report.write(stability.to_string(index=False))
        report.write("\n\nMARTINGALE-MISS FORWARD/RATE DIAGNOSTICS\n")
        report.write(
            martingale_diagnostics[
                [
                    "date",
                    "forward",
                    "rate",
                    "extracted_mean",
                    "signed_miss_points",
                    "martingale_miss_percent",
                ]
            ].to_string(index=False)
        )
        report.write(
            "\n\nMARTINGALE-DATE EXCLUSION REFIT SENSITIVITY\n"
            "Each flagged date is removed before calendar pairs are built; "
            "pairs involving an excluded endpoint are dropped, then models "
            "are refit on the remaining training pairs.\n"
        )
        report.write(martingale_sensitivity.to_string(index=False))
        if optimizer_warnings:
            report.write("\n\nEXPANDING-WINDOW OPTIMIZER WARNINGS\n")
            report.write("\n".join(optimizer_warnings))
        if martingale_optimizer_warnings:
            report.write("\n\nMARTINGALE-EXCLUSION OPTIMIZER WARNINGS\n")
            report.write("\n".join(martingale_optimizer_warnings))
        report.write(
            "\n\nLocal inference outputs:\n"
            f"- {DIFFERENTIALS_FILE.name}\n"
            f"- {INFERENCE_FILE.name}\n"
            f"- {EXPANDING_PREDICTIONS_FILE.name}\n"
            f"- {EXPANDING_SUMMARY_FILE.name}\n"
            f"- {REGIME_SUMMARY_FILE.name}\n"
            f"- {STABILITY_FILE.name}\n"
            f"- {MARTINGALE_DIAGNOSTICS_FILE.name}\n"
            f"- {MARTINGALE_SENSITIVITY_FILE.name}\n"
            f"- {HISTORY_PROBE_FILE.name}\n"
            f"- {FAILED_SMILE_QUOTE_FILE.name}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Reuse saved expanding-window outputs while rerunning holdout "
            "inference and the martingale-date exclusion refit"
        ),
    )
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["input_date", "date"],
    )
    inference, differentials = _paired_inference(predictions)
    if args.inference_only:
        expanding_summary = pd.read_csv(EXPANDING_SUMMARY_FILE)
        regime_summary = pd.read_csv(REGIME_SUMMARY_FILE)
        stability = pd.read_csv(STABILITY_FILE)
        expanding_predictions = pd.read_csv(EXPANDING_PREDICTIONS_FILE)
        optimizer_warnings = []
    else:
        (
            expanding_predictions,
            expanding_summary,
            regime_summary,
            stability,
            optimizer_warnings,
        ) = _expanding_window_evaluation()
        expanding_predictions.to_csv(EXPANDING_PREDICTIONS_FILE, index=False)
        expanding_summary.to_csv(EXPANDING_SUMMARY_FILE, index=False)
        regime_summary.to_csv(REGIME_SUMMARY_FILE, index=False)
        stability.to_csv(STABILITY_FILE, index=False)

    for frame in (
        expanding_predictions,
        expanding_summary,
        regime_summary,
        stability,
    ):
        if "model" in frame:
            frame["model"] = frame["model"].replace(
                "Linear moment baseline",
                GAUSSIAN_MOMENT_BASELINE,
            )
    expanding_predictions.to_csv(EXPANDING_PREDICTIONS_FILE, index=False)
    expanding_summary.to_csv(EXPANDING_SUMMARY_FILE, index=False)
    regime_summary.to_csv(REGIME_SUMMARY_FILE, index=False)
    stability.to_csv(STABILITY_FILE, index=False)

    differentials.to_csv(DIFFERENTIALS_FILE, index=False)
    inference.to_csv(INFERENCE_FILE, index=False)
    martingale_diagnostics, martingale_sensitivity, martingale_warnings = (
        _martingale_exclusion_sensitivity(inference)
    )
    _write_report_appendix(
        inference,
        expanding_summary,
        regime_summary,
        stability,
        optimizer_warnings,
        martingale_diagnostics,
        martingale_sensitivity,
        martingale_warnings,
    )
    print("PAIRED INFERENCE")
    print(inference.to_string(index=False))
    print("\nEXPANDING-WINDOW SIGN STABILITY")
    print(stability.to_string(index=False))
    print(f"\nUpdated report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
