from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_gp_model"))
sys.path.insert(0, str(ROOT / "05_baselines"))
sys.path.insert(0, str(ROOT / "06_evaluation"))
sys.path.insert(0, str(ROOT / "03_wasserstein_kernel"))

from fit_gp import load_rnds  # noqa: E402
from distribution_forecasters import WassersteinAutoregression  # noqa: E402
from run_inference_robustness import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    _hac_mean_variance,
    _moving_block_bootstrap_ci,
)
from trading_calendar_pairs import build_trading_pairs, nyse_sessions  # noqa: E402
from quantile_change_models import (  # noqa: E402
    GAMMA_MULTIPLIERS,
    PCA_EXPLAINED_VARIANCE,
    REGULARIZATION_GRID,
    forward_chaining_folds,
    median_distance_scale,
    moment_features,
    normalize_quantile_matrix,
    pairwise_squared_distances,
    pca_features,
    prepare_kernel_fit,
    project_quantile_matrix,
    ridge_predict,
    squared_w2_per_pair,
    standardize_features,
)


DATA_DIR = ROOT / "01_data_collection" / "data"
RND_FILE = ROOT / "02_rnd_extraction" / "data" / "rnd_30d.csv"
SVI_FILE = DATA_DIR / "svi_30d_params.csv"
SMILE_FILE = DATA_DIR / "spx_30d_smiles.csv"
RESULT_DIR = ROOT / "06_evaluation" / "results"
REPORT_FILE = RESULT_DIR / "pipeline_failure_report.txt"

MODEL_SPECS = {
    "Quantile-change GP (PCA factors)": "pca_gp",
    "Linear autoregression (PCA quantile factors)": "pca_linear",
    "GP (vol/skew/kurtosis)": "moments_gp",
    "GP (SVI parameters)": "svi_gp",
    "Wasserstein-input GP (quantile changes)": "wasserstein_gp",
}
PREDICTION_MODELS = (
    "Persistence",
    *MODEL_SPECS.keys(),
    "Wasserstein autoregression (legacy)",
)
GP_MODELS = {
    key for key in MODEL_SPECS if MODEL_SPECS[key].endswith("_gp")
}
EVALUATIONS = ("rolling_origin", "fixed_80_20")

OUTPUTS = {
    "predictions": RESULT_DIR / "phase4_forecast_predictions.csv",
    "differentials": RESULT_DIR / "phase4_loss_differentials.csv",
    "summary": RESULT_DIR / "phase4_forecast_summary.csv",
    "cv_curve": RESULT_DIR / "phase4_cv_loss_curve.csv",
    "selected": RESULT_DIR / "phase4_selected_hyperparameters.csv",
    "sensitivity": RESULT_DIR / "phase4_full_sweep_sensitivity.csv",
    "inference": RESULT_DIR / "phase4_inference.csv",
    "regimes": RESULT_DIR / "phase4_regime_summary.csv",
}


def _feature_pair(
    model_kind: str,
    input_quantiles_train: np.ndarray,
    input_quantiles_eval: np.ndarray,
    svi_train: np.ndarray,
    svi_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    if model_kind in {"pca_gp", "pca_linear"}:
        train_features, eval_features, component_count = pca_features(
            input_quantiles_train,
            input_quantiles_eval,
            explained_variance=PCA_EXPLAINED_VARIANCE,
        )
        train_features, eval_features = standardize_features(
            train_features, eval_features
        )
        return train_features, eval_features, component_count
    if model_kind == "moments_gp":
        train_features, eval_features = standardize_features(
            moment_features(input_quantiles_train),
            moment_features(input_quantiles_eval),
        )
        return train_features, eval_features, None
    if model_kind == "svi_gp":
        train_features, eval_features = standardize_features(svi_train, svi_eval)
        return train_features, eval_features, None
    if model_kind == "wasserstein_gp":
        probability_grid = np.linspace(
            1e-4, 1 - 1e-4, input_quantiles_train.shape[1]
        )
        weights = np.empty_like(probability_grid)
        spacing = np.diff(probability_grid)
        weights[0] = spacing[0] / 2
        weights[-1] = spacing[-1] / 2
        weights[1:-1] = (spacing[:-1] + spacing[1:]) / 2
        root_weights = np.sqrt(weights)
        return (
            input_quantiles_train * root_weights,
            input_quantiles_eval * root_weights,
            None,
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def _kernel_candidate_prediction(
    train_features: np.ndarray,
    evaluation_features: np.ndarray,
    train_changes: np.ndarray,
    gamma_multiplier: float,
    regularization: float,
) -> tuple[np.ndarray, float]:
    train_distances = pairwise_squared_distances(train_features)
    median_scale = median_distance_scale(train_distances)
    gamma = gamma_multiplier / median_scale
    train_kernel = np.exp(-gamma * train_distances)
    cross_distances = pairwise_squared_distances(
        evaluation_features, train_features
    )
    cross_kernel = np.exp(-gamma * cross_distances)
    fitted = prepare_kernel_fit(
        train_kernel,
        train_changes,
        regularization,
    )
    return fitted.predict(cross_kernel), gamma


def _predict_candidate(
    model_kind: str,
    train_inputs: np.ndarray,
    train_changes: np.ndarray,
    eval_inputs: np.ndarray,
    train_svi: np.ndarray,
    eval_svi: np.ndarray,
    regularization: float,
    gamma_multiplier: float | None = None,
) -> tuple[np.ndarray, float | None, int | None]:
    train_features, eval_features, pca_count = _feature_pair(
        model_kind,
        train_inputs,
        eval_inputs,
        train_svi,
        eval_svi,
    )
    if model_kind == "pca_linear":
        return (
            ridge_predict(
                train_features,
                train_changes,
                eval_features,
                alpha=regularization,
            ),
            None,
            pca_count,
        )
    if gamma_multiplier is None:
        raise ValueError("Kernel models require a gamma multiplier")
    delta, gamma = _kernel_candidate_prediction(
        train_features,
        eval_features,
        train_changes,
        gamma_multiplier,
        regularization,
    )
    return delta, gamma, pca_count


def _predict_candidate_grid(
    model_kind: str,
    train_inputs: np.ndarray,
    train_changes: np.ndarray,
    eval_inputs: np.ndarray,
    train_svi: np.ndarray,
    eval_svi: np.ndarray,
) -> dict[tuple[float | None, float], tuple[np.ndarray, float | None, int | None]]:
    train_features, eval_features, pca_count = _feature_pair(
        model_kind,
        train_inputs,
        eval_inputs,
        train_svi,
        eval_svi,
    )
    results = {}
    if model_kind == "pca_linear":
        for regularization in REGULARIZATION_GRID:
            results[(None, regularization)] = (
                ridge_predict(
                    train_features,
                    train_changes,
                    eval_features,
                    alpha=regularization,
                ),
                None,
                pca_count,
            )
        return results

    train_distances = pairwise_squared_distances(train_features)
    evaluation_distances = pairwise_squared_distances(
        eval_features, train_features
    )
    median_scale = median_distance_scale(train_distances)
    for gamma_multiplier in GAMMA_MULTIPLIERS:
        gamma = gamma_multiplier / median_scale
        train_kernel = np.exp(-gamma * train_distances)
        cross_kernel = np.exp(-gamma * evaluation_distances)
        fitted = prepare_kernel_fit(
            train_kernel,
            train_changes,
            REGULARIZATION_GRID[0],
        )
        for regularization in REGULARIZATION_GRID:
            results[(gamma_multiplier, regularization)] = (
                fitted.predict(cross_kernel, alpha=regularization),
                gamma,
                pca_count,
            )
    return results


def _candidate_grid(model_kind: str) -> list[tuple[float | None, float]]:
    if model_kind == "pca_linear":
        return [(None, value) for value in REGULARIZATION_GRID]
    return [
        (gamma_multiplier, regularization)
        for gamma_multiplier in GAMMA_MULTIPLIERS
        for regularization in REGULARIZATION_GRID
    ]


def _tune_model(
    model_name: str,
    model_kind: str,
    train_pairs: pd.DataFrame,
    date_to_index: dict[pd.Timestamp, int],
    quantiles: np.ndarray,
    svi_by_date: dict[pd.Timestamp, np.ndarray],
    horizon: int,
    evaluation: str,
    outer_split: str,
) -> tuple[dict[str, float | None], list[dict[str, object]], int]:
    folds = forward_chaining_folds(len(train_pairs))
    candidates = _candidate_grid(model_kind)
    accumulated = {
        candidate: {"loss_sum": 0.0, "n_pairs": 0, "fold_losses": []}
        for candidate in candidates
    }
    component_counts = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_train = train_pairs.iloc[train_indices]
        fold_validation = train_pairs.iloc[validation_indices]
        train_input_indices = np.array(
            [date_to_index[value] for value in fold_train.input_date], dtype=int
        )
        train_target_indices = np.array(
            [date_to_index[value] for value in fold_train.target_date], dtype=int
        )
        validation_input_indices = np.array(
            [date_to_index[value] for value in fold_validation.input_date], dtype=int
        )
        validation_target_indices = np.array(
            [date_to_index[value] for value in fold_validation.target_date],
            dtype=int,
        )
        fold_train_inputs = quantiles[train_input_indices]
        fold_train_changes = (
            quantiles[train_target_indices] - fold_train_inputs
        )
        fold_validation_inputs = quantiles[validation_input_indices]
        fold_validation_targets = quantiles[validation_target_indices]
        svi_train = np.stack(
            [svi_by_date[value] for value in fold_train.input_date]
        )
        svi_validation = np.stack(
            [svi_by_date[value] for value in fold_validation.input_date]
        )

        candidate_predictions = _predict_candidate_grid(
            model_kind,
            fold_train_inputs,
            fold_train_changes,
            fold_validation_inputs,
            svi_train,
            svi_validation,
        )
        for gamma_multiplier, regularization in candidates:
            predicted_changes, _, pca_count = candidate_predictions[
                (gamma_multiplier, regularization)
            ]
            component_counts.append(pca_count)
            forecasts = project_quantile_matrix(
                fold_validation_inputs,
                predicted_changes,
            )
            losses = squared_w2_per_pair(forecasts, fold_validation_targets)
            candidate_result = accumulated[(gamma_multiplier, regularization)]
            candidate_result["loss_sum"] += float(losses.sum())
            candidate_result["n_pairs"] += len(losses)
            candidate_result["fold_losses"].append(float(losses.mean()))

    curve_rows = []
    for gamma_multiplier, regularization in candidates:
        values = accumulated[(gamma_multiplier, regularization)]
        curve_rows.append(
            {
                "evaluation": evaluation,
                "outer_split": outer_split,
                "horizon_sessions": horizon,
                "model": model_name,
                "gamma_multiplier": gamma_multiplier,
                "regularization_noise_ratio": regularization,
                "cv_squared_w2": values["loss_sum"] / values["n_pairs"],
                "cv_n_pairs": values["n_pairs"],
                "cv_fold_losses": ";".join(
                    f"{value:.12g}" for value in values["fold_losses"]
                ),
                "inner_fold_count": len(folds),
                "pca_component_count_median": (
                    float(np.median([value for value in component_counts if value]))
                    if any(value for value in component_counts)
                    else np.nan
                ),
            }
        )
    best = min(
        curve_rows,
        key=lambda row: (
            row["cv_squared_w2"],
            row["gamma_multiplier"] or 0.0,
            row["regularization_noise_ratio"],
        ),
    )
    selected = {
        "gamma_multiplier": best["gamma_multiplier"],
        "regularization_noise_ratio": float(
            best["regularization_noise_ratio"]
        ),
        "cv_squared_w2": float(best["cv_squared_w2"]),
    }
    return selected, curve_rows, int(len(folds))


def _fit_selected_and_predict(
    model_kind: str,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    eval_inputs: np.ndarray,
    train_svi: np.ndarray,
    eval_svi: np.ndarray,
    parameters: dict[str, float | None],
) -> tuple[np.ndarray, int | None]:
    train_changes = train_targets - train_inputs
    delta, _, pca_count = _predict_candidate(
        model_kind,
        train_inputs,
        train_changes,
        eval_inputs,
        train_svi,
        eval_svi,
        float(parameters["regularization_noise_ratio"]),
        parameters["gamma_multiplier"],
    )
    return project_quantile_matrix(eval_inputs, delta), pca_count


def _fit_full_sweep(
    model_name: str,
    model_kind: str,
    evaluation: str,
    outer_split: str,
    training_condition: str,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    date_to_index: dict[pd.Timestamp, int],
    quantiles: np.ndarray,
    svi_by_date: dict[pd.Timestamp, np.ndarray],
    horizon: int,
    persistence_losses: np.ndarray,
) -> list[dict[str, object]]:
    train_input_indices = np.array(
        [date_to_index[value] for value in train_pairs.input_date], dtype=int
    )
    train_target_indices = np.array(
        [date_to_index[value] for value in train_pairs.target_date], dtype=int
    )
    test_input_indices = np.array(
        [date_to_index[value] for value in test_pairs.input_date], dtype=int
    )
    test_target_indices = np.array(
        [date_to_index[value] for value in test_pairs.target_date], dtype=int
    )
    train_inputs = quantiles[train_input_indices]
    train_changes = quantiles[train_target_indices] - train_inputs
    test_inputs = quantiles[test_input_indices]
    test_targets = quantiles[test_target_indices]
    train_svi = np.stack([svi_by_date[value] for value in train_pairs.input_date])
    test_svi = np.stack([svi_by_date[value] for value in test_pairs.input_date])
    output = []
    candidate_predictions = _predict_candidate_grid(
        model_kind,
        train_inputs,
        train_changes,
        test_inputs,
        train_svi,
        test_svi,
    )
    for gamma_multiplier, regularization in _candidate_grid(model_kind):
        delta, actual_gamma, _ = candidate_predictions[
            (gamma_multiplier, regularization)
        ]
        forecast = project_quantile_matrix(test_inputs, delta)
        model_losses_squared = squared_w2_per_pair(forecast, test_targets)
        skill = 1 - model_losses_squared.mean() / persistence_losses.mean()
        output.append(
            {
                "evaluation": evaluation,
                "outer_split": outer_split,
                "training_condition": training_condition,
                "analysis_role": "SENSITIVITY ONLY - not the selected result",
                "horizon_sessions": horizon,
                "model": model_name,
                "gamma_multiplier": gamma_multiplier,
                "actual_gamma": actual_gamma,
                "regularization_noise_ratio": regularization,
                "mean_squared_w2": float(model_losses_squared.mean()),
                "skill_vs_persistence_squared_w2": float(skill),
            }
        )
    return output


def _volatility_by_date() -> dict[pd.Timestamp, float]:
    smiles = pd.read_csv(SMILE_FILE, parse_dates=["date"])
    smiles["date"] = pd.to_datetime(smiles["date"]).dt.tz_localize(None).dt.normalize()
    atm_rows = (
        smiles.assign(abs_k=smiles["k"].abs())
        .sort_values(["date", "abs_k"])
        .groupby("date", as_index=False)
        .first()
    )
    return {
        row.date: float(row.iv_30d)
        for row in atm_rows.itertuples(index=False)
        if np.isfinite(row.iv_30d)
    }


def _svi_by_date(expected_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, np.ndarray]:
    parameters = pd.read_csv(SVI_FILE, parse_dates=["date"])
    parameters["date"] = (
        pd.to_datetime(parameters["date"]).dt.tz_localize(None).dt.normalize()
    )
    columns = ["a", "b", "rho", "m", "sigma"]
    result = {}
    for row in parameters.itertuples(index=False):
        values = np.asarray([getattr(row, column) for column in columns], dtype=float)
        if np.isfinite(values).all():
            result[row.date] = values
    missing = sorted(set(expected_dates) - set(result))
    if missing:
        raise ValueError(
            f"Missing finite SVI feature rows for {len(missing)} RND dates; "
            f"first missing={missing[0].date()}"
        )
    return result


def _svi_fit_status_note() -> str:
    parameters = pd.read_csv(SVI_FILE, parse_dates=["date"])
    success = (
        parameters["success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1"})
    )
    columns = ["a", "b", "rho", "m", "sigma"]
    finite_parameters = np.isfinite(parameters[columns].to_numpy(dtype=float)).all(
        axis=1
    )
    failed_finite = parameters.loc[~success & finite_parameters, "date"]
    failed_dates = ",".join(
        pd.to_datetime(failed_finite).dt.strftime("%Y-%m-%d").tolist()
    )
    return (
        f"SVI feature diagnostic: {int((~success).sum())} optimizer status failures "
        f"among {len(parameters)} SVI rows; {int((~success & finite_parameters).sum())} "
        "have finite parameter vectors and were retained as SVI-GP inputs to keep "
        f"the common date/pair set. Dates: {failed_dates or 'none'}."
    )


def _add_prediction_rows(
    output: list[dict[str, object]],
    evaluation: str,
    outer_split: str,
    training_condition: str,
    horizon: int,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    selected_predictions: dict[str, np.ndarray],
    quantiles: np.ndarray,
    date_to_index: dict[pd.Timestamp, int],
    volatility: dict[pd.Timestamp, float],
    volatility_low: float,
    volatility_high: float,
) -> np.ndarray:
    test_input_indices = np.array(
        [date_to_index[value] for value in test_pairs.input_date], dtype=int
    )
    test_target_indices = np.array(
        [date_to_index[value] for value in test_pairs.target_date], dtype=int
    )
    test_inputs = quantiles[test_input_indices]
    test_targets = quantiles[test_target_indices]
    persistence_losses = np.sqrt(
        squared_w2_per_pair(test_inputs, test_targets)
    )
    expected_pair_index = test_pairs[["input_date", "target_date"]].reset_index(
        drop=True
    )
    for model_name, predictions in selected_predictions.items():
        if predictions.shape != test_targets.shape:
            raise AssertionError(
                f"{model_name} does not share the h={horizon} test shape"
            )
        observed_losses = np.sqrt(squared_w2_per_pair(predictions, test_targets))
        for index, pair in enumerate(test_pairs.itertuples(index=False)):
            terminal_iv = volatility.get(pair.target_date, np.nan)
            regime = (
                "low_vol"
                if np.isfinite(terminal_iv) and terminal_iv <= volatility_low
                else "high_vol"
                if np.isfinite(terminal_iv) and terminal_iv >= volatility_high
                else "middle"
            )
            output.append(
                {
                    "evaluation": evaluation,
                    "outer_split": outer_split,
                    "training_condition": training_condition,
                    "horizon_sessions": horizon,
                    "input_date": pair.input_date,
                    "target_date": pair.target_date,
                    "model": model_name,
                    "w2": float(observed_losses[index]),
                    "w2_squared": float(observed_losses[index] ** 2),
                    "terminal_date_30d_atm_iv": terminal_iv,
                    "low_volatility_iv_cutoff": volatility_low,
                    "high_volatility_iv_cutoff": volatility_high,
                    "volatility_regime": regime,
                    "n_train_pairs": len(train_pairs),
                }
            )
    reference = {
        tuple(row)
        for row in expected_pair_index.itertuples(index=False, name=None)
    }
    for model_name in selected_predictions:
        matched = {
            (row["input_date"], row["target_date"])
            for row in output
            if row["evaluation"] == evaluation
            and row["outer_split"] == outer_split
            and row["training_condition"] == training_condition
            and row["horizon_sessions"] == horizon
            and row["model"] == model_name
        }
        if matched != reference:
            raise AssertionError(f"{model_name} does not use shared test pair indices")
    return persistence_losses


def _assert_shared_pair_indices(predictions: pd.DataFrame) -> None:
    group_columns = [
        "evaluation",
        "outer_split",
        "training_condition",
        "horizon_sessions",
    ]
    for group_key, group in predictions.groupby(group_columns, sort=False):
        pair_sets = [
            set(zip(model_rows["input_date"], model_rows["target_date"]))
            for _, model_rows in group.groupby("model", sort=False)
        ]
        if pair_sets and any(pair_set != pair_sets[0] for pair_set in pair_sets[1:]):
            raise AssertionError(
                f"Models do not share train/test evaluation pairs for {group_key}"
            )


def _pair_index_signature(pairs: pd.DataFrame) -> str:
    pair_values = pairs[["input_date", "target_date"]]
    hashed_values = pd.util.hash_pandas_object(
        pair_values,
        index=False,
    ).to_numpy()
    return hashlib.sha256(hashed_values.tobytes()).hexdigest()


def _assert_shared_index_signatures(selected: pd.DataFrame) -> None:
    group_columns = [
        "evaluation",
        "outer_split",
        "training_condition",
        "horizon_sessions",
    ]
    for group_key, group in selected.groupby(group_columns, sort=False):
        for column in ("train_pair_index_hash", "test_pair_index_hash"):
            if group[column].nunique(dropna=False) != 1:
                raise AssertionError(
                    f"Models do not share {column} for {group_key}"
                )


def _inference_for_group(
    predictions: pd.DataFrame,
    evaluation: str,
    training_condition: str,
    regime: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    subset = predictions.loc[
        (predictions["evaluation"] == evaluation)
        & (predictions["training_condition"] == training_condition)
    ]
    if regime is not None:
        subset = subset.loc[subset["volatility_regime"] == regime]
    persistence = subset.loc[subset["model"] == "Persistence"]
    rows = []
    for (horizon, model), group in subset.loc[
        subset["model"] != "Persistence"
    ].groupby(["horizon_sessions", "model"], sort=True):
        group = group.sort_values("target_date")
        base = persistence.loc[
            persistence["horizon_sessions"] == horizon
        ].sort_values("target_date")
        keys = ["input_date", "target_date"]
        group = group.merge(
            base[keys + ["w2", "w2_squared"]],
            on=keys,
            how="inner",
            suffixes=("", "_persistence"),
            validate="one_to_one",
        )
        if group.empty:
            continue
        for loss, column in (("W2", "w2"), ("Squared W2", "w2_squared")):
            persistence_losses = group[f"{column}_persistence"].to_numpy(float)
            model_losses = group[column].to_numpy(float)
            differential = persistence_losses - model_losses
            mean_persistence = float(persistence_losses.mean())
            mean_skill = 1 - float(model_losses.mean()) / mean_persistence
            influence = (
                differential - mean_skill * persistence_losses
            ) / mean_persistence
            primary_lags = max(int(horizon) - 1, 0)
            sensitivity_lags = 5 if horizon == 1 else 10
            standard_error = float(
                np.sqrt(_hac_mean_variance(influence, primary_lags))
            )
            sensitivity_error = float(
                np.sqrt(_hac_mean_variance(influence, sensitivity_lags))
            )
            differential_error = float(
                np.sqrt(_hac_mean_variance(differential, primary_lags))
            )
            t_stat = (
                float(differential.mean() / differential_error)
                if differential_error > 0
                else np.nan
            )
            bootstrap_h = _moving_block_bootstrap_ci(
                persistence_losses, model_losses, int(horizon), rng
            )
            bootstrap_10 = _moving_block_bootstrap_ci(
                persistence_losses, model_losses, 10, rng
            )
            rows.append(
                {
                    "evaluation": evaluation,
                    "training_condition": training_condition,
                    "regime": regime or "all",
                    "horizon_sessions": int(horizon),
                    "model": model,
                    "loss": loss,
                    "n_pairs": len(group),
                    "mean_w2": float(group[column].mean())
                    if loss == "W2"
                    else float(np.sqrt(group["w2_squared"].mean())),
                    "mean_squared_w2": float(group["w2_squared"].mean()),
                    "mean_skill": mean_skill,
                    "nw_se_skill": standard_error,
                    "nw_ci_lower": mean_skill - 1.96 * standard_error,
                    "nw_ci_upper": mean_skill + 1.96 * standard_error,
                    "nw_lags_primary": primary_lags,
                    "nw_lags_sensitivity": sensitivity_lags,
                    "nw_ci_sensitivity_lower": mean_skill
                    - 1.96 * sensitivity_error,
                    "nw_ci_sensitivity_upper": mean_skill
                    + 1.96 * sensitivity_error,
                    "nw_t_stat": t_stat,
                    "nw_p_value": (
                        float(2 * norm.sf(abs(t_stat)))
                        if np.isfinite(t_stat)
                        else np.nan
                    ),
                    "bootstrap_h_ci_lower": bootstrap_h[0],
                    "bootstrap_h_ci_upper": bootstrap_h[1],
                    "bootstrap_10_ci_lower": bootstrap_10[0],
                    "bootstrap_10_ci_upper": bootstrap_10[1],
                }
            )
    return pd.DataFrame(rows)


def _build_loss_differentials(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "evaluation",
        "outer_split",
        "training_condition",
        "horizon_sessions",
        "input_date",
        "target_date",
    ]
    persistence = predictions.loc[
        predictions["model"] == "Persistence",
        keys + ["w2", "w2_squared"],
    ].rename(
        columns={
            "w2": "persistence_w2",
            "w2_squared": "persistence_w2_squared",
        }
    )
    model_rows = predictions.loc[predictions["model"] != "Persistence"].merge(
        persistence,
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    model_rows["d_t_w2"] = model_rows["persistence_w2"] - model_rows["w2"]
    model_rows["d_t_squared_w2"] = (
        model_rows["persistence_w2_squared"] - model_rows["w2_squared"]
    )
    return model_rows[
        keys
        + [
            "model",
            "persistence_w2",
            "w2",
            "d_t_w2",
            "persistence_w2_squared",
            "w2_squared",
            "d_t_squared_w2",
        ]
    ]


def _holm_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(order) - rank) * values[index]))
        adjusted[index] = running_max
    return adjusted


def _fit_one_outer_split(
    evaluation: str,
    outer_split: str,
    training_condition: str,
    horizon: int,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    dates: list[pd.Timestamp],
    date_to_index: dict[pd.Timestamp, int],
    quantiles: np.ndarray,
    svi_by_date: dict[pd.Timestamp, np.ndarray],
    volatility: dict[pd.Timestamp, float],
    include_sensitivity: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    selected_predictions: dict[str, np.ndarray] = {}
    curve_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    train_input_indices = np.array(
        [date_to_index[value] for value in train_pairs.input_date], dtype=int
    )
    train_target_indices = np.array(
        [date_to_index[value] for value in train_pairs.target_date], dtype=int
    )
    test_input_indices = np.array(
        [date_to_index[value] for value in test_pairs.input_date], dtype=int
    )
    test_target_indices = np.array(
        [date_to_index[value] for value in test_pairs.target_date], dtype=int
    )
    train_inputs = quantiles[train_input_indices]
    train_targets = quantiles[train_target_indices]
    test_inputs = quantiles[test_input_indices]
    test_targets = quantiles[test_target_indices]
    train_svi = np.stack([svi_by_date[value] for value in train_pairs.input_date])
    test_svi = np.stack([svi_by_date[value] for value in test_pairs.input_date])

    selected_predictions["Persistence"] = project_quantile_matrix(
        test_inputs,
        np.zeros_like(test_inputs),
    )

    for model_name, model_kind in MODEL_SPECS.items():
        selected, model_curve, inner_fold_count = _tune_model(
            model_name,
            model_kind,
            train_pairs,
            date_to_index,
            quantiles,
            svi_by_date,
            horizon,
            evaluation,
            outer_split,
        )
        for row in model_curve:
            row["training_condition"] = training_condition
        curve_rows.extend(model_curve)
        forecast, pca_count = _fit_selected_and_predict(
            model_kind,
            train_inputs,
            train_targets,
            test_inputs,
            train_svi,
            test_svi,
            selected,
        )
        selected_predictions[model_name] = forecast
        training_features, _, _ = _feature_pair(
            model_kind,
            train_inputs,
            train_inputs[:1],
            train_svi,
            train_svi[:1],
        )
        median_gamma = (
            1.0 / median_distance_scale(pairwise_squared_distances(training_features))
            if selected["gamma_multiplier"] is not None
            else np.nan
        )
        selected_rows.append(
            {
                "evaluation": evaluation,
                "outer_split": outer_split,
                "training_condition": training_condition,
                "horizon_sessions": horizon,
                "model": model_name,
                "n_train_pairs": len(train_pairs),
                "n_test_pairs": len(test_pairs),
                "train_pair_index_hash": _pair_index_signature(train_pairs),
                "test_pair_index_hash": _pair_index_signature(test_pairs),
                "selected_gamma_multiplier": selected["gamma_multiplier"],
                "selected_median_heuristic_gamma": median_gamma,
                "selected_gamma": (
                    selected["gamma_multiplier"] * median_gamma
                    if selected["gamma_multiplier"] is not None
                    else np.nan
                ),
                "selected_regularization_noise_ratio": selected[
                    "regularization_noise_ratio"
                ],
                "cv_squared_w2_after_projection": selected["cv_squared_w2"],
                "inner_fold_count": inner_fold_count,
                "pca_component_count": pca_count,
            }
        )
        if include_sensitivity:
            test_persistence_squared = squared_w2_per_pair(
                test_inputs, test_targets
            )
            sensitivity_rows.extend(
                _fit_full_sweep(
                    model_name,
                    model_kind,
                    evaluation,
                    outer_split,
                    training_condition,
                    train_pairs,
                    test_pairs,
                    date_to_index,
                    quantiles,
                    svi_by_date,
                    horizon,
                    test_persistence_squared,
                )
            )

    legacy_model = WassersteinAutoregression().fit(train_inputs, train_targets)
    legacy_levels = legacy_model.predict(test_inputs)
    legacy_delta = legacy_levels - test_inputs
    selected_predictions["Wasserstein autoregression (legacy)"] = (
        project_quantile_matrix(test_inputs, legacy_delta)
    )

    train_terminal_vol = np.array(
        [volatility.get(value, np.nan) for value in train_pairs.target_date],
        dtype=float,
    )
    finite_training_vol = train_terminal_vol[np.isfinite(train_terminal_vol)]
    if len(finite_training_vol) < 3:
        raise ValueError(f"Insufficient training ATM IV for h={horizon}")
    volatility_low, volatility_high = np.quantile(
        finite_training_vol, [1 / 3, 2 / 3]
    )
    predictions: list[dict[str, object]] = []
    _add_prediction_rows(
        predictions,
        evaluation,
        outer_split,
        training_condition,
        horizon,
        train_pairs,
        test_pairs,
        selected_predictions,
        quantiles,
        date_to_index,
        volatility,
        float(volatility_low),
        float(volatility_high),
    )
    return predictions, curve_rows, selected_rows, sensitivity_rows


def run() -> dict[str, pd.DataFrame]:
    dates, quantiles, _, _, _ = load_rnds(str(RND_FILE), n_quantiles=200)
    normalized_dates = [
        pd.Timestamp(value).tz_localize(None).normalize() for value in dates
    ]
    quantiles = normalize_quantile_matrix(quantiles)
    date_to_index = {value: index for index, value in enumerate(normalized_dates)}
    sessions = nyse_sessions(
        min(normalized_dates).date(), max(normalized_dates).date()
    )
    svi_by_date = _svi_by_date(normalized_dates)
    volatility = _volatility_by_date()
    split_80 = sessions[int(np.floor(0.8 * len(sessions)))]
    rolling_cutoffs = [
        sessions[int(np.floor(fraction * len(sessions)))]
        for fraction in (0.50, 0.65, 0.80)
    ]

    predictions: list[dict[str, object]] = []
    cv_curves: list[dict[str, object]] = []
    selected_parameters: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    for horizon in (1, 5):
        pairs = build_trading_pairs(sessions, set(normalized_dates), horizon)
        rolling_ranges = list(zip(rolling_cutoffs, rolling_cutoffs[1:] + [None]))
        for fold_index, (cutoff, next_cutoff) in enumerate(rolling_ranges):
            full_train_pairs = pairs.loc[
                pairs["target_date"] < cutoff
            ].reset_index(drop=True)
            test_mask = pairs["target_date"] >= cutoff
            if next_cutoff is not None:
                test_mask &= pairs["target_date"] < next_cutoff
            test_pairs = pairs.loc[test_mask].reset_index(drop=True)
            split_name = f"origin_{(50, 65, 80)[fold_index]}pct"
            for training_condition in (
                "full_training",
                "excluding_april_2025",
            ):
                train_pairs = full_train_pairs
                if training_condition == "excluding_april_2025":
                    april_mask = (
                        full_train_pairs["input_date"].dt.to_period("M")
                        == "2025-04"
                    ) | (
                        full_train_pairs["target_date"].dt.to_period("M")
                        == "2025-04"
                    )
                    train_pairs = full_train_pairs.loc[~april_mask].reset_index(
                        drop=True
                    )
                result = _fit_one_outer_split(
                    "rolling_origin",
                    split_name,
                    training_condition,
                    horizon,
                    train_pairs,
                    test_pairs,
                    normalized_dates,
                    date_to_index,
                    quantiles,
                    svi_by_date,
                    volatility,
                    include_sensitivity=(
                        training_condition == "full_training"
                    ),
                )
                predictions.extend(result[0])
                cv_curves.extend(result[1])
                selected_parameters.extend(result[2])
                sensitivity.extend(result[3])

        fixed_train = pairs.loc[pairs["target_date"] < split_80].reset_index(drop=True)
        fixed_test = pairs.loc[pairs["target_date"] >= split_80].reset_index(drop=True)
        if len(fixed_test) != 143:
            raise AssertionError(
                f"Expected 143 fixed holdout pairs for h={horizon}, found {len(fixed_test)}"
            )
        for training_condition in ("full_training", "excluding_april_2025"):
            train_pairs = fixed_train
            if training_condition == "excluding_april_2025":
                april_mask = (
                    fixed_train["input_date"].dt.to_period("M") == "2025-04"
                ) | (fixed_train["target_date"].dt.to_period("M") == "2025-04")
                train_pairs = fixed_train.loc[~april_mask].reset_index(drop=True)
            result = _fit_one_outer_split(
                "fixed_80_20",
                "fixed_80_20",
                training_condition,
                horizon,
                train_pairs,
                fixed_test,
                normalized_dates,
                date_to_index,
                quantiles,
                svi_by_date,
                volatility,
                include_sensitivity=(training_condition == "full_training"),
            )
            predictions.extend(result[0])
            cv_curves.extend(result[1])
            selected_parameters.extend(result[2])
            sensitivity.extend(result[3])

    prediction_frame = pd.DataFrame(predictions)
    prediction_frame["input_date"] = pd.to_datetime(prediction_frame["input_date"])
    prediction_frame["target_date"] = pd.to_datetime(prediction_frame["target_date"])
    _assert_shared_pair_indices(prediction_frame)
    differential_frame = _build_loss_differentials(prediction_frame)
    curve_frame = pd.DataFrame(cv_curves)
    selected_frame = pd.DataFrame(selected_parameters)
    sensitivity_frame = pd.DataFrame(sensitivity)
    _assert_shared_index_signatures(selected_frame)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_frames = []
    for evaluation in EVALUATIONS:
        for training_condition in ("full_training", "excluding_april_2025"):
            inference_frames.append(
                _inference_for_group(
                    prediction_frame,
                    evaluation,
                    training_condition,
                    None,
                    rng,
                )
            )
            for regime in ("low_vol", "high_vol"):
                inference_frames.append(
                    _inference_for_group(
                        prediction_frame,
                        evaluation,
                        training_condition,
                        regime,
                        rng,
                    )
                )
    inference_frame = pd.concat(inference_frames, ignore_index=True)
    holm_groups = inference_frame.groupby(
        ["evaluation", "training_condition", "regime"]
    ).groups
    for _, indices in holm_groups.items():
        group_indices = list(indices)
        inference_frame.loc[group_indices, "holm_adjusted_p_value"] = _holm_adjust(
            inference_frame.loc[group_indices, "nw_p_value"].to_numpy(float)
        )

    summary = (
        prediction_frame.groupby(
            [
                "evaluation",
                "training_condition",
                "horizon_sessions",
                "model",
            ],
            as_index=False,
        )
        .agg(
            n_pairs=("w2", "size"),
            mean_w2=("w2", "mean"),
            mean_squared_w2=("w2_squared", "mean"),
        )
    )
    persistence = summary.loc[
        summary["model"] == "Persistence",
        [
            "evaluation",
            "training_condition",
            "horizon_sessions",
            "mean_w2",
            "mean_squared_w2",
        ],
    ].rename(
        columns={
            "mean_w2": "persistence_mean_w2",
            "mean_squared_w2": "persistence_mean_squared_w2",
        }
    )
    summary = summary.merge(
        persistence,
        on=["evaluation", "training_condition", "horizon_sessions"],
        how="left",
        validate="many_to_one",
    )
    summary["skill_vs_persistence_w2"] = 1 - (
        summary["mean_w2"] / summary["persistence_mean_w2"]
    )
    summary["skill_vs_persistence_squared_w2"] = 1 - (
        summary["mean_squared_w2"] / summary["persistence_mean_squared_w2"]
    )
    summary = summary.drop(
        columns=["persistence_mean_w2", "persistence_mean_squared_w2"]
    )

    regime_frame = (
        prediction_frame.loc[
            prediction_frame["volatility_regime"].isin(["low_vol", "high_vol"])
        ]
        .groupby(
            [
                "evaluation",
                "outer_split",
                "training_condition",
                "horizon_sessions",
                "model",
                "volatility_regime",
            ],
            as_index=False,
        )
        .agg(
            n_pairs=("w2", "size"),
            mean_w2=("w2", "mean"),
            mean_squared_w2=("w2_squared", "mean"),
        )
    )
    regime_persistence = regime_frame.loc[
        regime_frame["model"] == "Persistence",
        [
            "evaluation",
            "outer_split",
            "training_condition",
            "horizon_sessions",
            "volatility_regime",
            "mean_w2",
            "mean_squared_w2",
        ],
    ].rename(
        columns={
            "mean_w2": "persistence_mean_w2",
            "mean_squared_w2": "persistence_mean_squared_w2",
        }
    )
    regime_frame = regime_frame.merge(
        regime_persistence,
        on=[
            "evaluation",
            "outer_split",
            "training_condition",
            "horizon_sessions",
            "volatility_regime",
        ],
        how="left",
        validate="many_to_one",
    )
    regime_frame["skill_vs_persistence_w2"] = 1 - (
        regime_frame["mean_w2"] / regime_frame["persistence_mean_w2"]
    )
    regime_frame["skill_vs_persistence_squared_w2"] = 1 - (
        regime_frame["mean_squared_w2"] / regime_frame["persistence_mean_squared_w2"]
    )
    regime_frame = regime_frame.drop(
        columns=["persistence_mean_w2", "persistence_mean_squared_w2"]
    )

    for key, frame in (
        ("predictions", prediction_frame),
        ("differentials", differential_frame),
        ("summary", summary),
        ("cv_curve", curve_frame),
        ("selected", selected_frame),
        ("sensitivity", sensitivity_frame),
        ("inference", inference_frame),
        ("regimes", regime_frame),
    ):
        frame.to_csv(OUTPUTS[key], index=False)
    _append_report(
        summary,
        inference_frame,
        selected_frame,
        sensitivity_frame,
        regime_frame,
        curve_frame,
    )
    return {
        "predictions": prediction_frame,
        "summary": summary,
        "cv_curve": curve_frame,
        "selected": selected_frame,
        "sensitivity": sensitivity_frame,
        "inference": inference_frame,
        "regimes": regime_frame,
    }


def _append_report(
    summary: pd.DataFrame,
    inference: pd.DataFrame,
    selected: pd.DataFrame,
    sensitivity: pd.DataFrame,
    regimes: pd.DataFrame,
    cv_curve: pd.DataFrame,
) -> None:
    previous = REPORT_FILE.read_text(encoding="utf-8") if REPORT_FILE.exists() else ""
    marker = "\nPHASE 4 QUANTILE-CHANGE MODEL COMPARISON"
    if marker in previous:
        previous = previous.split(marker, maxsplit=1)[0]
    lines = [
        marker,
        (
            "Primary evaluation is rolling-origin with expanding training at the "
            "50%, 65%, and 80% calendar cutoffs and non-overlapping test blocks. "
            "The fixed 80/20 split at 2026-04-14 is secondary. Both use exact "
            "calendar h=1/h=5 pairs and common pair indices across models."
        ),
        (
            "All RNDs are represented by nondecreasing nonnegative quantiles on "
            "the fixed 200-point probability grid, rescaled to arithmetic mean "
            "one. Models predict target-minus-input quantile changes. Every "
            "forecast, including persistence, uses the same isotonic, nonnegative, "
            "mean-one projection before W2 scoring. The zero-change projection "
            "reproduces persistence exactly."
        ),
        (
            "CV uses three expanding forward-chaining folds (training prefixes "
            "50%, 65%, and 80% of each outer training window). Selection minimizes "
            "pooled inner validation squared W2 after projection. Every GP uses "
            "gamma multipliers {0.1, 0.3, 1, 3, 10} of the training median "
            "heuristic and standardized-output noise/regularization ratios "
            "{1e-6, 1e-4, 1e-2, 1e-1, 1}. Ridge autoregression uses the same "
            "regularization grid. Test performance uses CV-selected values only; "
            "the full parameter sweep for each outer split is separately labelled "
            "sensitivity-only. Training-condition sensitivity is reported both "
            "with all eligible training pairs and after dropping every training "
            "pair with either endpoint in April 2025."
        ),
        (
            "Models: PCA-factor quantile-change GP (main), PCA-factor linear "
            "autoregression, GP on volatility/skewness/kurtosis, GP on SVI "
            "parameters, Wasserstein-input GP predicting quantile changes "
            "(continuity of Wasserstein input geometry), persistence, and legacy "
            "Wasserstein autoregression. This phase's Wasserstein GP row is "
            "change-output/CV-tuned and is not the previous variance-regression "
            "plus barycenter implementation."
        ),
        _svi_fit_status_note(),
        (
            f"Bootstrap resamples={BOOTSTRAP_RESAMPLES}, seed={BOOTSTRAP_SEED}; "
            "block lengths h and 10. NW primary lags h-1; sensitivity 5 at h=1 "
            "and 10 at h=5. Holm adjustment is performed across model, horizon, "
            "and loss comparisons separately within each evaluation, training-"
            "condition, and regime group."
        ),
        (
            "Execution note: the first report-formatting pass raised a KeyError "
            "when it requested paired inference for the persistence reference, "
            "which intentionally has no model-vs-persistence differential. The "
            "saved forecast and inference tables were intact; the formatter was "
            "corrected to report persistence as the zero-skill reference, and "
            "this report was regenerated from those tables."
        ),
        "Selected hyperparameters by outer split:",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{row.evaluation} {row.outer_split} {row.training_condition} "
            f"h={row.horizon_sessions} "
            f"{row.model}: gamma multiplier={row.selected_gamma_multiplier}; "
            f"gamma={row.selected_gamma}; regularization/noise ratio="
            f"{row.selected_regularization_noise_ratio}; inner-CV projected "
            f"squared-W2={row.cv_squared_w2_after_projection:.8e}; "
            f"n_train={row.n_train_pairs}; n_test={row.n_test_pairs}."
        )
    lines.append("CV selected test summary; scores are descriptive pipeline results:")
    inference_lookup = inference.loc[inference["regime"] == "all"].set_index(
        [
            "evaluation",
            "training_condition",
            "horizon_sessions",
            "model",
            "loss",
        ]
    )
    for row in summary.itertuples(index=False):
        for loss_label, loss_name in (("W2", "w2"), ("Squared W2", "squared_w2")):
            score = row.mean_w2 if loss_name == "w2" else row.mean_squared_w2
            skill = (
                row.skill_vs_persistence_w2
                if loss_name == "w2"
                else row.skill_vs_persistence_squared_w2
            )
            if row.model == "Persistence":
                lines.append(
                    f"{row.evaluation} {row.training_condition} "
                    f"h={row.horizon_sessions} Persistence {loss_label}: "
                    f"mean loss={score:.8e}, reference skill={skill:.8e}, "
                    f"n_pairs={row.n_pairs}."
                )
                continue
            inference_row = inference_lookup.loc[
                (
                    row.evaluation,
                    row.training_condition,
                    row.horizon_sessions,
                    row.model,
                    loss_label,
                )
            ]
            lines.append(
                f"{row.evaluation} {row.training_condition} "
                f"h={row.horizon_sessions} {row.model} {loss_label}: "
                f"mean loss={score:.8e}, skill={skill:.8e}, n_pairs={row.n_pairs}; "
                f"NW95=[{inference_row.nw_ci_lower:.8e},"
                f"{inference_row.nw_ci_upper:.8e}], "
                f"NW sensitivity={inference_row.nw_lags_sensitivity} lags "
                f"[{inference_row.nw_ci_sensitivity_lower:.8e},"
                f"{inference_row.nw_ci_sensitivity_upper:.8e}], "
                f"MBB-h95=[{inference_row.bootstrap_h_ci_lower:.8e},"
                f"{inference_row.bootstrap_h_ci_upper:.8e}], "
                f"MBB-10=[{inference_row.bootstrap_10_ci_lower:.8e},"
                f"{inference_row.bootstrap_10_ci_upper:.8e}], "
                f"unadjusted p={inference_row.nw_p_value:.6g}, "
                f"Holm p={inference_row.holm_adjusted_p_value:.6g}."
            )
    lines.append(
        "High-/low-volatility regimes use terminal-date 30d ATM IV terciles "
        "estimated from each outer training window; the low/high cutoffs are "
        "recorded per prediction in the predictions CSV."
    )
    for row in regimes.itertuples(index=False):
        lines.append(
            f"{row.evaluation} {row.outer_split} {row.training_condition} "
            f"h={row.horizon_sessions} "
            f"{row.model} {row.volatility_regime}: n={row.n_pairs}, "
            f"mean W2={row.mean_w2:.8e}, mean squared W2="
            f"{row.mean_squared_w2:.8e}, W2 skill={row.skill_vs_persistence_w2:.8e}, "
            f"squared-W2 skill={row.skill_vs_persistence_squared_w2:.8e}."
        )
    lines.append(
        f"Full sweep sensitivity only across all outer splits: "
        f"{OUTPUTS['sensitivity'].name}; "
        f"CV loss curve: {OUTPUTS['cv_curve'].name} "
        f"({len(cv_curve)} rows); selected values in "
        f"{OUTPUTS['selected'].name}. The full grid is reported for full_training; "
        "the April-excluded condition is evaluated at CV-selected values only. "
        "No full-sweep row is presented as a selected result."
    )
    REPORT_FILE.write_text(
        previous.rstrip() + "\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
