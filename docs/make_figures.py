from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
RESULT_DIR = ROOT / "06_evaluation" / "results"
RND_FILE = ROOT / "02_rnd_extraction" / "data" / "rnd_30d.csv"

sys.path.insert(0, str(ROOT / "04_gp_model"))
sys.path.insert(0, str(ROOT / "06_evaluation"))

from fit_gp import load_rnds  # noqa: E402
from run_phase4_quantile_change import (  # noqa: E402
    _fit_selected_and_predict,
    _svi_by_date,
)
from trading_calendar_pairs import build_trading_pairs, nyse_sessions  # noqa: E402


MODEL = "Quantile-change GP (PCA factors)"
MODELS = [
    "Quantile-change GP (PCA factors)",
    "Linear autoregression (PCA quantile factors)",
    "GP (vol/skew/kurtosis)",
    "GP (SVI parameters)",
    "Wasserstein-input GP (quantile changes)",
    "Wasserstein autoregression (legacy)",
    "Persistence",
]
EVALUATION_LABELS = {
    ("rolling_origin", 1): "Rolling h=1",
    ("rolling_origin", 5): "Rolling h=5",
    ("fixed_80_20", 1): "Fixed h=1",
    ("fixed_80_20", 5): "Fixed h=5",
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / name)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _skill_heatmap(metric: str, output: str, title: str) -> None:
    summary = _read("phase4_forecast_summary.csv")
    summary = summary.loc[summary.training_condition == "full_training"]
    columns = [
        ("rolling_origin", 1),
        ("rolling_origin", 5),
        ("fixed_80_20", 1),
        ("fixed_80_20", 5),
    ]
    value_column = f"skill_vs_persistence_{metric}"
    matrix = (
        summary.pivot(
            index="model",
            columns=["evaluation", "horizon_sessions"],
            values=value_column,
        )
        .reindex(index=MODELS, columns=pd.MultiIndex.from_tuples(columns))
        .to_numpy(dtype=float)
    )
    limit = float(np.nanmax(np.abs(matrix)))
    limit = max(limit, 1e-12)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    image = ax.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(columns)),
        [EVALUATION_LABELS[column] for column in columns],
        rotation=12,
        ha="right",
    )
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_title(title)
    ax.set_xlabel("Evaluation design and horizon")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    fig.colorbar(image, ax=ax, label=f"{metric.replace('_', ' ')} skill")
    _save(fig, output)


def _april_training_bars() -> None:
    summary = _read("phase4_forecast_summary.csv")
    summary = summary.loc[summary.model == MODEL]
    settings = [
        ("rolling_origin", 1),
        ("rolling_origin", 5),
        ("fixed_80_20", 1),
        ("fixed_80_20", 5),
    ]
    conditions = ["full_training", "excluding_april_2025"]
    value_column = "skill_vs_persistence_squared_w2"
    x = np.arange(len(settings))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for offset, condition in enumerate(conditions):
        values = []
        for evaluation, horizon in settings:
            row = summary.loc[
                (summary.evaluation == evaluation)
                & (summary.horizon_sessions == horizon)
                & (summary.training_condition == condition)
            ]
            values.append(float(row.iloc[0][value_column]))
        ax.bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=condition.replace("_", " "),
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, [EVALUATION_LABELS[setting] for setting in settings])
    ax.set_ylabel("Squared-W2 skill vs persistence")
    ax.set_title("Main quantile-change GP: April-2025 training sensitivity")
    ax.legend(frameon=False)
    _save(fig, "03_april_training_sensitivity.png")


def _volatility_bars() -> None:
    inference = _read("phase4_inference.csv")
    inference = inference.loc[
        (inference.evaluation == "rolling_origin")
        & (inference.training_condition == "full_training")
        & (inference.model == MODEL)
        & (inference.loss == "Squared W2")
        & inference.regime.isin(["low_vol", "high_vol"])
    ]
    horizons = [1, 5]
    regimes = ["low_vol", "high_vol"]
    x = np.arange(len(horizons))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for offset, regime in enumerate(regimes):
        values = [
            float(
                inference.loc[
                    (inference.horizon_sessions == horizon)
                    & (inference.regime == regime),
                    "mean_skill",
                ].iloc[0]
            )
            for horizon in horizons
        ]
        ax.bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=regime.replace("_", " "),
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, [f"h={horizon}" for horizon in horizons])
    ax.set_ylabel("Squared-W2 skill vs persistence")
    ax.set_title("Main quantile-change GP by terminal-date volatility regime")
    ax.legend(frameon=False)
    _save(fig, "04_volatility_regimes.png")


def _cv_curve() -> None:
    curve = _read("phase4_cv_loss_curve.csv")
    curve = curve.loc[
        (curve.model == MODEL)
        & (curve.evaluation == "rolling_origin")
        & (curve.outer_split == "origin_80pct")
        & (curve.training_condition == "full_training")
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for horizon in (1, 5):
        horizon_rows = curve.loc[curve.horizon_sessions == horizon]
        best_by_multiplier = (
            horizon_rows.sort_values("cv_squared_w2")
            .groupby("gamma_multiplier", as_index=False)
            .first()
            .sort_values("gamma_multiplier")
        )
        ax.plot(
            best_by_multiplier.gamma_multiplier,
            best_by_multiplier.cv_squared_w2,
            marker="o",
            label=f"h={horizon}",
        )
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel("Kernel median-heuristic multiplier")
    ax.set_ylabel("Pooled forward-CV projected squared W2")
    ax.set_title("Main model CV curve, 80% rolling-origin training window")
    ax.legend(title="Horizon", frameon=False)
    _save(fig, "05_cv_loss_curve.png")


def _forest_plot() -> None:
    inference = _read("phase4_inference.csv")
    inference = inference.loc[
        (inference.evaluation == "rolling_origin")
        & (inference.training_condition == "full_training")
        & (inference.regime == "all")
        & (inference.loss == "Squared W2")
        & (inference.model != "Persistence")
    ]
    plot_models = [model for model in MODELS if model != "Persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True)
    for ax, horizon in zip(axes, (1, 5)):
        rows = (
            inference.loc[inference.horizon_sessions == horizon]
            .set_index("model")
            .reindex(plot_models)
        )
        y = np.arange(len(plot_models))
        mean = rows.mean_skill.to_numpy(dtype=float)
        lower = rows.nw_ci_lower.to_numpy(dtype=float)
        upper = rows.nw_ci_upper.to_numpy(dtype=float)
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack((mean - lower, upper - mean)),
            fmt="o",
            color="#2b5c8a",
            ecolor="#7799b5",
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_yticks(y, plot_models)
        ax.invert_yaxis()
        ax.set_title(f"h={horizon}; n={int(rows.n_pairs.iloc[0])}")
        ax.set_xlabel("Squared-W2 skill (Newey-West 95% CI)")
    fig.suptitle("Rolling-origin full-training model intervals")
    _save(fig, "06_newey_west_forest.png")


def _example_quantiles() -> None:
    dates, raw_quantiles, _, _, _ = load_rnds(str(RND_FILE), n_quantiles=200)
    dates = [
        pd.Timestamp(value).tz_localize(None).normalize() for value in dates
    ]
    quantiles = np.asarray(raw_quantiles, dtype=float)
    date_to_index = {value: index for index, value in enumerate(dates)}
    sessions = nyse_sessions(min(dates).date(), max(dates).date())
    split = sessions[int(np.floor(0.8 * len(sessions)))]
    pairs = build_trading_pairs(sessions, set(dates), 1)
    train_pairs = pairs.loc[pairs.target_date < split].reset_index(drop=True)
    test_pairs = pairs.loc[pairs.target_date >= split].reset_index(drop=True)

    prediction_rows = _read("phase4_forecast_predictions.csv")
    selected_pair = prediction_rows.loc[
        (prediction_rows.evaluation == "fixed_80_20")
        & (prediction_rows.training_condition == "full_training")
        & (prediction_rows.horizon_sessions == 1)
        & (prediction_rows.model == MODEL)
    ].sort_values("target_date").iloc[0]
    input_date = pd.Timestamp(selected_pair.input_date).normalize()
    target_date = pd.Timestamp(selected_pair.target_date).normalize()
    pair = test_pairs.loc[
        (test_pairs.input_date == input_date)
        & (test_pairs.target_date == target_date)
    ]
    if pair.empty:
        raise ValueError("Selected example pair is not in the fixed holdout pairs")

    train_inputs = quantiles[
        [date_to_index[value] for value in train_pairs.input_date]
    ]
    train_targets = quantiles[
        [date_to_index[value] for value in train_pairs.target_date]
    ]
    eval_inputs = quantiles[[date_to_index[input_date]]]
    train_svi_by_date = _svi_by_date(dates)
    train_svi = np.stack(
        [train_svi_by_date[value] for value in train_pairs.input_date]
    )
    eval_svi = np.stack([train_svi_by_date[input_date]])

    selected = _read("phase4_selected_hyperparameters.csv")
    selected = selected.loc[
        (selected.evaluation == "fixed_80_20")
        & (selected.training_condition == "full_training")
        & (selected.horizon_sessions == 1)
        & (selected.model == MODEL)
    ].iloc[0]
    forecast, _ = _fit_selected_and_predict(
        "pca_gp",
        train_inputs,
        train_targets,
        eval_inputs,
        train_svi,
        eval_svi,
        {
            "gamma_multiplier": float(selected.selected_gamma_multiplier),
            "regularization_noise_ratio": float(
                selected.selected_regularization_noise_ratio
            ),
        },
    )
    actual = quantiles[date_to_index[target_date]]
    probability = np.linspace(1e-4, 1 - 1e-4, quantiles.shape[1])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(probability, eval_inputs[0], label="Input quantile", color="#6b7280")
    ax.plot(
        probability,
        eval_inputs[0],
        label="Persistence",
        color="#111827",
        linestyle="--",
    )
    ax.plot(
        probability,
        forecast[0],
        label="Quantile-change GP forecast",
        color="#c2413b",
        linewidth=2,
    )
    ax.plot(
        probability,
        actual,
        label="Observed target quantile",
        color="#2563a6",
        linewidth=1.5,
    )
    ax.set_xlabel("Probability u")
    ax.set_ylabel("Forward-normalized quantile Q(u)")
    ax.set_title(
        f"One fixed-split h=1 example: {input_date.date()} to {target_date.date()}"
    )
    ax.legend(frameon=False)
    _save(fig, "07_example_quantile_forecast.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _skill_heatmap(
        "squared_w2",
        "01_squared_w2_skill_heatmap.png",
        "Squared-W2 skill relative to persistence",
    )
    _skill_heatmap(
        "w2",
        "02_w2_skill_heatmap.png",
        "W2 skill relative to persistence",
    )
    _april_training_bars()
    _volatility_bars()
    _cv_curve()
    _forest_plot()
    _example_quantiles()
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
