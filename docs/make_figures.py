from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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


def _animated_pipeline() -> None:
    labels = [
        "Option quotes",
        "30d smile + SVI",
        "Risk-neutral density",
        "Quantile forecast",
        "W2 evaluation",
    ]
    colors = ["#2563eb", "#7c3aed", "#0891b2", "#db2777", "#16a34a"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis("off")

    def update(frame: int) -> list:
        ax.clear()
        ax.set_xlim(0, len(labels))
        ax.set_ylim(0, 1)
        ax.axis("off")
        for index, (label, color) in enumerate(zip(labels, colors)):
            active = index == frame
            box = FancyBboxPatch(
                (index + 0.08, 0.34),
                0.78,
                0.34,
                boxstyle="round,pad=0.04,rounding_size=0.06",
                linewidth=2 if active else 1,
                edgecolor=color,
                facecolor=color if active else "#f8fafc",
            )
            ax.add_patch(box)
            ax.text(
                index + 0.47,
                0.51,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if active else "#0f172a",
                weight="bold" if active else "normal",
                wrap=True,
            )
            if index < len(labels) - 1:
                ax.annotate(
                    "",
                    xy=(index + 1.04, 0.51),
                    xytext=(index + 0.88, 0.51),
                    arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.8},
                )
        ax.text(
            len(labels) / 2,
            0.9,
            "From option data to a distributional score",
            ha="center",
            va="center",
            fontsize=14,
            color="#0f172a",
            weight="bold",
        )
        ax.text(
            len(labels) / 2,
            0.13,
            "Illustrative pipeline animation; no market observations shown",
            ha="center",
            va="center",
            fontsize=9,
            color="#64748b",
        )
        return list(ax.patches) + list(ax.texts)

    animation = FuncAnimation(fig, update, frames=len(labels), interval=900, blit=False)
    animation.save(
        FIGURE_DIR / "pipeline_walkthrough.gif",
        writer=PillowWriter(fps=1.2),
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def _animated_quantile_change() -> None:
    probability = np.linspace(0.01, 0.99, 240)
    base = np.exp(0.72 * (probability - 0.5))
    base /= base.mean()
    shape = np.sin(np.pi * (probability - 0.5)) * (probability - 0.5)
    target = np.maximum(base + 0.32 * shape, 0.02)
    target /= target.mean()

    fig, ax = plt.subplots(figsize=(8.5, 5))

    def update(frame: int) -> list:
        fraction = frame / 35
        forecast = np.maximum(base + fraction * (target - base), 0.02)
        forecast /= forecast.mean()
        ax.clear()
        ax.plot(
            probability,
            base,
            color="#475569",
            linestyle="--",
            linewidth=2,
            label="Persistence (zero change)",
        )
        ax.plot(
            probability,
            forecast,
            color="#db2777",
            linewidth=2.7,
            label="Projected forecast",
        )
        ax.plot(
            probability,
            target,
            color="#2563eb",
            alpha=0.55,
            linewidth=1.8,
            label="Illustrative target",
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, max(target) * 1.12),
            xlabel="Probability u",
            ylabel="Forward-normalized quantile Q(u)",
            title="Quantile-change forecast and validity projection",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", frameon=False)
        ax.text(
            0.99,
            0.02,
            "Synthetic illustration only — not SPX data",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#64748b",
        )
        return list(ax.lines) + list(ax.texts)

    animation = FuncAnimation(fig, update, frames=36, interval=100, blit=False)
    animation.save(
        FIGURE_DIR / "quantile_change_illustration.gif",
        writer=PillowWriter(fps=12),
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def _animated_w2_transport() -> None:
    probability = np.linspace(0.03, 0.97, 180)
    source = np.exp(0.85 * (probability - 0.5))
    source /= source.mean()
    target = np.exp(1.05 * (probability - 0.54))
    target /= target.mean()
    selected = np.linspace(8, len(probability) - 9, 15, dtype=int)

    fig, (ax_curve, ax_cost) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.65, 1]}
    )

    def update(frame: int) -> list:
        fraction = frame / 27
        moving = source[selected] + fraction * (target[selected] - source[selected])
        ax_curve.clear()
        ax_cost.clear()
        ax_curve.plot(probability, source, color="#2563eb", linewidth=2.2, label="Source Q₁(u)")
        ax_curve.plot(probability, target, color="#db2777", linewidth=2.2, label="Target Q₂(u)")
        ax_curve.scatter(source[selected], moving, color="#64748b", s=18, zorder=3)
        ax_curve.scatter(target[selected], moving, color="#db2777", s=18, zorder=3)
        for start, end, y in zip(source[selected], target[selected], moving):
            ax_curve.plot([start, end], [y, y], color="#94a3b8", linewidth=0.8, alpha=0.65)
        ax_curve.set(
            xlabel="Probability level u",
            ylabel="Quantile value Q(u)",
            title="Monotone quantile-to-quantile transport",
            xlim=(0, 1),
        )
        ax_curve.legend(frameon=False, loc="upper left")
        ax_curve.grid(alpha=0.18)
        cost = (target - source) ** 2
        ax_cost.fill_between(probability, cost, color="#7c3aed", alpha=0.18)
        ax_cost.plot(probability, cost, color="#7c3aed", linewidth=2.2)
        ax_cost.axvline(fraction, color="#f59e0b", linestyle="--", linewidth=1.5)
        ax_cost.set(
            xlabel="Probability level u",
            ylabel="Squared quantile gap",
            title="W₂² integrates squared gaps",
            xlim=(0, 1),
            ylim=(0, float(cost.max()) * 1.15),
        )
        fig.suptitle("One-dimensional Wasserstein-2 geometry", color="#0f172a", weight="bold")
        fig.text(
            0.5,
            0.01,
            "Synthetic curves for explanation only — not market observations",
            ha="center",
            fontsize=8,
            color="#64748b",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        return list(ax_curve.lines) + list(ax_curve.collections) + list(ax_cost.lines) + list(ax_cost.collections)

    animation = FuncAnimation(fig, update, frames=28, interval=110, blit=False)
    animation.save(
        FIGURE_DIR / "wasserstein_transport.gif",
        writer=PillowWriter(fps=10),
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def _animated_kernel_similarity() -> None:
    probability = np.linspace(0.01, 0.99, 220)
    query = np.exp(0.8 * (probability - 0.5))
    query /= query.mean()
    shape_offsets = np.linspace(-0.9, 0.9, 7)
    candidates = []
    for offset in shape_offsets:
        centered = probability - 0.5
        q = np.exp((0.8 + 0.25 * offset) * centered + 0.6 * offset * centered**2)
        candidates.append(q / q.mean())
    candidates_array = np.asarray(candidates)
    distances = np.mean((candidates_array - query) ** 2, axis=1)
    gamma_grid = np.geomspace(500, 0.08, 32)

    fig, (ax_curve, ax_kernel) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    def update(frame: int) -> list:
        gamma = gamma_grid[frame]
        weights = np.exp(-gamma * distances)
        weights /= weights.sum()
        ax_curve.clear()
        ax_kernel.clear()
        ax_curve.plot(probability, query, color="#0f172a", linewidth=2.4, label="Query density quantile")
        for index, candidate in enumerate(candidates_array):
            ax_curve.plot(
                probability,
                candidate,
                color="#94a3b8",
                alpha=0.35 + 0.6 * weights[index],
                linewidth=0.8 + 2.0 * weights[index],
            )
        ax_curve.set(
            xlabel="Probability level u",
            ylabel="Forward-normalized Q(u)",
            title="Training shapes around a query",
            xlim=(0, 1),
        )
        ax_curve.legend(frameon=False, loc="upper left")
        ax_curve.grid(alpha=0.18)
        colors = ["#2563eb" if i == np.argmin(distances) else "#7c3aed" for i in range(len(weights))]
        ax_kernel.bar(np.arange(1, len(weights) + 1), weights, color=colors, width=0.72)
        ax_kernel.set(
            xlabel="Illustrative training distribution",
            ylabel="Normalized kernel weight",
            title=f"Similarity weights · γ = {gamma:.2f}",
            ylim=(0, 1),
            xticks=np.arange(1, len(weights) + 1),
        )
        ax_kernel.grid(axis="y", alpha=0.18)
        fig.suptitle("Wasserstein kernel: distance becomes local similarity", color="#0f172a", weight="bold")
        fig.text(
            0.5,
            0.01,
            "Schematic quantiles and weights; animation does not use fitted model results",
            ha="center",
            fontsize=8,
            color="#64748b",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        return list(ax_curve.lines) + list(ax_kernel.patches) + list(ax_kernel.texts)

    animation = FuncAnimation(fig, update, frames=len(gamma_grid), interval=110, blit=False)
    animation.save(
        FIGURE_DIR / "wasserstein_kernel_weights.gif",
        writer=PillowWriter(fps=10),
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def _animated_calendar_pairs() -> None:
    sessions = np.arange(1, 14)
    missing_session = 7
    horizons = (1, 5)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)

    def update(frame: int) -> list:
        visible = frame + 6
        for ax, horizon in zip(axes, horizons):
            ax.clear()
            ax.set_xlim(0.3, len(sessions) + 0.7)
            ax.set_ylim(-0.7, 1.3)
            ax.set_yticks([])
            ax.set_title(f"h = {horizon} trading session{'s' if horizon != 1 else ''}", loc="left", fontsize=10)
            ax.scatter(sessions, np.zeros_like(sessions), color="#2563eb", s=35, zorder=3)
            ax.scatter([missing_session], [0], color="#dc2626", marker="x", s=85, linewidth=2.5, zorder=4)
            for origin in range(1, visible - horizon + 1):
                target = origin + horizon
                if target > len(sessions):
                    break
                involves_missing = origin == missing_session or target == missing_session
                color = "#cbd5e1" if involves_missing else ("#0891b2" if horizon == 1 else "#7c3aed")
                alpha = 0.42 if involves_missing else 0.8
                arc_height = min(0.18 + 0.055 * horizon, 0.58)
                ax.annotate(
                    "",
                    xy=(target, 0.02),
                    xytext=(origin, 0.02),
                    arrowprops={
                        "arrowstyle": "->",
                        "connectionstyle": f"arc3,rad={-arc_height}",
                        "color": color,
                        "lw": 1.5,
                        "alpha": alpha,
                    },
                )
            for day in sessions:
                ax.text(day, -0.26, f"S{day}", ha="center", va="top", fontsize=7, color="#475569")
            ax.grid(axis="x", alpha=0.12)
        axes[-1].set_xlabel("Ordered exchange sessions (schematic)")
        fig.suptitle("Calendar-first pairing: missing sessions do not create longer targets", color="#0f172a", weight="bold")
        fig.text(
            0.5,
            0.01,
            "Red × = unavailable distribution; grey links are removed, not bridged",
            ha="center",
            fontsize=8,
            color="#64748b",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        return [artist for ax in axes for artist in ax.get_children()]

    animation = FuncAnimation(fig, update, frames=8, interval=350, blit=False)
    animation.save(
        FIGURE_DIR / "calendar_first_pairs.gif",
        writer=PillowWriter(fps=4),
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def _animated_forward_chaining() -> None:
    folds = [
        (4, 4, 6),
        (6, 6, 8),
        (8, 8, 10),
        (10, 10, 12),
    ]
    n_sessions = 12
    fig, ax = plt.subplots(figsize=(10, 2.7))
    palette = {"train": "#2563eb", "validation": "#f59e0b", "test": "#e2e8f0"}

    def update(frame: int) -> list:
        train_end, validation_start, validation_end = folds[frame]
        ax.clear()
        ax.set_xlim(0, n_sessions)
        ax.set_ylim(-0.6, 1.5)
        ax.set_yticks([])
        ax.set_xticks(np.arange(n_sessions) + 0.5, [f"t{i+1}" for i in range(n_sessions)])
        ax.set_xlabel("Chronological sessions →")
        for index in range(n_sessions):
            if index < train_end:
                color, label = palette["train"], "train"
            elif validation_start <= index < validation_end:
                color, label = palette["validation"], "validation"
            else:
                color, label = palette["test"], "not yet used"
            ax.add_patch(
                FancyBboxPatch(
                    (index + 0.06, 0.2),
                    0.88,
                    0.48,
                    boxstyle="round,pad=0.02,rounding_size=0.04",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
        ax.text(0.1, 1.04, f"Forward-chaining fold {frame + 1} of {len(folds)}", fontsize=12, weight="bold", color="#0f172a")
        ax.text(0.1, -0.25, "Select hyperparameters using validation loss; test window remains untouched", fontsize=9, color="#475569")
        ax.scatter([], [], color=palette["train"], label="training window")
        ax.scatter([], [], color=palette["validation"], label="next validation block")
        ax.scatter([], [], color=palette["test"], label="future / held out")
        ax.legend(ncol=3, loc="upper right", frameon=False, fontsize=8)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.12)
        return list(ax.patches) + list(ax.texts) + list(ax.collections)

    animation = FuncAnimation(fig, update, frames=len(folds), interval=750, blit=False)
    animation.save(
        FIGURE_DIR / "forward_chaining_cv.gif",
        writer=PillowWriter(fps=1.5),
        savefig_kwargs={"facecolor": "white"},
    )
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
    _animated_pipeline()
    _animated_quantile_change()
    _animated_w2_transport()
    _animated_kernel_similarity()
    _animated_calendar_pairs()
    _animated_forward_chaining()
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
