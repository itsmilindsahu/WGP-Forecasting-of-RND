from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GP_FILE = (
    ROOT
    / "04_gp_model"
    / "results"
    / "wasserstein_forecast_comparison.csv"
)

LINEAR_FILE = (
    ROOT
    / "05_baselines"
    / "results"
    / "linear_spx_predictions.csv"
)

OUT_DIR = ROOT / "06_evaluation" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SPX 30D MODEL COMPARISON")
print("=" * 70)

gp = pd.read_csv(
    GP_FILE,
    parse_dates=["date"]
)

linear = pd.read_csv(
    LINEAR_FILE,
    parse_dates=["date"]
)


# ============================================================
# SHOW COLUMNS
# ============================================================

print()
print("GP columns:")
print(list(gp.columns))

print()
print("Linear columns:")
print(list(linear.columns))


# ============================================================
# NORMALIZE GP COLUMN NAMES
# ============================================================

# The GP file contains the Wasserstein GP and persistence
# errors for the same held-out dates.

gp = gp.rename(
    columns={
        "wasserstein_gp_w2": "wasserstein_gp_w2",
        "persistence_w2": "persistence_w2",
    }
)


# ============================================================
# MERGE ON EXACT SAME DATES
# ============================================================

comparison = pd.merge(
    gp[
        [
            "date",
            "wasserstein_gp_w2",
            "persistence_w2",
        ]
    ],
    linear[
        [
            "date",
            "linear_w2",
        ]
    ],
    on="date",
    how="inner",
)


comparison = comparison.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# CHECK
# ============================================================

print()
print("=" * 70)
print("COMMON HELD-OUT DATES")
print("=" * 70)

print(
    comparison[
        [
            "date",
            "persistence_w2",
            "linear_w2",
            "wasserstein_gp_w2",
        ]
    ].to_string(index=False)
)


print()
print(
    f"Common test dates : {len(comparison)}"
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "model": [
            "Persistence",
            "Linear moment baseline",
            "Wasserstein GP",
        ],
        "mean_w2": [
            comparison["persistence_w2"].mean(),
            comparison["linear_w2"].mean(),
            comparison["wasserstein_gp_w2"].mean(),
        ],
        "median_w2": [
            comparison["persistence_w2"].median(),
            comparison["linear_w2"].median(),
            comparison["wasserstein_gp_w2"].median(),
        ],
        "std_w2": [
            comparison["persistence_w2"].std(),
            comparison["linear_w2"].std(),
            comparison["wasserstein_gp_w2"].std(),
        ],
    }
)


# ============================================================
# DIFFERENCE RELATIVE TO PERSISTENCE
# ============================================================

persistence_mean = (
    comparison["persistence_w2"].mean()
)

summary["difference_vs_persistence"] = (
    summary["mean_w2"]
    - persistence_mean
)

summary["relative_change_vs_persistence_pct"] = (
    summary["difference_vs_persistence"]
    / persistence_mean
    * 100.0
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_file = (
    OUT_DIR
    / "model_comparison.csv"
)

summary_file = (
    OUT_DIR
    / "model_comparison_summary.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    comparison["date"],
    comparison["persistence_w2"],
    marker="o",
    label="Persistence"
)

plt.plot(
    comparison["date"],
    comparison["linear_w2"],
    marker="o",
    label="Linear moment baseline"
)

plt.plot(
    comparison["date"],
    comparison["wasserstein_gp_w2"],
    marker="o",
    label="Wasserstein GP"
)

plt.xlabel(
    "Forecast date"
)

plt.ylabel(
    "Wasserstein-2 error"
)

plt.title(
    "SPX 30D RND Forecast Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plot_file = (
    OUT_DIR
    / "model_comparison.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(comparison_file)
print(summary_file)
print(plot_file)

print("=" * 70)