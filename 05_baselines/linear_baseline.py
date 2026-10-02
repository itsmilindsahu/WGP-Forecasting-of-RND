from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RND_FILE = (
    ROOT
    / "02_rnd_extraction"
    / "data"
    / "rnd_30d.csv"
)

OUT_DIR = ROOT / "05_baselines" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUT_DIR
    / "linear_spx_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FRACTION = 0.80


# ============================================================
# LOAD RND
# ============================================================

print("=" * 70)
print("LINEAR MOMENT BASELINE — SPX 30D RND")
print("=" * 70)

rnd = pd.read_csv(
    RND_FILE,
    parse_dates=["date"]
)

rnd = rnd.sort_values(
    ["date", "strike"]
).reset_index(drop=True)

dates = sorted(
    rnd["date"].unique()
)

print(f"RND days : {len(dates)}")


# ============================================================
# CONVERT EACH RND TO FORWARD-NORMALIZED MOMENTS
# ============================================================

records = []

for dt, g in rnd.groupby("date"):

    g = g.sort_values("strike").copy()

    x = g["strike"].to_numpy(dtype=float)
    density = g["density"].to_numpy(dtype=float)

    forward = float(g["forward"].iloc[0])

    # Forward-normalized price coordinate
    z = x / forward

    # Numerical normalization
    mass = np.trapezoid(
        density,
        z
    )

    if mass <= 0:
        continue

    density = density / mass

    # Moments
    mean = np.trapezoid(
        z * density,
        z
    )

    variance = np.trapezoid(
        (z - mean) ** 2 * density,
        z
    )

    std = np.sqrt(
        max(variance, 1e-16)
    )

    skew = np.trapezoid(
        ((z - mean) / std) ** 3 * density,
        z
    )

    records.append(
        {
            "date": pd.Timestamp(dt),
            "mean": mean,
            "variance": variance,
            "skew": skew,
        }
    )


moments = pd.DataFrame(records)

moments = moments.sort_values(
    "date"
).reset_index(drop=True)


print(
    f"Moment rows : {len(moments)}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

n = len(moments)

n_train = int(
    np.floor(
        TRAIN_FRACTION * n
    )
)

train = moments.iloc[
    :n_train
].copy()

test = moments.iloc[
    n_train:
].copy()


print(
    f"Training days : {len(train)}"
)

print(
    f"Test days     : {len(test)}"
)


# ============================================================
# LINEAR MODEL
#
# Predict tomorrow's variance from today's:
# mean, variance, skewness.
# ============================================================

X_train = train[
    ["mean", "variance", "skew"]
].to_numpy()

y_train = train[
    "variance"
].shift(-1)


# Last training row has no next-day target.
X_train = X_train[:-1]
y_train = y_train.iloc[:-1].to_numpy()


# Add intercept
X_design = np.column_stack(
    [
        np.ones(len(X_train)),
        X_train,
    ]
)


beta = np.linalg.lstsq(
    X_design,
    y_train,
    rcond=None
)[0]


# ============================================================
# W2 HELPER
# ============================================================

def quantile_from_rnd(
    g: pd.DataFrame,
    n_points: int = 200,
) -> np.ndarray:

    g = g.sort_values(
        "strike"
    ).copy()

    x = g["strike"].to_numpy(
        dtype=float
    )

    density = g["density"].to_numpy(
        dtype=float
    )

    forward = float(
        g["forward"].iloc[0]
    )

    z = x / forward

    mass = np.trapezoid(
        density,
        z
    )

    density = density / mass

    cdf = np.zeros_like(
        density
    )

    for i in range(
        1,
        len(z)
    ):

        cdf[i] = (
            cdf[i - 1]
            + 0.5
            * (
                density[i]
                + density[i - 1]
            )
            * (
                z[i]
                - z[i - 1]
            )
        )

    cdf = np.maximum.accumulate(
        cdf
    )

    cdf[-1] = 1.0

    q = np.linspace(
        0.0025,
        0.9975,
        n_points
    )

    return np.interp(
        q,
        cdf,
        z
    )


def w2(
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (q1 - q2) ** 2
            )
        )
    )


# ============================================================
# GAUSSIAN QUANTILE FUNCTION
# ============================================================

from statistics import NormalDist

normal = NormalDist()

q_levels = np.linspace(
    0.0025,
    0.9975,
    200
)

normal_z = np.array(
    [
        normal.inv_cdf(float(p))
        for p in q_levels
    ]
)


# ============================================================
# FORECAST
# ============================================================

results = []


for i in range(
    n_train,
    n - 1
):

    today = moments.iloc[i]
    tomorrow = moments.iloc[i + 1]

    # --------------------------------------------------------
    # Predict tomorrow's variance
    # --------------------------------------------------------

    x_today = np.array(
        [
            1.0,
            today["mean"],
            today["variance"],
            today["skew"],
        ]
    )

    predicted_variance = float(
        x_today @ beta
    )

    predicted_variance = max(
        predicted_variance,
        1e-8
    )

    # --------------------------------------------------------
    # Gaussian forecast
    # --------------------------------------------------------

    predicted_mean = float(
        today["mean"]
    )

    predicted_std = np.sqrt(
        predicted_variance
    )

    predicted_q = (
        predicted_mean
        + predicted_std * normal_z
    )

    # --------------------------------------------------------
    # Actual next-day RND
    # --------------------------------------------------------

    actual_group = rnd[
        rnd["date"]
        == tomorrow["date"]
    ]

    actual_q = quantile_from_rnd(
        actual_group
    )

    error = w2(
        predicted_q,
        actual_q
    )

    results.append(
        {
            "date": tomorrow["date"],
            "linear_w2": error,
            "predicted_variance":
                predicted_variance,
            "actual_variance":
                tomorrow["variance"],
        }
    )


results = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("LINEAR BASELINE RESULTS")
print("=" * 70)

print(
    f"Test predictions : {len(results)}"
)

if len(results) > 0:

    print(
        f"Mean W2          : "
        f"{results['linear_w2'].mean():.6f}"
    )

    print(
        f"Median W2        : "
        f"{results['linear_w2'].median():.6f}"
    )

    print()
    print(results.to_string(index=False))


print()
print("Saved:")
print(OUTPUT_FILE)

print("=" * 70)