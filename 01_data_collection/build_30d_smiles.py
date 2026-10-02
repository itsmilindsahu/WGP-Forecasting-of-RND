from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "01_data_collection"
    / "data"
    / "spx_iv_20d.csv"
)

OUTPUT = (
    ROOT
    / "01_data_collection"
    / "data"
    / "spx_30d_smiles.csv"
)

TARGET_DTE = 30

# Common log-forward-moneyness grid
K_GRID = np.linspace(-0.20, 0.20, 81)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SPX 30-DAY CONSTANT-MATURITY SMILES")
print("=" * 70)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])
df["expiry_date"] = pd.to_datetime(df["expiry_date"])

print(
    f"Rows loaded : {len(df):,}"
)

print(
    f"Dates       : {df['date'].nunique()}"
)

print()


# ============================================================
# PROCESS EACH DATE
# ============================================================

all_smiles = []

for value_date, day in df.groupby("date"):

    print(
        f"Processing "
        f"{value_date.date()} ..."
    )

    # --------------------------------------------------------
    # Available expiries
    # --------------------------------------------------------

    expiry_info = (
        day[
            [
                "expiry_date",
                "dte",
                "tau_years",
            ]
        ]
        .drop_duplicates()
        .sort_values("dte")
    )

    lower = expiry_info[
        expiry_info["dte"] <= TARGET_DTE
    ]

    upper = expiry_info[
        expiry_info["dte"] >= TARGET_DTE
    ]

    if lower.empty or upper.empty:

        print(
            "  SKIP: no expiries bracketing 30D"
        )

        continue

    lower = lower.iloc[-1]
    upper = upper.iloc[0]

    lower_expiry = lower["expiry_date"]
    upper_expiry = upper["expiry_date"]

    lower_dte = float(
        lower["dte"]
    )

    upper_dte = float(
        upper["dte"]
    )

    lower_tau = float(
        lower["tau_years"]
    )

    upper_tau = float(
        upper["tau_years"]
    )

    target_tau = TARGET_DTE / 365.0

    # --------------------------------------------------------
    # If exactly 30D exists, use it directly.
    # --------------------------------------------------------

    if lower_dte == TARGET_DTE:

        upper_dte = TARGET_DTE

        upper_tau = lower_tau

        upper_expiry = lower_expiry

        alpha = 0.0

    else:

        alpha = (
            target_tau - lower_tau
        ) / (
            upper_tau - lower_tau
        )

    # --------------------------------------------------------
    # Extract the two expiry smiles
    # --------------------------------------------------------

    lower_df = day[
        day["expiry_date"]
        == lower_expiry
    ].copy()

    upper_df = day[
        day["expiry_date"]
        == upper_expiry
    ].copy()

    # --------------------------------------------------------
    # Remove duplicate moneyness points
    # --------------------------------------------------------

    lower_df = (
        lower_df
        .dropna(
            subset=[
                "log_forward_moneyness",
                "total_variance",
            ]
        )
        .sort_values(
            "log_forward_moneyness"
        )
        .drop_duplicates(
            "log_forward_moneyness"
        )
    )

    upper_df = (
        upper_df
        .dropna(
            subset=[
                "log_forward_moneyness",
                "total_variance",
            ]
        )
        .sort_values(
            "log_forward_moneyness"
        )
        .drop_duplicates(
            "log_forward_moneyness"
        )
    )

    if len(lower_df) < 5 or len(upper_df) < 5:

        print(
            "  SKIP: insufficient smile points"
        )

        continue

    # --------------------------------------------------------
    # Interpolation helper
    # --------------------------------------------------------

    def interpolate(
        data,
        grid,
    ):

        x = data[
            "log_forward_moneyness"
        ].to_numpy(
            dtype=float
        )

        y = data[
            "total_variance"
        ].to_numpy(
            dtype=float
        )

        # Only interpolate inside actual coverage
        result = np.full(
            len(grid),
            np.nan,
        )

        mask = (
            (grid >= x.min())
            & (grid <= x.max())
        )

        result[mask] = np.interp(
            grid[mask],
            x,
            y,
        )

        return result

    w_lower = interpolate(
        lower_df,
        K_GRID,
    )

    w_upper = interpolate(
        upper_df,
        K_GRID,
    )

    valid = (
        np.isfinite(w_lower)
        & np.isfinite(w_upper)
    )

    if valid.sum() < 20:

        print(
            "  SKIP: insufficient common "
            "moneyness coverage"
        )

        continue

    k = K_GRID[valid]

    w1 = w_lower[valid]

    w2 = w_upper[valid]

    # --------------------------------------------------------
    # INTERPOLATE TOTAL VARIANCE
    # --------------------------------------------------------

    if lower_dte == TARGET_DTE:

        w30 = w1

    else:

        w30 = (
            w1
            + alpha * (w2 - w1)
        )

    # Numerical safety
    w30 = np.maximum(
        w30,
        0.0,
    )

    iv30 = np.sqrt(
        w30 / target_tau
    )

    # --------------------------------------------------------
    # Forward at 30D
    #
    # Interpolate the two expiry forwards
    # for bookkeeping.
    # --------------------------------------------------------

    F1 = float(
        lower_df["forward"].iloc[0]
    )

    F2 = float(
        upper_df["forward"].iloc[0]
    )

    if lower_dte == TARGET_DTE:

        F30 = F1

    else:

        F30 = (
            F1
            + alpha * (F2 - F1)
        )

    rate = float(
        lower_df["rate"].iloc[0]
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    smile = pd.DataFrame(
        {
            "date": value_date,

            "target_dte": TARGET_DTE,

            "k": k,

            "total_variance": w30,

            "iv_30d": iv30,

            "forward_30d": F30,

            "rate": rate,

            "tau_years": target_tau,

            "lower_expiry":
                lower_expiry,

            "upper_expiry":
                upper_expiry,

            "lower_dte":
                lower_dte,

            "upper_dte":
                upper_dte,

            "interpolation_weight":
                alpha,
        }
    )

    all_smiles.append(
        smile
    )

    print(
        f"  {lower_dte:.0f}D -> "
        f"{upper_dte:.0f}D | "
        f"{len(smile)} smile points | "
        f"alpha={alpha:.3f}"
    )


# ============================================================
# COMBINE
# ============================================================

if not all_smiles:

    raise RuntimeError(
        "No 30-day smiles were created."
    )

result = pd.concat(
    all_smiles,
    ignore_index=True,
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"30D smiles created : "
    f"{result['date'].nunique()}"
)

print(
    f"Total smile points : "
    f"{len(result):,}"
)

print()

print(
    result.groupby("date")
    .size()
)

print()

print(
    f"Saved to:"
)

print(OUTPUT)

print("=" * 70)
print("DONE")
print("=" * 70)