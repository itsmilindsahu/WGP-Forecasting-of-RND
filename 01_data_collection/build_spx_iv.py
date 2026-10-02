from __future__ import annotations

import os
from pathlib import Path
from datetime import date

from dotenv import load_dotenv

# Load VolForge API key BEFORE importing VolForge
load_dotenv(
    r"C:\Users\MY PC\Downloads\VolForge\.env"
)

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from vol_forge.theta_data.theta_data_client import ThetaDataClientFactory


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "01_data_collection"
    / "data"
    / "option_chains_spx_20d.csv"
)

OUTPUT = (
    ROOT
    / "01_data_collection"
    / "data"
    / "spx_iv_20d.csv"
)

# Use the existing VolForge API configuration
load_dotenv(
    r"C:\Users\MY PC\Downloads\VolForge\.env"
)


# ============================================================
# BLACK-76
# ============================================================

def black76_price(F, K, T, r, sigma, option_type):

    if T <= 0 or sigma <= 0:
        intrinsic = (
            max(F - K, 0.0)
            if option_type == "CALL"
            else max(K - F, 0.0)
        )

        return np.exp(-r * T) * intrinsic

    sqrt_T = np.sqrt(T)

    d1 = (
        np.log(F / K)
        + 0.5 * sigma**2 * T
    ) / (sigma * sqrt_T)

    d2 = d1 - sigma * sqrt_T

    discount = np.exp(-r * T)

    if option_type == "CALL":

        return discount * (
            F * norm.cdf(d1)
            - K * norm.cdf(d2)
        )

    return discount * (
        K * norm.cdf(-d2)
        - F * norm.cdf(-d1)
    )


def implied_volatility(
    price,
    F,
    K,
    T,
    r,
    option_type,
):

    if not np.isfinite(price):
        return np.nan

    if price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    discount = np.exp(-r * T)

    if option_type == "CALL":

        intrinsic = discount * max(F - K, 0.0)
        upper = discount * F

    else:

        intrinsic = discount * max(K - F, 0.0)
        upper = discount * K

    if price <= intrinsic or price >= upper:
        return np.nan

    def objective(sigma):

        return (
            black76_price(
                F,
                K,
                T,
                r,
                sigma,
                option_type,
            )
            - price
        )

    try:

        return brentq(
            objective,
            1e-6,
            5.0,
        )

    except ValueError:

        return np.nan


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SPX FORWARD + IMPLIED VOLATILITY — 20 DAY PILOT")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"Rows loaded: {len(df):,}")
print(
    f"Dates: {df['date'].nunique()}"
)
print()


# ============================================================
# THETADATA CLIENT
# ============================================================

client = ThetaDataClientFactory.create_instance()


# ============================================================
# OUTPUT ROWS
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH DATE
# ============================================================

for value_date, day_df in df.groupby("date"):

    print(f"Processing {value_date} ...")

    value_date_obj = pd.Timestamp(
        value_date
    ).date()

    # --------------------------------------------------------
    # SOFR
    # --------------------------------------------------------

    try:

        sofr = client.interest_rate_history_eod(
            "SOFR",
            value_date_obj,
            value_date_obj,
        )

        rate = float(
            sofr["rate"][0]
        ) / 100.0

    except Exception as exc:

        print(
            f"  SOFR unavailable: {exc}"
        )

        continue

    day_results = []

    # --------------------------------------------------------
    # Each expiry separately
    # --------------------------------------------------------

    for expiry, expiry_df in day_df.groupby(
        "expiry_date"
    ):

        expiry_df = expiry_df.copy()

        expiry_date = pd.Timestamp(
            expiry
        ).date()

        T = (
            expiry_date
            - value_date_obj
        ).days / 365.0

        if T <= 0:
            continue

        # ----------------------------------------------------
        # Calls / puts
        # ----------------------------------------------------

        calls = expiry_df[
            expiry_df["option_type"] == "CALL"
        ][
            ["strike", "mid"]
        ].rename(
            columns={"mid": "call_mid"}
        )

        puts = expiry_df[
            expiry_df["option_type"] == "PUT"
        ][
            ["strike", "mid"]
        ].rename(
            columns={"mid": "put_mid"}
        )

        common = calls.merge(
            puts,
            on="strike",
            how="inner",
        )

        if len(common) == 0:
            continue

        # ----------------------------------------------------
        # Put-call parity
        #
        # F = K + exp(rT)(C-P)
        # ----------------------------------------------------

        common["forward_estimate"] = (
            common["strike"]
            + np.exp(rate * T)
            * (
                common["call_mid"]
                - common["put_mid"]
            )
        )

        common = common[
            np.isfinite(
                common["forward_estimate"]
            )
            & (
                common["forward_estimate"]
                > 0
            )
        ]

        if len(common) == 0:
            continue

        preliminary_F = (
            common["forward_estimate"]
            .median()
        )

        # Near-ATM parity estimates are more stable
        common["k_temp"] = np.log(
            common["strike"]
            / preliminary_F
        )

        near_atm = common[
            common["k_temp"].abs() <= 0.05
        ]

        if len(near_atm) >= 3:

            F = near_atm[
                "forward_estimate"
            ].median()

        else:

            F = preliminary_F

        # ----------------------------------------------------
        # Add expiry-level information
        # ----------------------------------------------------

        expiry_df["forward"] = F
        expiry_df["rate"] = rate
        expiry_df["tau_years"] = T

        # ----------------------------------------------------
        # Log-forward moneyness
        # ----------------------------------------------------

        expiry_df[
            "log_forward_moneyness"
        ] = np.log(
            expiry_df["strike"] / F
        )

        # ----------------------------------------------------
        # IV
        # ----------------------------------------------------

        expiry_df["iv_mid"] = np.nan

        for idx, row in expiry_df.iterrows():

            iv = implied_volatility(
                float(row["mid"]),
                F,
                float(row["strike"]),
                T,
                rate,
                row["option_type"],
            )

            expiry_df.loc[
                idx,
                "iv_mid"
            ] = iv

        # ----------------------------------------------------
        # Total variance
        # ----------------------------------------------------

        expiry_df[
            "total_variance"
        ] = (
            expiry_df["iv_mid"] ** 2
            * T
        )

        day_results.append(
            expiry_df
        )

    if day_results:

        result = pd.concat(
            day_results,
            ignore_index=True,
        )

        all_results.append(result)

        print(
            f"  {len(result):,} contracts"
        )


# ============================================================
# COMBINE
# ============================================================

if not all_results:

    raise RuntimeError(
        "No IV data was produced."
    )

out = pd.concat(
    all_results,
    ignore_index=True,
)


# ============================================================
# CLEAN
# ============================================================

out = out[
    np.isfinite(
        out["iv_mid"]
    )
    & np.isfinite(
        out["total_variance"]
    )
].copy()


# ============================================================
# SAVE
# ============================================================

out.to_csv(
    OUTPUT,
    index=False,
)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Dates processed : "
    f"{out['date'].nunique()}"
)

print(
    f"Rows            : "
    f"{len(out):,}"
)

print(
    f"Valid IV        : "
    f"{out['iv_mid'].notna().sum():,}"
)

print()
print(
    "Saved:"
)
print(OUTPUT)

print("=" * 70)
print("DONE")
print("=" * 70)