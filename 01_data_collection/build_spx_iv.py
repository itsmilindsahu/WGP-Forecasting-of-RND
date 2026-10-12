from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "01_data_collection" / "data" / "option_chains_spx_20d.csv"
OUTPUT = ROOT / "01_data_collection" / "data" / "spx_iv_20d.csv"


def black76_price(F, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if option_type == "CALL" else max(K - F, 0.0)
        return np.exp(-r * T) * intrinsic

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * T)

    if option_type == "CALL":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_volatility(price, F, K, T, r, option_type):
    if not np.isfinite(price) or price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    discount = np.exp(-r * T)
    if option_type == "CALL":
        intrinsic, upper = discount * max(F - K, 0.0), discount * F
    else:
        intrinsic, upper = discount * max(K - F, 0.0), discount * K

    if price <= intrinsic or price >= upper:
        return np.nan

    def objective(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0)
    except ValueError:
        return np.nan


def estimate_forward(expiry_df: pd.DataFrame, rate: float, T: float) -> float:
    calls = expiry_df.loc[
        expiry_df["option_type"].str.upper() == "CALL", ["strike", "mid"]
    ].rename(columns={"mid": "call_mid"})
    puts = expiry_df.loc[
        expiry_df["option_type"].str.upper() == "PUT", ["strike", "mid"]
    ].rename(columns={"mid": "put_mid"})
    common = calls.merge(puts, on="strike", how="inner")
    if common.empty:
        return np.nan

    estimates = common["strike"] + np.exp(rate * T) * (common["call_mid"] - common["put_mid"])
    estimates = estimates[np.isfinite(estimates) & (estimates > 0)]
    if estimates.empty:
        return np.nan

    preliminary = float(estimates.median())
    near_atm = np.log(common["strike"] / preliminary).abs() <= 0.05
    near_atm_estimates = (common.loc[near_atm, "strike"] + np.exp(rate * T) * (
        common.loc[near_atm, "call_mid"] - common.loc[near_atm, "put_mid"]
    ))
    return float(near_atm_estimates.median()) if len(near_atm_estimates) >= 3 else preliminary


def enrich_option_chain(
    chain: pd.DataFrame,
    expiry_inputs: pd.DataFrame,
    rate_provider: Callable[[pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    """Add expiry-level forward/rate and bid/mid/ask Black-76 IVs."""
    source = chain.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["expiry_date"] = pd.to_datetime(source["expiry_date"])
    if "option_type" not in source:
        source["option_type"] = source["right"].astype(str).str.upper()

    lookup: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[float, float]] = {}
    if not expiry_inputs.empty:
        inputs = expiry_inputs.copy()
        inputs["date"] = pd.to_datetime(inputs["date"])
        inputs["expiry_date"] = pd.to_datetime(inputs["expiry_date"])
        for (date, expiry), group in inputs.groupby(["date", "expiry_date"]):
            lookup[(date, expiry)] = (float(group["forward"].median()), float(group["rate"].median()))

    results = []
    rate_cache: dict[pd.Timestamp, float] = {}
    for (date, expiry), group in source.groupby(["date", "expiry_date"], sort=True):
        group = group.copy()
        T = (expiry - date).days / 365.0
        if T <= 0:
            continue

        saved = lookup.get((date, expiry))
        if saved is not None:
            forward, rate = saved
        else:
            if rate_provider is None:
                raise ValueError(f"No saved forward/rate or rate provider for {date.date()} {expiry.date()}")
            if date not in rate_cache:
                rate_cache[date] = float(rate_provider(date))
            rate = rate_cache[date]
            forward = estimate_forward(group, rate, T)

        if not np.isfinite(forward) or forward <= 0 or not np.isfinite(rate):
            continue

        group["forward"] = forward
        group["rate"] = rate
        group["tau_years"] = T
        group["log_forward_moneyness"] = np.log(group["strike"] / forward)

        for side, quote_column in (("bid", "bid"), ("mid", "mid"), ("ask", "ask")):
            group[f"iv_{side}"] = [
                implied_volatility(float(row[quote_column]), forward, float(row["strike"]), T, rate,
                                   str(row["option_type"]).upper())
                for _, row in group.iterrows()
            ]

        group["total_variance"] = group["iv_mid"] ** 2 * T
        results.append(group)

    if not results:
        raise ValueError("No expiry groups had usable forward and rate inputs")
    return pd.concat(results, ignore_index=True)


def theta_rate_provider() -> Callable[[pd.Timestamp], float]:
    from dotenv import load_dotenv

    load_dotenv()
    from vol_forge.theta_data.theta_data_client import ThetaDataClientFactory

    client = ThetaDataClientFactory.create_instance()

    def get_rate(date: pd.Timestamp) -> float:
        response = client.interest_rate_history_eod("SOFR", date.date(), date.date())
        return float(response["rate"][0]) / 100.0

    return get_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SPX quote/forward/IV adapter table")
    parser.add_argument("--rates-csv", type=Path, help="Existing table with date, expiry_date, forward and rate")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    chain = pd.read_csv(INPUT)
    expiry_inputs = pd.read_csv(args.rates_csv) if args.rates_csv else pd.DataFrame()
    rate_provider = None if args.rates_csv else theta_rate_provider()
    enriched = enrich_option_chain(chain, expiry_inputs, rate_provider)
    enriched.to_csv(args.output, index=False)
    print(f"Rows written: {len(enriched):,}; dates: {enriched['date'].nunique()}")
    print(f"Valid bid/mid/ask IV rows: {enriched[['iv_bid', 'iv_mid', 'iv_ask']].notna().sum().to_dict()}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()