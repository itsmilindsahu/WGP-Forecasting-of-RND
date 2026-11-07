from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "01_data_collection" / "data" / "option_chains_spx.csv"
OUTPUT = ROOT / "01_data_collection" / "data" / "spx_iv.csv"
RATE_CACHE = ROOT / "01_data_collection" / "data" / "sofr_rates.csv"
DIAGNOSTICS = ROOT / "01_data_collection" / "data" / "spx_iv_diagnostics.csv"
RATE_ERRORS: dict[pd.Timestamp, str] = {}


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


def implied_volatilities(prices, F, strikes, T, r, option_types):
    prices = np.asarray(prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    option_types = np.char.upper(np.asarray(option_types, dtype=str))
    calls = option_types == "CALL"
    valid = (
        np.isfinite(prices)
        & np.isfinite(strikes)
        & (prices > 0)
        & (strikes > 0)
        & (F > 0)
        & (T > 0)
        & np.isin(option_types, ("CALL", "PUT"))
    )
    result = np.full(prices.shape, np.nan, dtype=float)
    if not valid.any():
        return result

    valid_strikes = strikes[valid]
    valid_prices = prices[valid]
    valid_calls = calls[valid]
    discount = np.exp(-r * T)
    intrinsic = np.where(
        valid_calls,
        discount * np.maximum(F - valid_strikes, 0.0),
        discount * np.maximum(valid_strikes - F, 0.0),
    )
    upper = np.where(valid_calls, discount * F, discount * valid_strikes)
    solvable = (valid_prices > intrinsic) & (valid_prices < upper)
    if not solvable.any():
        return result

    strikes_to_solve = valid_strikes[solvable]
    prices_to_solve = valid_prices[solvable]
    calls_to_solve = valid_calls[solvable]

    def black76_prices(volatilities):
        sqrt_t = np.sqrt(T)
        sigma_sqrt_t = volatilities * sqrt_t
        d1 = (
            np.log(F / strikes_to_solve) + 0.5 * volatilities**2 * T
        ) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        call_prices = discount * (
            F * ndtr(d1) - strikes_to_solve * ndtr(d2)
        )
        put_prices = discount * (
            strikes_to_solve * ndtr(-d2) - F * ndtr(-d1)
        )
        return np.where(calls_to_solve, call_prices, put_prices)

    low = np.full(len(prices_to_solve), 1e-6)
    high = np.full(len(prices_to_solve), 5.0)
    low_prices = black76_prices(low)
    high_prices = black76_prices(high)
    bracketed = (prices_to_solve >= low_prices) & (prices_to_solve <= high_prices)
    if not bracketed.any():
        return result

    strikes_to_solve = strikes_to_solve[bracketed]
    calls_to_solve = calls_to_solve[bracketed]
    low = low[bracketed]
    high = high[bracketed]
    target_prices = prices_to_solve[bracketed]
    exact_low = target_prices == low_prices[bracketed]
    exact_high = target_prices == high_prices[bracketed]
    for _ in range(52):
        middle = (low + high) / 2.0
        middle_prices = black76_prices(middle)
        below_target = middle_prices < target_prices
        low = np.where(below_target, middle, low)
        high = np.where(below_target, high, middle)

    solved = (low + high) / 2.0
    solved[exact_low] = 1e-6
    solved[exact_high] = 5.0
    valid_indices = np.flatnonzero(valid)
    solve_indices = valid_indices[np.flatnonzero(solvable)]
    result[solve_indices[bracketed]] = solved
    return result


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
    if "root" in source:
        source["root"] = source["root"].astype(str).str.upper()

    lookup: dict[
        tuple[pd.Timestamp, pd.Timestamp, str | None],
        tuple[float, float],
    ] = {}
    if not expiry_inputs.empty:
        inputs = expiry_inputs.copy()
        inputs["date"] = pd.to_datetime(inputs["date"])
        inputs["expiry_date"] = pd.to_datetime(inputs["expiry_date"])
        input_group_columns = ["date", "expiry_date"]
        if "root" in inputs:
            inputs["root"] = inputs["root"].astype(str).str.upper()
            input_group_columns.append("root")

        for key, group in inputs.groupby(input_group_columns):
            date, expiry = key[:2]
            root = str(key[2]).upper() if len(key) == 3 else None
            lookup[(date, expiry, root)] = (
                float(group["forward"].median()),
                float(group["rate"].median()),
            )

    results = []
    rate_cache: dict[pd.Timestamp, float] = {}
    source_group_columns = ["date", "expiry_date"]
    if "root" in source:
        source_group_columns.append("root")

    for key, group in source.groupby(source_group_columns, sort=True):
        date, expiry = key[:2]
        root = str(key[2]).upper() if len(key) == 3 else None
        group = group.copy()
        T = (expiry - date).days / 365.0
        if T <= 0:
            continue

        saved = lookup.get((date, expiry, root))
        if saved is None:
            saved = lookup.get((date, expiry, None))
        if saved is not None:
            forward, rate = saved
        else:
            if rate_provider is None:
                raise ValueError(f"No saved forward/rate or rate provider for {date.date()} {expiry.date()}")
            if date not in rate_cache:
                try:
                    rate_cache[date] = float(rate_provider(date))
                except Exception as exc:
                    RATE_ERRORS[date] = f"{type(exc).__name__}: {exc}"
                    rate_cache[date] = np.nan
            rate = rate_cache[date]
            forward = estimate_forward(group, rate, T)

        if not np.isfinite(forward) or forward <= 0 or not np.isfinite(rate):
            continue

        group["forward"] = forward
        group["rate"] = rate
        group["tau_years"] = T
        group["log_forward_moneyness"] = np.log(group["strike"] / forward)

        strikes = group["strike"].to_numpy(dtype=float)
        option_types = group["option_type"].to_numpy(dtype=str)
        for side, quote_column in (("bid", "bid"), ("mid", "mid"), ("ask", "ask")):
            group[f"iv_{side}"] = implied_volatilities(
                group[quote_column].to_numpy(dtype=float),
                forward,
                strikes,
                T,
                rate,
                option_types,
            )

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
    if RATE_CACHE.exists():
        cached_rates = pd.read_csv(RATE_CACHE, parse_dates=["date"])
        rate_cache = {
            pd.Timestamp(row.date).normalize(): float(row.rate)
            for row in cached_rates.itertuples(index=False)
        }
    else:
        rate_cache = {}
    unavailable_before: date | None = None
    unavailable_reason = ""

    time.sleep(1.0)
    try:
        response = client.interest_rate_history_eod(
            "SOFR",
            date(2023, 11, 6),
            date(2026, 11, 4),
        )
    except Exception as exc:
        message = str(exc)
        if "before 2024-01-01 with FREE subscription" not in message:
            raise
        unavailable_before = date(2024, 1, 1)
        unavailable_reason = message
        time.sleep(1.0)
        response = client.interest_rate_history_eod(
            "SOFR",
            unavailable_before,
            date(2026, 11, 4),
        )

    rates = response.to_pandas() if hasattr(response, "to_pandas") else pd.DataFrame(response)
    if "created" not in rates.columns or "rate" not in rates.columns:
        raise ValueError("ThetaData SOFR response is missing created or rate columns")
    rates["date"] = (
        pd.to_datetime(rates["created"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    rates["rate"] = pd.to_numeric(rates["rate"], errors="coerce") / 100.0
    rates = rates.dropna(subset=["date", "rate"])
    rate_cache.update(
        rates.groupby("date")["rate"].median().astype(float).to_dict()
    )
    RATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    rate_frame = pd.DataFrame(
        [{"date": rate_date, "rate": rate} for rate_date, rate in rate_cache.items()]
    ).sort_values("date")
    temporary_path = RATE_CACHE.with_name(f"{RATE_CACHE.name}.tmp")
    rate_frame.to_csv(temporary_path, index=False)
    temporary_path.replace(RATE_CACHE)

    def get_rate(date: pd.Timestamp) -> float:
        date = pd.Timestamp(date).normalize()
        if date in rate_cache:
            return rate_cache[date]
        if unavailable_before is not None and date.date() < unavailable_before:
            RATE_ERRORS[date] = unavailable_reason
            return np.nan

        time.sleep(1.0)
        response = client.interest_rate_history_eod("SOFR", date.date(), date.date())
        rate = float(response["rate"][0]) / 100.0
        rate_cache[date] = rate
        cache_frame = pd.DataFrame(
            [{"date": cache_date, "rate": cache_rate} for cache_date, cache_rate in rate_cache.items()]
        ).sort_values("date")
        temporary_path = RATE_CACHE.with_name(f"{RATE_CACHE.name}.tmp")
        cache_frame.to_csv(temporary_path, index=False)
        temporary_path.replace(RATE_CACHE)
        return rate

    return get_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SPX quote/forward/IV adapter table")
    parser.add_argument("--rates-csv", type=Path, help="Existing table with date, expiry_date, forward and rate")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    chain = pd.read_csv(INPUT)
    expiry_inputs = pd.read_csv(args.rates_csv) if args.rates_csv else pd.DataFrame()
    RATE_ERRORS.clear()
    rate_provider = None if args.rates_csv else theta_rate_provider()
    enriched = enrich_option_chain(chain, expiry_inputs, rate_provider)
    enriched.to_csv(args.output, index=False)
    source = chain.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["expiry_date"] = pd.to_datetime(source["expiry_date"])
    group_columns = ["date", "expiry_date"]
    if "root" in source:
        group_columns.append("root")

    enriched_status = (
        enriched.groupby(group_columns)
        .agg(
            enriched_rows=("iv_mid", "size"),
            invalid_mid_iv_rows=("iv_mid", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )
    diagnostics = (
        source[group_columns]
        .drop_duplicates()
        .merge(enriched_status, on=group_columns, how="left")
    )
    diagnostics["root"] = diagnostics["root"] if "root" in diagnostics else ""
    diagnostics["status"] = np.where(
        diagnostics["enriched_rows"].isna(),
        "error",
        np.where(diagnostics["invalid_mid_iv_rows"] > 0, "warning", "ok"),
    )
    diagnostics["reason"] = np.select(
        [
            diagnostics["enriched_rows"].isna(),
            diagnostics["invalid_mid_iv_rows"].fillna(0) > 0,
        ],
        [
            "No usable forward/rate for expiry group",
            "Some midpoint IVs could not be solved",
        ],
        default="",
    )
    if RATE_ERRORS:
        diagnostics["reason"] = diagnostics.apply(
            lambda row: (
                f"Rate lookup failed: {RATE_ERRORS[pd.Timestamp(row['date']).normalize()]}"
                if pd.isna(row["enriched_rows"])
                and pd.Timestamp(row["date"]).normalize() in RATE_ERRORS
                else row["reason"]
            ),
            axis=1,
        )
    diagnostics.to_csv(DIAGNOSTICS, index=False)
    print(f"Rows written: {len(enriched):,}; dates: {enriched['date'].nunique()}")
    print(f"Valid bid/mid/ask IV rows: {enriched[['iv_bid', 'iv_mid', 'iv_ask']].notna().sum().to_dict()}")
    print(f"Saved: {args.output}")
    print(f"Diagnostics: {DIAGNOSTICS}")


if __name__ == "__main__":
    main()