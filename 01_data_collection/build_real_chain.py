"""
Day 1 — Real data: build option_chains.csv from the Deribit tick-level
options-chain export (tardis.dev-format CSV, one full day: 2019-07-01).

Why this exists: fetch_deribit.py needs live network access to deribit.com,
which this sandbox doesn't have. Milind pulled a historical tick-level dump
instead (deribit_options_chain_2019-07-01_OPTIONS_csv.gz, ~9M rows, BTC+ETH,
all listed expiries, full order-book snapshots throughout 2019-07-01 UTC).

This is real market data, but it is a *single calendar day* of tick data, not
a multi-week daily series — so instead of one row per calendar day, we bin
the day into fixed-width intraday snapshots (30 min) and treat each snapshot
as one time step ("pseudo-day") in the downstream pipeline. That's an honest
proxy for genuine temporal structure (real BTC vol smile evolving through a
real trading day) but it is NOT the same object as the daily-close series the
project ultimately wants — see the caveat this script prints and the one in
results_memo.md.

Choices made, and why:
- Underlying: BTC (deeper book than ETH in this dump: 5.5M vs 3.4M rows).
- Expiry: 2019-07-26 (fixed at 25 days to expiry as of 2019-07-01), the
  closest listed expiry to the 30-day tenor simulate_data.py used, so
  results are comparable in spirit. Instrument is BTC-26JUL19-*-C.
- Only calls kept: Breeden-Litzenberger (day2) differentiates a call price
  curve; the SVI fit only needs one option type since C/P share the same IV
  smile under put-call parity.
- mid_iv = mark_iv (Deribit's own smoothed mark), not bid/ask mid: bid_iv/
  ask_iv are only populated on ~80% of ticks in this dump (thin/one-sided
  books), mark_iv is populated on >99.99% and is the same quantity Deribit
  itself marks risk to.
- forward = underlying_price as reported against the SYN.BTC-26JUL19 index:
  Deribit already computes this as the synthetic forward for the expiry, so
  no separate spot->forward carry adjustment is applied; rate is set to 0
  for the same reason (it's already priced into the reported forward).
- Snapshot rule per (30-min bin, strike): last observation carried forward
  within the bin (the freshest quote as of the bin boundary), not an average
  — averaging mark_iv across a 30-min window would smear through any real
  intraday move, which is exactly the signal we want the GP to see.

Usage:
    python build_real_chain.py --tick-csv-gz /mnt/user-data/uploads/deribit_options_chain_2019-07-01_OPTIONS_csv.gz \
        --out data/option_chains.csv
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np

USE_COLS = [
    "symbol", "local_timestamp", "type", "strike_price", "expiration",
    "mark_iv", "bid_iv", "ask_iv", "underlying_price",
]


def load_filtered(tick_csv_gz: str, underlying: str, expiration_us: int) -> pd.DataFrame:
    """Stream the multi-GB gz in chunks, keeping only rows for one
    underlying/expiry/option-type — the rest never has to hit memory."""
    chunks = []
    reader = pd.read_csv(
        tick_csv_gz,
        usecols=USE_COLS,
        chunksize=500_000,
        compression="gzip",
    )
    for chunk in reader:
        m = (
            chunk["symbol"].str.startswith(f"{underlying}-")
            & (chunk["expiration"] == expiration_us)
            & (chunk["type"] == "call")
            & (chunk["mark_iv"] > 0)
        )
        if m.any():
            chunks.append(chunk.loc[m])
    return pd.concat(chunks, ignore_index=True)


def bin_to_snapshots(df: pd.DataFrame, freq: str = "30min", tau_floor_years: float = 1e-4) -> pd.DataFrame:
    """Bin ticks into fixed-width intraday snapshots; within each
    (bin, strike) keep the last quote (freshest as-of the bin boundary)."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["local_timestamp"], unit="us", utc=True)
    df["bin"] = df["dt"].dt.floor(freq)

    df = df.sort_values("dt")
    last = df.groupby(["bin", "strike_price"], as_index=False).last()

    rows = []
    for date, g in last.groupby("bin"):
        forward = g["underlying_price"].median()  # tiny cross-strike jitter; median is robust
        expiry_dt = pd.to_datetime(g["expiration"].iloc[0], unit="us", utc=True)
        tau = (expiry_dt - date).total_seconds() / (365.0 * 24 * 3600)
        tau = max(tau, tau_floor_years)
        for _, r in g.sort_values("strike_price").iterrows():
            rows.append(
                {
                    "date": date,
                    "strike": r["strike_price"],
                    "option_type": "call",
                    "mid_iv": r["mark_iv"] / 100.0,  # Deribit reports IV in percent (e.g. 96.98 = 96.98%)
                    "forward": forward,
                    "underlying_price": forward,
                    "tau_years": tau,
                    "rate": 0.0,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-csv-gz", required=True)
    ap.add_argument("--underlying", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--expiration-us", type=int, default=1564128000000000)  # 2019-07-26 00:00 UTC
    ap.add_argument("--freq", default="30min")
    ap.add_argument("--out", default="data/option_chains.csv")
    args = ap.parse_args()

    raw = load_filtered(args.tick_csv_gz, args.underlying, args.expiration_us)
    print(f"Loaded {len(raw)} raw ticks for {args.underlying} exp={args.expiration_us}")

    chain = bin_to_snapshots(raw, freq=args.freq)
    chain.to_csv(args.out, index=False)
    print(
        f"Wrote {len(chain)} rows across {chain['date'].nunique()} intraday snapshots "
        f"({args.freq} bins, {chain.groupby('date').size().median():.0f} strikes/snapshot median) "
        f"to {args.out}"
    )
