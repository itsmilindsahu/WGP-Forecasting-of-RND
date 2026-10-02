"""
Day 1 — Data collection: Deribit public API.

Deribit's public endpoints require no authentication and expose full order-book
/ settlement history for BTC and ETH options, which is why the plan uses it for
a fast end-to-end prototype.

Usage:
    python fetch_deribit.py --currency BTC --out data/deribit_btc_chain.csv

Notes:
- This talks to https://www.deribit.com/api/v2/public/... . It needs outbound
  network access to deribit.com, which this sandbox does not have — run it on
  your own machine.
"""
import argparse
import time
import requests
import pandas as pd

BASE_URL = "https://www.deribit.com/api/v2/public"


def get_instruments(currency: str, kind: str = "option") -> list[dict]:
    """List all live option instruments for a currency (e.g. BTC, ETH)."""
    r = requests.get(
        f"{BASE_URL}/get_instruments",
        params={"currency": currency, "kind": kind, "expired": "false"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["result"]


def get_order_book(instrument_name: str) -> dict:
    """Pull the current order book / mark price / greeks for one instrument."""
    r = requests.get(
        f"{BASE_URL}/get_order_book",
        params={"instrument_name": instrument_name},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["result"]


def fetch_chain(currency: str = "BTC", sleep_s: float = 0.1) -> pd.DataFrame:
    """
    Build a clean option-chain snapshot: one row per (expiry, strike, type)
    with mid price, mid IV, underlying index price, and time-to-maturity.
    """
    instruments = get_instruments(currency)
    rows = []
    for inst in instruments:
        name = inst["instrument_name"]
        try:
            ob = get_order_book(name)
        except requests.RequestException:
            continue

        best_bid = ob.get("best_bid_price")
        best_ask = ob.get("best_ask_price")
        mark_iv = ob.get("mark_iv")  # already in % per Deribit convention
        underlying = ob.get("underlying_price") or ob.get("index_price")
        if best_bid is None or best_ask is None or mark_iv is None:
            continue

        rows.append(
            {
                "instrument_name": name,
                "currency": currency,
                "strike": inst["strike"],
                "option_type": inst["option_type"],  # "call" / "put"
                "expiration_timestamp_ms": inst["expiration_timestamp"],
                "mid_price": 0.5 * (best_bid + best_ask),
                "mid_iv_pct": mark_iv,
                "underlying_price": underlying,
                "fetched_at": pd.Timestamp.utcnow().isoformat(),
            }
        )
        time.sleep(sleep_s)  # be polite to the public endpoint

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["expiration"] = pd.to_datetime(df["expiration_timestamp_ms"], unit="ms")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["tau_years"] = (
        df["expiration"] - df["fetched_at"]
    ).dt.total_seconds() / (365.0 * 24 * 3600)
    df = df[df["tau_years"] > 0].reset_index(drop=True)
    return df


def clean_chain(df: pd.DataFrame, min_strikes_per_expiry: int = 5) -> pd.DataFrame:
    """Drop thin expiries and obviously bad quotes."""
    df = df[(df["mid_price"] > 0) & (df["mid_iv_pct"] > 0)].copy()
    counts = df.groupby("expiration")["strike"].transform("count")
    df = df[counts >= min_strikes_per_expiry].reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--out", default="data/deribit_chain.csv")
    args = ap.parse_args()

    raw = fetch_chain(args.currency)
    clean = clean_chain(raw)
    clean.to_csv(args.out, index=False)
    print(f"Wrote {len(clean)} rows to {args.out}")
