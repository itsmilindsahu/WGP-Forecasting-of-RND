"""
Day 1 — Data collection: Yahoo Finance via yfinance.

Free daily option chains for liquid US equities/indices (e.g. SPY, QQQ). Good
for near-term snapshots to validate the extraction pipeline before committing
to a paid vendor for deep history.

Usage:
    python fetch_yfinance.py --ticker SPY --out data/yf_spy_chain.csv

Notes:
- Needs outbound network access to Yahoo's query endpoints, which this sandbox
  does not have — run it on your own machine.
"""
import argparse
import pandas as pd
import yfinance as yf


def fetch_chain(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    expiries = tk.options  # list of "YYYY-MM-DD" strings

    rows = []
    fetched_at = pd.Timestamp.utcnow()
    for exp in expiries:
        chain = tk.option_chain(exp)
        for side, df_side in (("call", chain.calls), ("put", chain.puts)):
            for _, r in df_side.iterrows():
                bid, ask = r.get("bid"), r.get("ask")
                iv = r.get("impliedVolatility")
                if bid is None or ask is None or iv is None:
                    continue
                if bid <= 0 or ask <= 0 or iv <= 0:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "expiration": exp,
                        "strike": r["strike"],
                        "option_type": side,
                        "mid_price": 0.5 * (bid + ask),
                        "mid_iv": iv,  # already a decimal (e.g. 0.21)
                        "underlying_price": spot,
                        "open_interest": r.get("openInterest"),
                        "volume": r.get("volume"),
                        "fetched_at": fetched_at.isoformat(),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["tau_years"] = (df["expiration"] - df["fetched_at"]).dt.total_seconds() / (
        365.0 * 24 * 3600
    )
    df = df[df["tau_years"] > 0].reset_index(drop=True)
    return df


def clean_chain(df: pd.DataFrame, min_strikes_per_expiry: int = 5) -> pd.DataFrame:
    df = df[(df["mid_price"] > 0) & (df["mid_iv"] > 0)].copy()
    counts = df.groupby("expiration")["strike"].transform("count")
    df = df[counts >= min_strikes_per_expiry].reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--out", default="data/yf_chain.csv")
    args = ap.parse_args()

    raw = fetch_chain(args.ticker)
    clean = clean_chain(raw)
    clean.to_csv(args.out, index=False)
    print(f"Wrote {len(clean)} rows to {args.out}")
