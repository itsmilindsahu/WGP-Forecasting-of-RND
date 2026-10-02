"""
Day 1 — Offline stand-in for fetch_deribit.py / fetch_yfinance.py.

Generates a multi-day synthetic option-chain time series using the raw SVI
parametrization for the implied-variance smile:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

where k = log(K / F) is log-moneyness and w = sigma_BS^2 * tau is total implied
variance. SVI parameters drift slowly day-to-day (level, skew, curvature) via a
small random walk so the resulting RND sequence has genuine temporal structure
for the GP in Day 4 to learn from, rather than being i.i.d. noise.

This module has no network dependency and is used by run_all.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def simulate_chain(
    n_days: int = 40,
    n_strikes: int = 41,
    tau_years: float = 30 / 365.0,
    spot0: float = 100.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame with columns:
    date, strike, option_type, mid_iv, forward, underlying_price, tau_years, rate
    covering `n_days` consecutive daily snapshots.
    """
    rng = np.random.default_rng(seed)

    # SVI parameter starting point, calibrated so ATM total variance
    # w(0) = a + b*sigma implies an ATM annualized vol of ~18% (equity-index
    # level), not the ~68% (crypto-level) the previous (a=0.02, b=0.15)
    # starting point implied: w(0) = 0.02 + 0.15*0.12 = 0.038 ->
    # iv = sqrt(0.038/0.0822) = 68%. Fixed per Sven's review: the docstring
    # says "equity-index smile" so the simulator should actually produce one.
    a, b, rho, m, sigma = 0.0005, 0.018, -0.5, 0.0, 0.12
    rate = 0.03

    spot = spot0
    rows = []
    dates = pd.bdate_range("2026-06-01", periods=n_days)

    for date in dates:
        # small random walk in log-price and in SVI params (mean-reverting-ish)
        spot *= np.exp(rng.normal(0, 0.01))
        a = max(1e-5, a + rng.normal(0, 0.00008))
        b = max(1e-4, b + rng.normal(0, 0.0005))
        rho = np.clip(rho + rng.normal(0, 0.02), -0.95, 0.95)
        m = np.clip(m + rng.normal(0, 0.005), -0.1, 0.1)
        sigma = max(0.03, sigma + rng.normal(0, 0.004))

        forward = spot * np.exp(rate * tau_years)

        # Strike grid widened to +/-4 std of log-moneyness (previously a
        # fixed +/-0.35, which was only ~1.8 std at the old 68%-vol
        # parameterization and truncated ~7% of the RND's mass -- see
        # Sven's review). std of log-moneyness over the option's horizon is
        # sqrt(ATM total variance) = sqrt(w(0)); grid is recomputed each day
        # since (a,b,rho,m,sigma) drift.
        w_atm = svi_total_variance(np.array([m]), a, b, rho, m, sigma)[0]
        std_logm = np.sqrt(max(w_atm, 1e-8))
        half_width = 4.0 * std_logm
        strikes_grid_k = np.linspace(-half_width, half_width, n_strikes)

        w = svi_total_variance(strikes_grid_k, a, b, rho, m, sigma)
        w = np.maximum(w, 1e-6)
        iv = np.sqrt(w / tau_years)

        strikes = forward * np.exp(strikes_grid_k)
        for K, sig in zip(strikes, iv):
            rows.append(
                {
                    "date": date,
                    "strike": K,
                    "option_type": "call",
                    "mid_iv": sig,
                    "forward": forward,
                    "underlying_price": spot,
                    "tau_years": tau_years,
                    "rate": rate,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_chain()
    df.to_csv("data/option_chains.csv", index=False)
    print(f"Wrote {len(df)} rows across {df['date'].nunique()} days to data/option_chains.csv")
