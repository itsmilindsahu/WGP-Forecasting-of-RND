"""
Day 1 — Fit a smoothed implied-vol curve per day.

Takes the cleaned option-chain long table (date, strike, option_type, mid_iv,
forward, tau_years, ...) and, for each day, fits the raw-SVI total-variance
parametrization by nonlinear least squares. The fitted curve is what Day 2
differentiates (Breeden–Litzenberger needs a smooth, arbitrage-consistent call
price curve — fitting raw market ticks directly is numerically unstable).

Output: one row per (date, strike) with the *fitted* mid_iv, plus the fitted
SVI parameters per day for reference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_svi_day(k: np.ndarray, w_obs: np.ndarray, x0=None) -> np.ndarray:
    """Least-squares fit of SVI params (a, b, rho, m, sigma) to one day's smile."""
    if x0 is None:
        x0 = np.array([max(w_obs.min(), 1e-4), 0.15, -0.3, 0.0, 0.1])

    def resid(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or not (-1 < rho < 1):
            return np.full_like(w_obs, 1e3)
        return svi_total_variance(k, a, b, rho, m, sigma) - w_obs

    lb = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    ub = [1.0, 5.0, 0.999, 1.0, 2.0]
    res = least_squares(resid, x0, bounds=(lb, ub), method="trf")
    return res.x


def smooth_all_days(chain: pd.DataFrame, n_grid: int = 61) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each date: fit SVI to (log-moneyness, total variance), then evaluate
    the fitted curve on a common, dense log-moneyness grid so every day is
    comparable strike-by-strike (needed before Day 2/3).

    Returns:
        curves_df: date, k (log-moneyness), strike, tau_years, fitted_iv
        params_df: date, a, b, rho, m, sigma  (fit diagnostics)
    """
    curve_rows, param_rows = [], []
    prev_params = None

    for date, g in chain.sort_values("strike").groupby("date"):
        forward = g["forward"].iloc[0]
        tau = g["tau_years"].iloc[0]
        rate = g["rate"].iloc[0] if "rate" in g.columns else 0.03
        k = np.log(g["strike"].to_numpy() / forward)
        w_obs = (g["mid_iv"].to_numpy() ** 2) * tau

        params = fit_svi_day(k, w_obs, x0=prev_params)
        prev_params = params
        a, b, rho, m, sigma = params
        param_rows.append(
            {"date": date, "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}
        )

        k_grid = np.linspace(k.min(), k.max(), n_grid)
        w_grid = svi_total_variance(k_grid, *params)
        iv_grid = np.sqrt(np.maximum(w_grid, 1e-10) / tau)
        strike_grid = forward * np.exp(k_grid)

        for kk, K, iv in zip(k_grid, strike_grid, iv_grid):
            curve_rows.append(
                {
                    "date": date,
                    "k": kk,
                    "strike": K,
                    "forward": forward,
                    "tau_years": tau,
                    "rate": rate,
                    "fitted_iv": iv,
                }
            )

    curves_df = pd.DataFrame(curve_rows)
    params_df = pd.DataFrame(param_rows)
    return curves_df, params_df


if __name__ == "__main__":
    chain = pd.read_csv("data/option_chains.csv", parse_dates=["date"])
    curves, params = smooth_all_days(chain)
    curves.to_csv("data/iv_curves.csv", index=False)
    params.to_csv("data/svi_params.csv", index=False)
    print(f"Fitted SVI smiles for {params['date'].nunique() if 'date' in params else len(params)} days")
    print(f"Wrote data/iv_curves.csv ({len(curves)} rows) and data/svi_params.csv")
