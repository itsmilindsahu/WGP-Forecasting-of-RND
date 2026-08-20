"""
Day 2 — Extract the daily risk-neutral density via Breeden–Litzenberger.

Breeden & Litzenberger (1978): under no-arbitrage, the risk-neutral density of
the terminal underlying price is the second derivative of the (discounted)
call price with respect to strike:

    q(K) = e^{r*tau} * d^2 C / dK^2

We take the *fitted* SVI IV curve from Day 1 (smooth by construction), convert
to Black-Scholes call prices on a dense strike grid, and apply a central
finite-difference second derivative. Smoothing upstream is what keeps this
numerically stable — differentiating raw ticks directly is why the plan
explicitly smooths first.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_call_price(S0: float, K: np.ndarray, tau: float, r: float, sigma: np.ndarray) -> np.ndarray:
    """Black-76 call price, forward-measure form (S0 here IS the forward F).

    Fixed per Sven's review: this previously used the forward-measure d1
    (built from S0 = F directly, no drift term) but then multiplied by
    S0 * exp(r*tau) instead of S0, which implicitly treats S0 as the *spot*
    and re-applies a second forward adjustment -- so the effective forward
    priced into the option was F*e^{r*tau}, not F. Mixing the two
    conventions is a small error at 30-day tenors (~0.25%) that grows with
    maturity. Correct Black-76: C = e^{-r*tau}[F*N(d1) - K*N(d2)], with
    d1 = (ln(F/K) + 0.5*sigma^2*tau) / (sigma*sqrt(tau)).
    """
    sigma = np.maximum(sigma, 1e-6)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S0 / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    disc = np.exp(-r * tau)
    return disc * (S0 * norm.cdf(d1) - K * norm.cdf(d2))


def extract_rnd_from_curve(
    strike: np.ndarray, forward: float, tau: float, rate: float, iv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given one day's smoothed (strike, iv) curve, return (strike_mid, density)
    on the interior grid (finite differences lose the two boundary points).
    """
    order = np.argsort(strike)
    K, sig = strike[order], iv[order]

    C = bs_call_price(forward, K, tau, rate, sig)

    dK = np.diff(K)
    if not np.allclose(dK, dK[0], rtol=1e-3):
        # resample onto a uniform grid if the input grid isn't uniform
        K_uniform = np.linspace(K.min(), K.max(), len(K))
        C = np.interp(K_uniform, K, C)
        K = K_uniform
        dK = np.diff(K)

    h = dK[0]
    d2C = (C[2:] - 2 * C[1:-1] + C[:-2]) / (h**2)
    q = np.exp(rate * tau) * d2C
    q = np.maximum(q, 0.0)  # clip small negative numerical noise

    K_mid = K[1:-1]

    # renormalize to integrate to 1 (finite-difference + truncated support both
    # introduce small mass leakage; this is a standard practical correction).
    # NOTE: this renormalization makes "does it integrate to 1.0" a vacuous
    # check by construction -- see martingale_check() below for the real
    # correctness test, which this renormalization cannot paper over.
    mass = np.trapezoid(q, K_mid)
    if mass > 0:
        q = q / mass

    return K_mid, q


def martingale_check(
    K_mid: np.ndarray, density: np.ndarray, forward: float, tol: float = 0.01
) -> dict:
    """
    Real correctness test for an extracted RND (per Sven's review): under the
    risk-neutral measure the underlying is a martingale, so E[K] under the
    extracted density must equal the forward. Unlike "does it integrate to
    1.0" -- which extract_rnd_from_curve's renormalization guarantees will
    always pass -- this check can actually fail, and did: on the old
    +/-0.35-log-moneyness grid, truncating ~7% of the tail mass and then
    renormalizing pushed the extracted mean ~2.98% above the forward.

    Returns a dict with extracted_mean, forward, rel_error, and passed
    (True iff rel_error <= tol).
    """
    extracted_mean = float(np.trapezoid(K_mid * density, K_mid))
    rel_error = abs(extracted_mean - forward) / forward
    return {
        "extracted_mean": extracted_mean,
        "forward": float(forward),
        "rel_error": float(rel_error),
        "passed": bool(rel_error <= tol),
    }


def extract_all_days(curves: pd.DataFrame, tol: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply extract_rnd_from_curve to every day in the Day-1 iv_curves table.

    Returns (rnd_df, diagnostics_df); diagnostics_df carries the per-day
    martingale_check() results so extraction correctness can be audited
    across the whole sample, not just spot-checked on day 1.
    """
    rows = []
    diag_rows = []
    for date, g in curves.sort_values("strike").groupby("date"):
        g = g.sort_values("strike")
        forward = g["forward"].iloc[0]
        tau = g["tau_years"].iloc[0]
        # Use the chain's own rate if fit_iv_curve.py carried it through (real
        # data: 0.0, since Deribit's reported forward is already rate-adjusted);
        # fall back to simulate_data.py's constant 0.03 for older curves files
        # that predate the rate column.
        rate = g["rate"].iloc[0] if "rate" in g.columns else 0.03
        K_mid, q = extract_rnd_from_curve(
            g["strike"].to_numpy(), forward, tau, rate, g["fitted_iv"].to_numpy()
        )
        for K, dens in zip(K_mid, q):
            rows.append({"date": date, "strike": K, "density": dens, "forward": forward, "tau_years": tau})

        diag = martingale_check(K_mid, q, forward, tol=tol)
        diag["date"] = date
        diag_rows.append(diag)

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


if __name__ == "__main__":
    curves = pd.read_csv("../day1_data_collection/data/iv_curves.csv", parse_dates=["date"])
    rnd, diagnostics = extract_all_days(curves)
    rnd.to_csv("data/rnd_curves.csv", index=False)
    diagnostics.to_csv("data/martingale_diagnostics.csv", index=False)
    n_fail = (~diagnostics["passed"]).sum()
    print(f"Extracted RNDs for {rnd['date'].nunique()} days -> data/rnd_curves.csv")
    print(
        f"Martingale check (extracted mean vs forward, tol=1%): "
        f"{len(diagnostics) - n_fail}/{len(diagnostics)} days pass, "
        f"max rel_error = {diagnostics['rel_error'].max():.4%}"
    )
    print("Wrote data/martingale_diagnostics.csv")
