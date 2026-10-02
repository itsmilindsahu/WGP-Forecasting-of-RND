from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# FILE PATHS
# ============================================================

INPUT = r"01_data_collection\data\svi_30d_curves.csv"

RND_OUTPUT = r"02_rnd_extraction\data\rnd_30d.csv"
DIAG_OUTPUT = r"02_rnd_extraction\data\rnd_30d_diagnostics.csv"


# ============================================================
# BLACK-76 CALL PRICE
# ============================================================

def bs_call_price(
    forward: float,
    strike: np.ndarray,
    tau: float,
    rate: float,
    sigma: np.ndarray
) -> np.ndarray:

    sigma = np.maximum(sigma, 1e-8)

    sqrt_tau = np.sqrt(tau)

    d1 = (
        np.log(forward / strike)
        + 0.5 * sigma**2 * tau
    ) / (sigma * sqrt_tau)

    d2 = d1 - sigma * sqrt_tau

    discount = np.exp(-rate * tau)

    return discount * (
        forward * norm.cdf(d1)
        - strike * norm.cdf(d2)
    )


# ============================================================
# BREEDEN-LITZENBERGER
# ============================================================

def extract_rnd(
    strike,
    forward,
    tau,
    rate,
    iv
):

    # Sort by strike
    order = np.argsort(strike)

    K = np.asarray(strike)[order]
    IV = np.asarray(iv)[order]

    # Convert IV to Black-76 call prices
    C = bs_call_price(
        forward,
        K,
        tau,
        rate,
        IV
    )

    # Uniform strike grid
    K_uniform = np.linspace(
        K.min(),
        K.max(),
        len(K)
    )

    C_uniform = np.interp(
        K_uniform,
        K,
        C
    )

    K = K_uniform
    C = C_uniform

    # Strike spacing
    h = K[1] - K[0]

    # Second derivative of call price
    d2C = (
        C[2:]
        - 2.0 * C[1:-1]
        + C[:-2]
    ) / (h ** 2)

    # Breeden-Litzenberger density
    density_raw = (
        np.exp(rate * tau)
        * d2C
    )

    K_mid = K[1:-1]

    # Remove tiny negative numerical noise
    density = np.maximum(
        density_raw,
        0.0
    )

    # Mass BEFORE normalization
    raw_mass = np.trapezoid(
        density,
        K_mid
    )

    # Normalize density
    if raw_mass > 0:
        density = density / raw_mass

    return (
        K_mid,
        density,
        density_raw,
        raw_mass
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def calculate_diagnostics(
    K,
    density,
    density_raw,
    forward,
    rate,
    tau,
    raw_mass
):

    # Probability mass after normalization
    normalized_mass = np.trapezoid(
        density,
        K
    )

    # Expected terminal price
    extracted_mean = np.trapezoid(
        K * density,
        K
    )

    # Martingale condition:
    # E[S_T] should equal forward
    martingale_error = (
        abs(extracted_mean - forward)
        / forward
    )

    # Fraction of grid where raw density was negative
    negative_fraction = np.mean(
        density_raw < 0
    )

    # Minimum raw density
    min_raw_density = np.min(
        density_raw
    )

    # Call convexity:
    # d²C/dK² >= 0
    call_convexity_pass = bool(
        np.all(density_raw >= -1e-8)
    )

    density_nonnegative = bool(
        np.all(density >= 0)
    )

    martingale_pass = bool(
        martingale_error <= 0.01
    )

    mass_pass = bool(
        abs(normalized_mass - 1.0) <= 1e-6
    )

    return {
        "forward": forward,
        "tau_years": tau,
        "rate": rate,

        "raw_mass": raw_mass,
        "normalized_mass": normalized_mass,

        "extracted_mean": extracted_mean,

        "martingale_rel_error": martingale_error,
        "martingale_pass": martingale_pass,

        "negative_density_fraction": negative_fraction,
        "min_raw_density": min_raw_density,

        "density_nonnegative": density_nonnegative,
        "call_convexity_pass": call_convexity_pass,
        "mass_pass": mass_pass
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BREEDEN-LITZENBERGER RND EXTRACTION")
    print("=" * 70)

    # Load SVI curves
    curves = pd.read_csv(
        INPUT,
        parse_dates=["date"]
    )

    print(f"Input rows : {len(curves)}")
    print(f"Dates      : {curves['date'].nunique()}")
    print()

    rnd_rows = []
    diagnostic_rows = []

    # Process every trading day
    for date, g in curves.groupby("date"):

        g = g.sort_values("strike")

        forward = float(
            g["forward"].iloc[0]
        )

        tau = float(
            g["tau_years"].iloc[0]
        )

        rate = float(
            g["rate"].iloc[0]
        )

        # Extract RND
        K, density, density_raw, raw_mass = extract_rnd(
            g["strike"].to_numpy(),
            forward,
            tau,
            rate,
            g["fitted_iv"].to_numpy()
        )

        # Store density
        for strike, q in zip(K, density):

            rnd_rows.append({
                "date": date,
                "strike": strike,
                "density": q,
                "forward": forward,
                "tau_years": tau
            })

        # Calculate diagnostics
        diag = calculate_diagnostics(
            K,
            density,
            density_raw,
            forward,
            rate,
            tau,
            raw_mass
        )

        diag["date"] = date

        diagnostic_rows.append(diag)

        print(
            f"{date.date()} | "
            f"raw mass={raw_mass:.4f} | "
            f"mean={diag['extracted_mean']:.2f} | "
            f"forward={forward:.2f} | "
            f"martingale error="
            f"{diag['martingale_rel_error']:.2%}"
        )

    # Convert to DataFrames
    rnd = pd.DataFrame(
        rnd_rows
    )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    # Save
    rnd.to_csv(
        RND_OUTPUT,
        index=False
    )

    diagnostics.to_csv(
        DIAG_OUTPUT,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RND EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"RND days   : "
        f"{rnd['date'].nunique()}"
    )

    print(
        f"RND points : "
        f"{len(rnd)}"
    )

    print()
    print(f"Saved:")
    print(RND_OUTPUT)
    print(DIAG_OUTPUT)

    print()
    print("DIAGNOSTICS")
    print("-" * 70)

    mass_pass_count = int(
        diagnostics["mass_pass"].sum()
    )

    martingale_pass_count = int(
        diagnostics["martingale_pass"].sum()
    )

    nonnegative_count = int(
        diagnostics["density_nonnegative"].sum()
    )

    convexity_pass_count = int(
        diagnostics["call_convexity_pass"].sum()
    )

    total_days = len(
        diagnostics
    )

    max_martingale_error = diagnostics[
        "martingale_rel_error"
    ].max()

    print(
        f"Mass ≈ 1              : "
        f"{mass_pass_count}/{total_days}"
    )

    print(
        f"Martingale pass       : "
        f"{martingale_pass_count}/{total_days}"
    )

    print(
        f"Non-negative density  : "
        f"{nonnegative_count}/{total_days}"
    )

    print(
        f"Call convexity pass   : "
        f"{convexity_pass_count}/{total_days}"
    )

    print(
        f"Max martingale error  : "
        f"{max_martingale_error:.2%}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()