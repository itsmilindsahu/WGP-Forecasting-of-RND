from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


INPUT = r"01_data_collection\data\spx_30d_smiles.csv"
CURVE_OUTPUT = r"01_data_collection\data\svi_30d_curves.csv"
PARAM_OUTPUT = r"01_data_collection\data\svi_30d_params.csv"

N_GRID = 81


def svi_total_variance(k, a, b, rho, m, sigma):
    """
    SVI total variance:
        w(k) = a + b * [rho(k-m) + sqrt((k-m)^2 + sigma^2)]
    """
    return a + b * (
        rho * (k - m)
        + np.sqrt((k - m) ** 2 + sigma**2)
    )


def fit_svi_day(k, w_obs, x0=None):
    """
    Fit SVI to one 30D total-variance smile.
    """

    if x0 is None:
        x0 = np.array([
            max(float(w_obs.min()) * 0.8, 1e-5),
            0.15,
            -0.3,
            0.0,
            0.10
        ])

    def residuals(params):
        a, b, rho, m, sigma = params

        w_model = svi_total_variance(
            k, a, b, rho, m, sigma
        )

        return w_model - w_obs

    # SVI parameter bounds
    lower = np.array([
        -1.0,      # a
        1e-6,      # b
        -0.999,    # rho
        -1.0,      # m
        1e-4       # sigma
    ])

    upper = np.array([
        1.0,       # a
        5.0,       # b
        0.999,     # rho
        1.0,       # m
        2.0        # sigma
    ])

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=5000
    )

    return result.x, result.cost, result.success


def main():

    print("=" * 70)
    print("FITTING SVI TO 30D SPX SMILES")
    print("=" * 70)

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"]
    )

    print(f"Input rows : {len(df)}")
    print(f"Dates      : {df['date'].nunique()}")

    curve_rows = []
    param_rows = []

    previous_params = None

    for date, g in df.groupby("date"):

        g = g.sort_values("k").copy()

        k = g["k"].to_numpy(dtype=float)
        w = g["total_variance"].to_numpy(dtype=float)

        # Remove invalid observations
        mask = (
            np.isfinite(k)
            & np.isfinite(w)
            & (w >= 0)
        )

        k = k[mask]
        w = w[mask]

        if len(k) < 20:
            print(f"Skipping {date.date()} - only {len(k)} points")
            continue

        # Fit SVI
        params, cost, success = fit_svi_day(
            k,
            w,
            x0=previous_params
        )

        a, b, rho, m, sigma = params

        previous_params = params

        # Dense fitted grid
        k_grid = np.linspace(
            k.min(),
            k.max(),
            N_GRID
        )

        w_grid = svi_total_variance(
            k_grid,
            a,
            b,
            rho,
            m,
            sigma
        )

        # Numerical safety
        w_grid = np.maximum(
            w_grid,
            1e-10
        )

        tau = float(g["tau_years"].iloc[0])

        iv_grid = np.sqrt(
            w_grid / tau
        )

        forward = float(
            g["forward_30d"].iloc[0]
        )

        rate = float(
            g["rate"].iloc[0]
        )

        # Store fitted curve
        for kk, ww, iv in zip(
            k_grid,
            w_grid,
            iv_grid
        ):

            K = forward * np.exp(kk)

            curve_rows.append({
                "date": date,
                "k": kk,
                "strike": K,
                "forward": forward,
                "tau_years": tau,
                "rate": rate,
                "total_variance": ww,
                "fitted_iv": iv
            })

        # Calculate fit diagnostics
        fitted_obs = svi_total_variance(
            k,
            *params
        )

        rmse = np.sqrt(
            np.mean(
                (fitted_obs - w) ** 2
            )
        )

        param_rows.append({
            "date": date,
            "a": a,
            "b": b,
            "rho": rho,
            "m": m,
            "sigma": sigma,
            "rmse_total_variance": rmse,
            "cost": cost,
            "success": success,
            "n_points": len(k)
        })

        print(
            f"{date.date()} | "
            f"RMSE={rmse:.6f} | "
            f"rho={rho:.4f} | "
            f"b={b:.4f}"
        )

    curves = pd.DataFrame(curve_rows)
    params = pd.DataFrame(param_rows)

    curves.to_csv(
        CURVE_OUTPUT,
        index=False
    )

    params.to_csv(
        PARAM_OUTPUT,
        index=False
    )

    print("\n" + "=" * 70)
    print(f"SVI curves created : {len(params)}")
    print(f"Curve points       : {len(curves)}")
    print(f"Parameter rows     : {len(params)}")
    print("=" * 70)

    print(f"\nSaved:")
    print(CURVE_OUTPUT)
    print(PARAM_OUTPUT)


if __name__ == "__main__":
    main()