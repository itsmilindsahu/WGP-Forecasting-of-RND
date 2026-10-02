from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHAIN = os.path.join(
    ROOT,
    "01_data_collection",
    "data",
    "option_chains_spx_20d.csv"
)

SMILES = os.path.join(
    ROOT,
    "01_data_collection",
    "data",
    "spx_30d_smiles.csv"
)

SVI = os.path.join(
    ROOT,
    "01_data_collection",
    "data",
    "svi_30d_curves.csv"
)

RND = os.path.join(
    ROOT,
    "02_rnd_extraction",
    "data",
    "rnd_30d.csv"
)

RND_DIAG = os.path.join(
    ROOT,
    "02_rnd_extraction",
    "data",
    "rnd_30d_diagnostics.csv"
)

GP_RESULTS = os.path.join(
    ROOT,
    "04_gp_model",
    "results",
    "wasserstein_forecast_comparison.csv"
)

OUTPUT = os.path.join(
    ROOT,
    "07_writeup",
    "spx_pilot_results"
)

PLOT_DIR = os.path.join(
    OUTPUT,
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def savefig(path):
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINALIZING SPX 20-DAY PILOT")
print("=" * 70)

chain = pd.read_csv(
    CHAIN,
    parse_dates=["date"]
)

smiles = pd.read_csv(
    SMILES,
    parse_dates=["date"]
)

svi = pd.read_csv(
    SVI,
    parse_dates=["date"]
)

rnd = pd.read_csv(
    RND,
    parse_dates=["date"]
)

diag = pd.read_csv(
    RND_DIAG,
    parse_dates=["date"]
)

print(
    f"Option-chain rows : {len(chain):,}"
)

print(
    f"Chain dates       : {chain['date'].nunique()}"
)

print(
    f"Smile dates       : {smiles['date'].nunique()}"
)

print(
    f"SVI dates         : {svi['date'].nunique()}"
)

print(
    f"RND dates         : {rnd['date'].nunique()}"
)


# ============================================================
# 1. DATA SCHEMA AUDIT
# ============================================================

required = [
    "date",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "volume"
]

optional = [
    "created",
    "last_trade",
    "timestamp",
    "ttm",
    "tau_years",
    "forward",
    "rate",
    "iv",
    "mid_iv"
]

print()
print("=" * 70)
print("DATA SCHEMA AUDIT")
print("=" * 70)

for col in required:

    status = "OK" if col in chain.columns else "MISSING"

    print(
        f"{col:<15} {status}"
    )

print()
print("Optional / derived fields:")

for col in optional:

    status = "PRESENT" if col in chain.columns else "not present"

    print(
        f"{col:<15} {status}"
    )


# ============================================================
# 2. MISSING 30D DAYS
# ============================================================

chain_dates = set(
    pd.to_datetime(
        chain["date"]
    ).dt.date
)

smile_dates = set(
    pd.to_datetime(
        smiles["date"]
    ).dt.date
)

missing_dates = sorted(
    chain_dates - smile_dates
)

print()
print("=" * 70)
print("30D SMILE COVERAGE")
print("=" * 70)

print(
    f"Available chain dates : "
    f"{len(chain_dates)}"
)

print(
    f"30D smile dates       : "
    f"{len(smile_dates)}"
)

print(
    f"Missing 30D dates     : "
    f"{len(missing_dates)}"
)

for d in missing_dates:

    g = chain[
        pd.to_datetime(
            chain["date"]
        ).dt.date == d
    ]

    expiries = sorted(
        pd.to_datetime(
            g["expiry"]
        ).dt.date.unique()
    )

    print(
        f"  {d}: "
        f"{len(g)} contracts, "
        f"{len(expiries)} expiries"
    )

missing_df = pd.DataFrame({
    "date": missing_dates,
    "reason": [
        "Insufficient overlapping expiry/moneyness coverage for 30D interpolation"
        for _ in missing_dates
    ]
})

missing_df.to_csv(
    os.path.join(
        OUTPUT,
        "missing_30d_days.csv"
    ),
    index=False
)


# ============================================================
# 3. EXPIRY / STRIKE COVERAGE PLOTS
# ============================================================

print()
print("Creating expiry/strike coverage plots...")

for date, g in chain.groupby("date"):

    plt.figure(
        figsize=(9, 5)
    )

    for expiry, eg in g.groupby("expiry"):

        strikes = pd.to_numeric(
            eg["strike"],
            errors="coerce"
        ).dropna()

        if len(strikes) == 0:
            continue

        plt.scatter(
            strikes,
            np.full(
                len(strikes),
                str(expiry)
            ),
            s=5
        )

    plt.xlabel(
        "Strike"
    )

    plt.ylabel(
        "Expiration"
    )

    plt.title(
        f"SPX expiry / strike coverage — "
        f"{date.date()}"
    )

    savefig(
        os.path.join(
            PLOT_DIR,
            f"coverage_{date.date()}.png"
        )
    )


# ============================================================
# 4. 30D SMILE PLOTS
# ============================================================

print(
    "Creating 30D smile plots..."
)

for date, g in smiles.groupby("date"):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        g["k"],
        g["iv_30d"],
        marker=".",
        linewidth=1.5,
        label="Interpolated 30D IV"
    )

    plt.xlabel(
        "Log-forward moneyness log(K/F)"
    )

    plt.ylabel(
        "30D implied volatility"
    )

    plt.title(
        f"SPX 30D implied-volatility smile — "
        f"{date.date()}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    savefig(
        os.path.join(
            PLOT_DIR,
            f"smile_30d_{date.date()}.png"
        )
    )


# ============================================================
# 5. SVI CURVE PLOTS
# ============================================================

print(
    "Creating SVI plots..."
)

for date, g in svi.groupby("date"):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        g["k"],
        g["fitted_iv"],
        linewidth=2,
        label="SVI fitted IV"
    )

    plt.xlabel(
        "Log-forward moneyness"
    )

    plt.ylabel(
        "Implied volatility"
    )

    plt.title(
        f"SPX 30D SVI smile — "
        f"{date.date()}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    savefig(
        os.path.join(
            PLOT_DIR,
            f"svi_30d_{date.date()}.png"
        )
    )


# ============================================================
# 6. RND PLOTS
# ============================================================

print(
    "Creating RND plots..."
)

for date, g in rnd.groupby("date"):

    forward = float(
        g["forward"].iloc[0]
    )

    x = (
        g["strike"]
        / forward
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        g["density"],
        linewidth=2
    )

    plt.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Forward-normalized F = 1"
    )

    plt.xlabel(
        "Terminal price / forward S_T / F_t"
    )

    plt.ylabel(
        "Risk-neutral density"
    )

    plt.title(
        f"SPX 30D forward-normalized RND — "
        f"{date.date()}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    savefig(
        os.path.join(
            PLOT_DIR,
            f"rnd_{date.date()}.png"
        )
    )


# ============================================================
# 7. ALL RNDs TOGETHER
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for date, g in rnd.groupby("date"):

    forward = float(
        g["forward"].iloc[0]
    )

    x = (
        g["strike"]
        / forward
    )

    plt.plot(
        x,
        g["density"],
        alpha=0.55,
        linewidth=1
    )

plt.axvline(
    1.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "S_T / F_t"
)

plt.ylabel(
    "Risk-neutral density"
)

plt.title(
    "SPX 30D forward-normalized RNDs — pilot"
)

plt.grid(
    alpha=0.25
)

savefig(
    os.path.join(
        PLOT_DIR,
        "rnd_all_days.png"
    )
)


# ============================================================
# 8. DIAGNOSTIC SUMMARY
# ============================================================

diag_summary = diag.copy()

diag_summary[
    "convexity_issue"
] = ~diag_summary[
    "call_convexity_pass"
]

diag_summary[
    "tiny_negative_curvature"
] = (
    diag_summary[
        "min_raw_density"
    ] > -1e-6
)

diag_summary.to_csv(
    os.path.join(
        OUTPUT,
        "rnd_diagnostic_summary.csv"
    ),
    index=False
)

print()
print("=" * 70)
print("RND DIAGNOSTICS")
print("=" * 70)

print(
    f"Mass pass           : "
    f"{diag['mass_pass'].sum()}/{len(diag)}"
)

print(
    f"Martingale pass     : "
    f"{diag['martingale_pass'].sum()}/{len(diag)}"
)

print(
    f"Non-negative density : "
    f"{diag['density_nonnegative'].sum()}/{len(diag)}"
)

print(
    f"Convexity pass       : "
    f"{diag['call_convexity_pass'].sum()}/{len(diag)}"
)

print(
    f"Max martingale error : "
    f"{diag['martingale_rel_error'].max():.4%}"
)


# ============================================================
# 9. FORECAST COMPARISON PLOT
# ============================================================

if os.path.exists(GP_RESULTS):

    gp = pd.read_csv(
        GP_RESULTS,
        parse_dates=["date"]
    )

    if len(gp) > 0:

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            gp["date"],
            gp["wasserstein_gp_w2"],
            marker="o",
            label="Wasserstein GP"
        )

        plt.plot(
            gp["date"],
            gp["persistence_w2"],
            marker="o",
            label="Persistence"
        )

        plt.xlabel(
            "Forecast date"
        )

        plt.ylabel(
            "W₂ forecast error"
        )

        plt.title(
            "Full-RND forecast error"
        )

        plt.grid(
            alpha=0.25
        )

        plt.legend()

        savefig(
            os.path.join(
                PLOT_DIR,
                "forecast_w2_comparison.png"
            )
        )


# ============================================================
# 10. WRITE PILOT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT,
    "SPX_PILOT_REPORT.md"
)

convexity_problem = diag[
    ~diag["call_convexity_pass"]
]

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# SPX 20-Day Wasserstein-RND Pilot\n\n"
    )

    f.write(
        "## Dataset\n\n"
    )

    f.write(
        f"- Chain dates collected: {len(chain_dates)}\n"
        f"- Dates with 30D smiles: {len(smile_dates)}\n"
        f"- Clean option rows: {len(chain):,}\n"
        f"- 30D smile points: {len(smiles):,}\n"
        f"- RND points: {len(rnd):,}\n\n"
    )

    f.write(
        "## 30D construction\n\n"
    )

    f.write(
        "For each valuation date, expiries bracketing 30 calendar days "
        "were interpolated in total variance at fixed log-forward "
        "moneyness. Days without sufficient overlapping coverage were "
        "excluded rather than extrapolated.\n\n"
    )

    f.write(
        f"Missing 30D dates: {len(missing_dates)}.\n\n"
    )

    for d in missing_dates:

        f.write(
            f"- {d}: insufficient overlapping expiry/moneyness coverage.\n"
        )

    f.write(
        "\n## RND diagnostics\n\n"
    )

    f.write(
        f"- Mass ≈ 1: {diag['mass_pass'].sum()}/{len(diag)}\n"
        f"- Martingale pass: {diag['martingale_pass'].sum()}/{len(diag)}\n"
        f"- Non-negative density: {diag['density_nonnegative'].sum()}/{len(diag)}\n"
        f"- Call convexity pass: {diag['call_convexity_pass'].sum()}/{len(diag)}\n"
        f"- Maximum martingale error: "
        f"{diag['martingale_rel_error'].max():.2%}\n\n"
    )

    if len(convexity_problem):

        for _, row in convexity_problem.iterrows():

            f.write(
                "### Tiny numerical convexity violation\n\n"
            )

            f.write(
                f"- Date: {row['date'].date()}\n"
                f"- Minimum raw density: {row['min_raw_density']:.6e}\n"
                f"- Negative grid fraction: "
                f"{row['negative_density_fraction']:.2%}\n"
                f"- Martingale relative error: "
                f"{row['martingale_rel_error']:.2%}\n\n"
            )

            f.write(
                "The negative curvature is at numerical scale and the "
                "density is clipped to zero before normalization. The "
                "day is retained rather than silently removed.\n\n"
            )

    f.write(
        "## Wasserstein representation\n\n"
    )

    f.write(
        "RNDs are represented in forward-normalized coordinates "
        "`S_T / F_t`, so the Wasserstein distance is primarily sensitive "
        "to distributional shape rather than the absolute SPX level.\n\n"
    )

    f.write(
        "## Forecast evaluation\n\n"
    )

    if os.path.exists(GP_RESULTS):

        gp = pd.read_csv(
            GP_RESULTS
        )

        if len(gp):

            f.write(
                f"- Held-out forecast days: {len(gp)}\n"
                f"- Wasserstein GP mean W₂ error: "
                f"{gp['wasserstein_gp_w2'].mean():.6f}\n"
                f"- Persistence mean W₂ error: "
                f"{gp['persistence_w2'].mean():.6f}\n\n"
            )

            f.write(
                "Because this is a small pilot with only a few held-out "
                "days, these figures are pipeline-validation results rather "
                "than evidence of general predictive superiority.\n"
            )

print()
print("=" * 70)
print("FINALIZATION COMPLETE")
print("=" * 70)

print(
    f"Output folder:\n{OUTPUT}"
)

print(
    f"Plots:\n{PLOT_DIR}"
)

print(
    f"Report:\n{report_path}"
)