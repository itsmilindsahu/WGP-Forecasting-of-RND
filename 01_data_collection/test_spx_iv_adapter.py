from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_spx_iv import (
    black76_price,
    enrich_option_chain,
    implied_volatilities,
    implied_volatility,
)


def test_adapter_computes_bid_mid_ask_iv():
    valuation_date = pd.Timestamp("2026-01-01")
    expiry_date = pd.Timestamp("2026-01-31")
    forward, strike, rate = 100.0, 100.0, 0.03
    tau = (expiry_date - valuation_date).days / 365.0
    quotes = {
        side: black76_price(forward, strike, tau, rate, volatility, "CALL")
        for side, volatility in (("bid", 0.19), ("mid", 0.20), ("ask", 0.21))
    }
    chain = pd.DataFrame([{
        "date": valuation_date,
        "expiry_date": expiry_date,
        "expiry": expiry_date,
        "strike": strike,
        "right": "CALL",
        "option_type": "CALL",
        "bid": quotes["bid"],
        "mid": quotes["mid"],
        "ask": quotes["ask"],
        "volume": 1,
    }])
    expiry_inputs = pd.DataFrame([{
        "date": valuation_date,
        "expiry_date": expiry_date,
        "forward": forward,
        "rate": rate,
    }])

    result = enrich_option_chain(chain, expiry_inputs)

    assert np.allclose(
        result[["iv_bid", "iv_mid", "iv_ask"]].iloc[0].to_numpy(dtype=float),
        [0.19, 0.20, 0.21],
        atol=1e-8,
    )
    assert result.loc[0, "forward"] == forward
    assert result.loc[0, "rate"] == rate
    assert result.loc[0, "tau_years"] == tau


def test_vectorized_iv_matches_scalar_call_and_put_solver():
    forward, tau, rate = 5000.0, 30 / 365, 0.045
    strikes = np.array([4700.0, 4900.0, 5000.0, 5100.0, 5300.0])
    option_types = np.array(["CALL", "PUT", "CALL", "PUT", "CALL"])
    volatilities = np.array([0.18, 0.22, 0.35, 0.28, 0.4])
    prices = np.array([
        black76_price(forward, strike, tau, rate, volatility, option_type)
        for strike, volatility, option_type in zip(strikes, volatilities, option_types)
    ])

    actual = implied_volatilities(
        prices,
        forward,
        strikes,
        tau,
        rate,
        option_types,
    )
    expected = np.array([
        implied_volatility(price, forward, strike, tau, rate, option_type)
        for price, strike, option_type in zip(prices, strikes, option_types)
    ])

    assert np.allclose(actual, expected, rtol=0, atol=1e-8)
    invalid = implied_volatilities(
        [0.0, np.nan],
        forward,
        [forward, forward],
        tau,
        rate,
        ["CALL", "PUT"],
    )
    assert np.isnan(invalid).all()


def test_adapter_preserves_snapshot_timestamp_and_finite_iv_across_expiries():
    valuation_date = pd.Timestamp("2024-07-16")
    group_inputs = [
        (pd.Timestamp("2024-08-02"), "SPX", 0.21, 0.025),
        (pd.Timestamp("2024-08-16"), "SPXW", 0.24, 0.027),
        (pd.Timestamp("2024-09-20"), "SPXW", 0.19, 0.029),
    ]
    chain_rows = []
    expiry_rows = []
    expected_ivs = []
    for expiry, root, volatility, rate in group_inputs:
        tau = (expiry - valuation_date).days / 365.0
        strike = 100.0
        price = black76_price(100.0, strike, tau, rate, volatility, "CALL")
        chain_rows.append(
            {
                "date": valuation_date,
                "root": root,
                "expiry": expiry,
                "expiry_date": expiry,
                "strike": strike,
                "right": "CALL",
                "option_type": "CALL",
                "bid": price - 0.01,
                "mid": price,
                "ask": price + 0.01,
                "volume": 1,
                "created": valuation_date.tz_localize("America/New_York")
                + pd.Timedelta(hours=17),
            }
        )
        expiry_rows.append(
            {
                "date": valuation_date,
                "root": root,
                "expiry_date": expiry,
                "forward": 100.0,
                "rate": rate,
            }
        )
        expected_ivs.append(volatility)

    enriched = enrich_option_chain(
        pd.DataFrame(chain_rows, index=[2, 5, 9]),
        pd.DataFrame(expiry_rows),
    )

    assert np.allclose(enriched["iv_mid"], expected_ivs, rtol=0, atol=1e-8)
    assert np.isfinite(enriched["total_variance"]).all()
    assert enriched["created"].notna().all()
    assert enriched["created"].equals(pd.Series(
        [
            row["created"]
            for row in chain_rows
        ],
        name="created",
    ))


if __name__ == "__main__":
    test_adapter_computes_bid_mid_ask_iv()
    test_vectorized_iv_matches_scalar_call_and_put_solver()
    test_adapter_preserves_snapshot_timestamp_and_finite_iv_across_expiries()
    print("SPX IV adapter: PASS")