from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_spx_iv import black76_price, enrich_option_chain


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


if __name__ == "__main__":
    test_adapter_computes_bid_mid_ask_iv()
    print("SPX IV adapter: PASS")