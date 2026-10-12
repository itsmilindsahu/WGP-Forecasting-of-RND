from __future__ import annotations

import numpy as np

from breeden_litzenberger import bs_call_price, project_call_prices


def test_projection_enforces_call_no_arbitrage():
    forward, tau, rate = 100.0, 30 / 365, 0.03
    strikes = np.linspace(70.0, 130.0, 81)
    prices = bs_call_price(
        forward,
        strikes,
        tau,
        rate,
        np.full_like(strikes, 0.2),
    )
    prices[34] -= 2.0

    projected, projection_rmse, pre_projection_convex = project_call_prices(
        strikes,
        prices,
        forward,
        tau,
        rate,
    )

    discount = np.exp(-rate * tau)
    slopes = np.diff(projected) / np.diff(strikes)
    assert not pre_projection_convex
    assert projection_rmse > 0
    assert np.all(projected >= discount * np.maximum(forward - strikes, 0) - 1e-8)
    assert np.all(projected <= discount * forward + 1e-8)
    assert np.all(slopes >= -discount - 1e-8)
    assert np.all(slopes <= 1e-8)
    assert np.all(np.diff(slopes) >= -1e-8)


if __name__ == "__main__":
    test_projection_enforces_call_no_arbitrage()
    print("No-arbitrage call projection: PASS")