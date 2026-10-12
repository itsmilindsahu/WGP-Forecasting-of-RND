from __future__ import annotations

import numpy as np

from distribution_forecasters import (
    MomentGaussianProcess,
    WassersteinAutoregression,
    quantile_moments,
)


def test_distribution_forecasters_return_monotone_quantiles():
    rng = np.random.default_rng(17)
    inputs = np.sort(rng.normal(1.0, 0.08, size=(15, 40)), axis=1)
    targets = np.sort(0.9 * inputs + 0.1 + rng.normal(0, 0.004, size=inputs.shape), axis=1)

    ar = WassersteinAutoregression().fit(inputs[:12], targets[:12])
    ar_prediction = ar.predict(inputs[12:])

    moments = quantile_moments(inputs)
    gp = MomentGaussianProcess().fit(moments[:12], targets[:12])
    gp_prediction = gp.predict(moments[12:])

    assert ar_prediction.shape == (3, 40)
    assert gp_prediction.shape == (3, 40)
    assert np.isfinite(ar_prediction).all() and np.isfinite(gp_prediction).all()
    assert np.all(np.diff(ar_prediction, axis=1) >= 0)
    assert np.all(np.diff(gp_prediction, axis=1) >= 0)


if __name__ == "__main__":
    test_distribution_forecasters_return_monotone_quantiles()
    print("Distribution forecasters: PASS")