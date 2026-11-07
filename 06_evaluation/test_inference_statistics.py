from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference_robustness import (  # noqa: E402
    _hac_mean_variance,
    _moving_block_bootstrap_ci,
)


class InferenceStatisticsTests(unittest.TestCase):
    def test_hac_white_noise_zero_lags_matches_sample_variance_of_mean(self):
        values = np.array([1.0, 2.0, 4.0, 8.0])
        expected = values.var(ddof=0) / len(values)
        self.assertTrue(np.isclose(_hac_mean_variance(values, 0), expected))

    def test_moving_block_bootstrap_is_reproducible_and_paired(self):
        persistence = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        model = persistence * 0.8
        first = _moving_block_bootstrap_ci(
            persistence,
            model,
            block_length=2,
            rng=np.random.default_rng(42),
        )
        second = _moving_block_bootstrap_ci(
            persistence,
            model,
            block_length=2,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(first, second)
        self.assertTrue(np.allclose(first, (0.2, 0.2)))

if __name__ == "__main__":
    unittest.main()
