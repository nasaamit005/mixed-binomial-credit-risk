import numpy as np
import pandas as pd
import pytest

from mbcr.calibration import (
    fit_bayes,
    fit_mle,
    log_likelihood,
    rates_to_pseudo_counts,
    simulate_default_counts,
)
from mbcr.capital import capital_gap, irb_capital_requirement, irb_correlation, irb_risk_weight


def test_mle_recovers_truth_on_long_synthetic_series():
    n = np.full(300, 2000)
    d = simulate_default_counts(n, 0.02, 0.15, seed=42)
    fit = fit_mle(d, n)
    assert fit["success"]
    assert fit["pd"] == pytest.approx(0.02, rel=0.2)
    assert fit["rho"] == pytest.approx(0.15, abs=0.05)


def test_likelihood_peaks_near_truth():
    n = np.full(200, 1500)
    d = simulate_default_counts(n, 0.03, 0.10, seed=1)
    assert log_likelihood(0.03, 0.10, d, n) > log_likelihood(0.03, 0.40, d, n)
    assert log_likelihood(0.03, 0.10, d, n) > log_likelihood(0.15, 0.10, d, n)


def test_bayes_posterior_covers_truth_and_mixes():
    n = np.full(120, 1500)
    d = simulate_default_counts(n, 0.02, 0.12, seed=8)
    post = fit_bayes(d, n, n_iter=3000, burn=800, seed=3)
    assert 0.1 < post["acceptance"] < 0.8
    lo, hi = np.percentile(post["pd"], [1, 99])
    assert lo < 0.02 < hi


def test_rates_to_pseudo_counts():
    idx = pd.date_range("2000-01-01", periods=8, freq="QS")
    s = pd.Series([1.0] * 4 + [2.0] * 4, index=idx)
    d, _n, years = rates_to_pseudo_counts(s, n_eff=1000)
    assert list(d) == [10, 20] and list(years) == [2000, 2001]


def test_irb_sanity():
    # PD 1%, LGD 45%, M=2.5: capital requirement should be ~7-8% (risk weight ~90-95%)
    k = irb_capital_requirement(0.01, 0.45)
    assert 0.06 < k < 0.09
    assert irb_risk_weight(0.01, 0.45) == pytest.approx(12.5 * k)
    assert irb_correlation(1e-6) == pytest.approx(0.24, abs=1e-3)
    assert irb_correlation(0.5) == pytest.approx(0.12, abs=1e-3)


def test_capital_gap_positive_and_shrinking():
    g1 = capital_gap(100, 0.02, 0.15)
    g2 = capital_gap(1000, 0.02, 0.15)
    assert g1["rel_gap"] > g2["rel_gap"] > 0
    assert capital_gap(100, 0.02, 0.15, use_correction=True)["K_n"] == pytest.approx(g1["K_n"], rel=0.01)
