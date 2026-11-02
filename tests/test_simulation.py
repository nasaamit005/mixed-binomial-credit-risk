import numpy as np
import pytest

from mbcr import Portfolio, mixed_binomial_pmf, var_es_discrete
from mbcr.exact import mixed_binomial_pmf_t, var_es_continuous
from mbcr.simulation import default_is_shift, estimate_risk, simulate_losses
from mbcr.vasicek import var_limit_t, var_vasicek


def test_seed_reproducibility():
    pf = Portfolio.homogeneous(100, 0.02, 0.15)
    a, _ = simulate_losses(pf, 5000, seed=1)
    b, _ = simulate_losses(pf, 5000, seed=1)
    c, _ = simulate_losses(pf, 5000, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_plain_mc_matches_exact_within_se():
    pf = Portfolio.homogeneous(100, 0.02, 0.15)
    L, w = simulate_losses(pf, 1_000_000, seed=7)
    est = estimate_risk(L, w, 0.99)
    v, e = var_es_discrete(mixed_binomial_pmf(100, 0.02, 0.15), 0.99)
    assert abs(est["es"] - e) < 4 * est["es_se"] + 1e-4
    assert abs(est["var"] - v) <= 0.011


def test_importance_sampling_matches_exact_and_reduces_se():
    pf = Portfolio.homogeneous(100, 0.02, 0.15)
    N = 200_000
    _v, e = var_es_discrete(mixed_binomial_pmf(100, 0.02, 0.15), 0.999)
    L0, w0 = simulate_losses(pf, N, seed=3)
    Li, wi = simulate_losses(pf, N, seed=3, is_shift=default_is_shift(0.999))
    e0, ei = estimate_risk(L0, w0, 0.999), estimate_risk(Li, wi, 0.999)
    assert abs(ei["es"] - e) < 4 * ei["es_se"] + 2e-3
    assert ei["es_se"] < e0["es_se"]
    assert ei["ess"] > 5000
    assert ei["es_se"] < e0["es_se"] / 3


def test_weights_have_unit_mean():
    pf = Portfolio.homogeneous(50, 0.02, 0.15)
    _, w = simulate_losses(pf, 400_000, seed=11, is_shift=-2.0)
    assert w.mean() == pytest.approx(1.0, abs=0.05)


def test_t_copula_simulation_matches_exact():
    pf = Portfolio.homogeneous(100, 0.02, 0.15)
    L, w = simulate_losses(pf, 600_000, seed=5, model="t", nu=5)
    est = estimate_risk(L, w, 0.99)
    exact_v, _ = var_es_continuous(mixed_binomial_pmf_t(100, 0.02, 0.15, nu=5), 0.99)
    assert abs(est["var"] - exact_v) < 0.02


def test_t_limit_exceeds_gaussian_limit_in_tail():
    assert var_limit_t(0.02, 0.15, 4, 0.999) > var_vasicek(0.02, 0.15, 0.999)


def test_heterogeneous_expected_loss_and_two_factor():
    pf = Portfolio(n=[60, 40], pd=[0.01, 0.05], rho=[0.12, 0.2], ead=[1.0, 2.0],
                   lgd=[0.45, 0.6], sector_rho=[0.05, 0.1])
    L, w = simulate_losses(pf, 400_000, seed=9)
    assert (L * w).mean() == pytest.approx(pf.expected_loss_fraction(), rel=0.05)
    pf0 = Portfolio(n=[60, 40], pd=[0.01, 0.05], rho=[0.12, 0.2], ead=[1.0, 2.0], lgd=[0.45, 0.6])
    L0, w0 = simulate_losses(pf0, 400_000, seed=9)
    # sector factor adds extra (group-level) dependence => heavier tail at same marginal PD/EL
    assert estimate_risk(L, w, 0.999)["var"] >= estimate_risk(L0, w0, 0.999)["var"] - 0.01


def test_stochastic_lgd_same_mean_more_tail_risk():
    pf = Portfolio.homogeneous(200, 0.03, 0.15, lgd=0.45)
    Lf, _wf = simulate_losses(pf, 300_000, seed=4, lgd_model="fixed")
    Lb, _wb = simulate_losses(pf, 300_000, seed=4, lgd_model="beta", lgd_sd=0.3)
    assert Lb.mean() == pytest.approx(Lf.mean(), rel=0.05)
    assert Lb.var() > Lf.var()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Portfolio.homogeneous(10, 0.02, 1.2)
    with pytest.raises(ValueError):
        simulate_losses(Portfolio.homogeneous(10, 0.02, 0.1), 10, model="bad")
