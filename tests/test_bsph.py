import numpy as np
import pytest

from bsph.analytic import bs_price, bs_delta, bs_gamma, bs_vega
from bsph.montecarlo import mc_plain, mc_antithetic, mc_control_variate
from bsph.pde import pde_price
from bsph.greeks import delta_pathwise, delta_likelihood_ratio, delta_bump
from bsph.hedging import hedge_pnl

P = dict(S0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2)
A = (100.0, 100.0, 1.0, 0.03, 0.2)  # same parameters, positional, for closed forms


def test_reference_value():
    # Hull-style textbook check: S=K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert bs_price(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    c = bs_price(100, 90, 0.5, 0.02, 0.3, "call")
    p = bs_price(100, 90, 0.5, 0.02, 0.3, "put")
    assert c - p == pytest.approx(100 - 90 * np.exp(-0.02 * 0.5), abs=1e-12)


def test_greeks_match_finite_differences():
    h = 1e-4
    f = lambda s: bs_price(s, 100, 1, 0.03, 0.2)
    assert bs_delta(100, 100, 1, 0.03, 0.2) == pytest.approx((f(100 + h) - f(100 - h)) / (2 * h), abs=1e-6)
    assert bs_gamma(100, 100, 1, 0.03, 0.2) == pytest.approx((f(100 + h) - 2 * f(100) + f(100 - h)) / h**2, abs=1e-4)
    g = lambda v: bs_price(100, 100, 1, 0.03, v)
    assert bs_vega(100, 100, 1, 0.03, 0.2) == pytest.approx((g(0.2 + h) - g(0.2 - h)) / (2 * h), abs=1e-5)


@pytest.mark.parametrize("estimator", [mc_plain, mc_antithetic, mc_control_variate])
def test_monte_carlo_within_four_standard_errors(estimator):
    res = estimator(**P, n=200_000, rng=np.random.default_rng(1))
    assert abs(res.price - bs_price(*A)) < 4 * res.stderr


def test_variance_reduction_reduces_error():
    rng = np.random.default_rng(2)
    plain = mc_plain(**P, n=100_000, rng=rng).stderr
    assert mc_antithetic(**P, n=100_000, rng=rng).stderr < plain
    assert mc_control_variate(**P, n=100_000, rng=rng).stderr < plain


@pytest.mark.parametrize("kind", ["call", "put"])
def test_pde_matches_closed_form(kind):
    assert pde_price(**P, kind=kind) == pytest.approx(bs_price(*A, kind=kind), abs=5e-3)


@pytest.mark.parametrize("estimator", [delta_pathwise, delta_likelihood_ratio, delta_bump])
def test_delta_estimators(estimator):
    est, se = estimator(**P, n=400_000, rng=np.random.default_rng(3))
    assert abs(est - bs_delta(*A)) < 4 * se + 1e-4


def test_hedging_error_shrinks_with_rebalancing():
    kw = dict(S0=100, K=100, T=1, r=0.03, mu=0.08, sigma_true=0.2,
              sigma_implied=0.2, sigma_hedge=0.2, n_paths=20_000)
    coarse = hedge_pnl(**kw, n_steps=13, rng=np.random.default_rng(4)).std()
    fine = hedge_pnl(**kw, n_steps=208, rng=np.random.default_rng(4)).std()
    assert fine < coarse / 3  # theory predicts a factor 4 = sqrt(208/13)
