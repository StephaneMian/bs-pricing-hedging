"""Monte Carlo pricing of European options under Black-Scholes.

Three estimators of the same price, each returned with its standard error:
  * plain Monte Carlo,
  * antithetic variates (Z and -Z),
  * control variate using the discounted terminal price, whose expectation S0
    is known exactly. The coefficient is estimated on the same sample, which
    adds an O(1/n) bias that is negligible at the sample sizes used here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class MCResult:
    price: float
    stderr: float
    n_paths: int


def _terminal(S0, T, r, sigma, z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def _payoff(ST, K, kind):
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def mc_plain(S0, K, T, r, sigma, n, rng, kind="call"):
    z = rng.standard_normal(n)
    x = np.exp(-r * T) * _payoff(_terminal(S0, T, r, sigma, z), K, kind)
    return MCResult(x.mean(), x.std(ddof=1) / np.sqrt(n), n)


def mc_antithetic(S0, K, T, r, sigma, n, rng, kind="call"):
    """n is the total number of payoff evaluations (n/2 antithetic pairs)."""
    m = n // 2
    z = rng.standard_normal(m)
    disc = np.exp(-r * T)
    a = disc * _payoff(_terminal(S0, T, r, sigma, z), K, kind)
    b = disc * _payoff(_terminal(S0, T, r, sigma, -z), K, kind)
    pair = 0.5 * (a + b)  # pair averages are i.i.d.: the correct unit for the error
    return MCResult(pair.mean(), pair.std(ddof=1) / np.sqrt(m), 2 * m)


def mc_control_variate(S0, K, T, r, sigma, n, rng, kind="call"):
    z = rng.standard_normal(n)
    ST = _terminal(S0, T, r, sigma, z)
    disc = np.exp(-r * T)
    x = disc * _payoff(ST, K, kind)
    y = disc * ST  # E[y] = S0 exactly under the risk-neutral measure
    c = np.cov(x, y)
    beta = c[0, 1] / c[1, 1]
    adj = x - beta * (y - S0)
    return MCResult(adj.mean(), adj.std(ddof=1) / np.sqrt(n), n)
