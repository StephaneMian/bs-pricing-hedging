"""Monte Carlo estimators of the delta of a European call.

* pathwise: differentiate the discounted payoff along each path
  (valid because the call payoff is Lipschitz in S0),
* likelihood ratio: weight the payoff by the score of the terminal density,
* bump-and-revalue with common random numbers (central difference).
Each function returns (estimate, standard error).
"""
import numpy as np
from .montecarlo import _terminal


def delta_pathwise(S0, K, T, r, sigma, n, rng):
    z = rng.standard_normal(n)
    ST = _terminal(S0, T, r, sigma, z)
    x = np.exp(-r * T) * (ST > K) * ST / S0
    return x.mean(), x.std(ddof=1) / np.sqrt(n)


def delta_likelihood_ratio(S0, K, T, r, sigma, n, rng):
    z = rng.standard_normal(n)
    ST = _terminal(S0, T, r, sigma, z)
    x = np.exp(-r * T) * np.maximum(ST - K, 0) * z / (S0 * sigma * np.sqrt(T))
    return x.mean(), x.std(ddof=1) / np.sqrt(n)


def delta_bump(S0, K, T, r, sigma, n, rng, h=1e-2):
    z = rng.standard_normal(n)
    disc = np.exp(-r * T)
    up = disc * np.maximum(_terminal(S0 * (1 + h), T, r, sigma, z) - K, 0)
    dn = disc * np.maximum(_terminal(S0 * (1 - h), T, r, sigma, z) - K, 0)
    x = (up - dn) / (2 * h * S0)
    return x.mean(), x.std(ddof=1) / np.sqrt(n)
