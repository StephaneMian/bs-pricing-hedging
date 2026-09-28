"""Discrete delta hedging of a short European call.

The seller receives the Black-Scholes premium computed with `sigma_implied`,
then holds delta(sigma_hedge) shares, rebalanced at n_steps equally spaced
dates. The stock follows GBM with drift `mu` and volatility `sigma_true`;
cash accrues at r. Returns the discounted terminal P&L of each path.
"""
import numpy as np
from .analytic import bs_price, bs_delta


def hedge_pnl(S0, K, T, r, mu, sigma_true, sigma_implied, sigma_hedge,
              n_steps, n_paths, rng):
    dt = T / n_steps
    S = np.full(n_paths, float(S0))
    delta = bs_delta(S, K, T, r, sigma_hedge)
    cash = bs_price(S0, K, T, r, sigma_implied) - delta * S
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * z)
        cash = cash * np.exp(r * dt)
        if k < n_steps:
            new_delta = bs_delta(S, K, T - k * dt, r, sigma_hedge)
            cash -= (new_delta - delta) * S
            delta = new_delta
    value = cash + delta * S - np.maximum(S - K, 0.0)
    return np.exp(-r * T) * value
