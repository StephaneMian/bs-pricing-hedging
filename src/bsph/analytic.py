"""Closed-form Black-Scholes prices and Greeks (no dividends).

All functions are vectorised over NumPy arrays. `kind` is "call" or "put".
"""
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol
    return d1, d1 - vol


def bs_price(S, K, T, r, sigma, kind="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * np.asarray(T, float))
    call = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    if kind == "call":
        return call
    if kind == "put":
        return call - S + K * disc  # put-call parity
    raise ValueError(f"unknown option kind: {kind}")


def bs_delta(S, K, T, r, sigma, kind="call"):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)
