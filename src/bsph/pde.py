"""Finite-difference solution of the Black-Scholes PDE in log-price.

With x = log S and tau = T - t, the price u(x, tau) solves
    u_tau = 0.5 sigma^2 u_xx + (r - 0.5 sigma^2) u_x - r u.
Crank-Nicolson in time. Optionally, the first two time steps are replaced by
four fully implicit half-steps (Rannacher start-up) to damp the oscillations
caused by the kink of the payoff at the strike. Dirichlet boundaries use the
asymptotic prices of the option.
"""
import numpy as np
from scipy.linalg import solve_banded


def _coefficients(dx, r, sigma):
    a = 0.5 * sigma**2 / dx**2
    b = (r - 0.5 * sigma**2) / (2 * dx)
    return a - b, -2 * a - r, a + b  # weights of u_{i-1}, u_i, u_{i+1}


def _theta_step(u, dt, theta, coeffs, bc_new):
    lower, diag, upper = coeffs
    m = u.size - 2
    rhs = u[1:-1] + (1 - theta) * dt * (lower * u[:-2] + diag * u[1:-1] + upper * u[2:])
    rhs[0] += theta * dt * lower * bc_new[0]
    rhs[-1] += theta * dt * upper * bc_new[1]
    ab = np.zeros((3, m))
    ab[0, 1:] = -theta * dt * upper
    ab[1, :] = 1 - theta * dt * diag
    ab[2, :-1] = -theta * dt * lower
    out = np.empty_like(u)
    out[1:-1] = solve_banded((1, 1), ab, rhs)
    out[0], out[-1] = bc_new
    return out


def pde_solve(S0, K, T, r, sigma, n_space=400, n_time=200, width=5.0,
              kind="call", rannacher=True):
    """Return the log-price grid x and the option values u(x) at t = 0.

    The grid is centred on log S0 and spans +-width*sigma*sqrt(T)."""
    x0 = np.log(S0)
    half = width * sigma * np.sqrt(T)
    x = np.linspace(x0 - half, x0 + half, n_space + 1)
    S = np.exp(x)
    u = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)

    def boundary(tau):
        disc = K * np.exp(-r * tau)
        return (0.0, S[-1] - disc) if kind == "call" else (disc - S[0], 0.0)

    coeffs = _coefficients(x[1] - x[0], r, sigma)
    dt = T / n_time
    tau, full_steps = 0.0, n_time
    if rannacher:
        for _ in range(4):
            tau += dt / 2
            u = _theta_step(u, dt / 2, 1.0, coeffs, boundary(tau))
        full_steps -= 2
    for _ in range(full_steps):
        tau += dt
        u = _theta_step(u, dt, 0.5, coeffs, boundary(tau))
    return x, u


def pde_price(S0, K, T, r, sigma, **kw):
    x, u = pde_solve(S0, K, T, r, sigma, **kw)
    return float(np.interp(np.log(S0), x, u))


def pde_gamma(S0, K, T, r, sigma, **kw):
    """Gamma on the grid: d2V/dS2 = (u_xx - u_x) / S^2 in log coordinates."""
    x, u = pde_solve(S0, K, T, r, sigma, **kw)
    dx = x[1] - x[0]
    u_x = (u[2:] - u[:-2]) / (2 * dx)
    u_xx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    S = np.exp(x[1:-1])
    return S, (u_xx - u_x) / S**2
