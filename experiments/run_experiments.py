"""Reproduce every figure and number of the report.

Usage:  python experiments/run_experiments.py   (about one minute on a laptop)
Outputs: report/figures/*.pdf and experiments/results.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bsph.analytic import bs_price, bs_delta, bs_vega            # noqa: E402
from bsph.montecarlo import mc_plain, mc_antithetic, mc_control_variate  # noqa: E402
from bsph.pde import pde_price, pde_gamma                                    # noqa: E402
from bsph.greeks import delta_pathwise, delta_likelihood_ratio, delta_bump  # noqa: E402
from bsph.hedging import hedge_pnl                                # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.figsize": (6.4, 4.0), "axes.grid": True, "grid.alpha": 0.3})
S0, T, r, sigma = 100.0, 1.0, 0.03, 0.20
results = {}


def timed(fn, *a, repeat=3, **k):
    best, out = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        best = min(best, time.perf_counter() - t0)
    return out, best


# ---------------------------------------------------------------- 1. Monte Carlo
print("1. Monte Carlo variance reduction")
estimators = {"plain": mc_plain, "antithetic": mc_antithetic, "control variate": mc_control_variate}
strikes = [80.0, 100.0, 120.0]
vr = {}
for K in strikes:
    ref = float(bs_price(S0, K, T, r, sigma))
    row = {"closed_form": ref}
    base = None
    for name, est in estimators.items():
        res = est(S0, K, T, r, sigma, 1_000_000, np.random.default_rng(10))
        var = res.stderr**2 * res.n_paths  # variance per payoff evaluation
        base = var if name == "plain" else base
        row[name] = {"price": res.price, "stderr": res.stderr,
                     "z": (res.price - ref) / res.stderr, "variance_ratio": base / var}
    vr[str(K)] = row
results["mc_strikes"] = vr

ns = np.unique(np.logspace(3, 6.3, 10).astype(int))
K = 100.0
ref = float(bs_price(S0, K, T, r, sigma))
fig, ax = plt.subplots()
conv = {}
for name, est in estimators.items():
    rmse, secs = [], []
    for n in ns:
        errs, t_tot = [], 0.0
        for seed in range(30):
            (res, dt) = timed(est, S0, K, T, r, sigma, int(n), np.random.default_rng(seed), repeat=1)
            errs.append(res.price - ref)
            t_tot += dt
        rmse.append(float(np.sqrt(np.mean(np.square(errs)))))
        secs.append(t_tot / 30)
    conv[name] = {"n": ns.tolist(), "rmse": rmse, "seconds": secs}
    ax.loglog(secs, rmse, "o-", label=name)
results["mc_convergence"] = conv

# ---------------------------------------------------------------- 2. PDE
print("2. Finite differences")
pde = {}
for ran in (False, True):
    errs, secs = [], []
    for m in [25, 50, 100, 200, 400, 800]:
        price, dt = timed(pde_price, S0, K, T, r, sigma, n_space=2 * m, n_time=m, rannacher=ran)
        errs.append(abs(price - ref))
        secs.append(dt)
    pde["rannacher" if ran else "crank_nicolson"] = {"n_time": [25, 50, 100, 200, 400, 800],
                                                     "abs_error": errs, "seconds": secs}
    ax.loglog(secs, errs, "s--", label="PDE, " + ("Rannacher + CN" if ran else "plain CN"))
results["pde"] = pde
# empirical orders of convergence (slope of log error vs log n_time on the finest grids)
for k, v in pde.items():
    e = np.array(v["abs_error"])
    v["order"] = float(-np.polyfit(np.log(v["n_time"][2:]), np.log(e[2:]), 1)[0])
ax.set_xlabel("CPU time (s)")
ax.set_ylabel("absolute pricing error")
ax.set_title("Accuracy vs cost, ATM call (K = 100)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "accuracy_vs_cost.pdf")
plt.close(fig)

# Gamma near the strike: large time step, short maturity -> CN oscillates
from bsph.analytic import bs_gamma  # noqa: E402
Tg, Kg = 0.1, 100.0
fig, ax = plt.subplots()
gam = {}
for ran, style in ((False, "-"), (True, "-")):
    Sg, g = pde_gamma(S0, Kg, Tg, r, sigma, n_space=1000, n_time=10, rannacher=ran, width=6)
    mask = (Sg > 85) & (Sg < 115)
    exact = bs_gamma(Sg[mask], Kg, Tg, r, sigma)
    gam["rannacher" if ran else "crank_nicolson"] = float(np.max(np.abs(g[mask] - exact)))
    ax.plot(Sg[mask], g[mask], style, lw=1.2, label="Rannacher + CN" if ran else "plain Crank-Nicolson")
ax.plot(Sg[mask], exact, "k--", lw=1, label="closed form")
ax.set_xlabel("spot S")
ax.set_ylabel(r"$\Gamma$")
ax.set_ylim(-0.15, 0.3)  # plain CN spikes to ~10 at the strike; clipped for readability
ax.set_title("Gamma from the PDE grid (T = 0.1, 10 time steps)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "pde_gamma.pdf")
plt.close(fig)
results["pde_gamma_max_abs_error"] = gam
print("gamma errors", gam)

# ---------------------------------------------------------------- 3. Delta estimators
print("3. Delta estimators")
dstats = {}
for K in [80.0, 100.0, 120.0]:
    true = float(bs_delta(S0, K, T, r, sigma))
    row = {"closed_form": true}
    for name, f in {"pathwise": delta_pathwise, "likelihood_ratio": delta_likelihood_ratio,
                    "bump_crn": delta_bump}.items():
        est, se = f(S0, K, T, r, sigma, 500_000, np.random.default_rng(20))
        row[name] = {"estimate": est, "stderr": se, "z": (est - true) / se}
    dstats[str(K)] = row
results["delta"] = dstats

# ---------------------------------------------------------------- 4. Hedging frequency
print("4. Discrete hedging")
K, mu, n_paths = 100.0, 0.08, 50_000
steps = [4, 13, 26, 52, 104, 252, 504]
stds, means = [], []
for n in steps:
    pnl = hedge_pnl(S0, K, T, r, mu, sigma, sigma, sigma, n, n_paths, np.random.default_rng(30))
    stds.append(float(pnl.std(ddof=1)))
    means.append(float(pnl.mean()))
slope = float(np.polyfit(np.log(steps), np.log(stds), 1)[0])
vega = float(bs_vega(S0, K, T, r, sigma))
theory = [np.sqrt(np.pi / 4) * vega * sigma / np.sqrt(n) for n in steps]  # Derman & Kamal approximation
results["hedging_frequency"] = {"n_steps": steps, "std": stds, "mean": means,
                                "loglog_slope": slope, "theory_std": theory,
                                "premium": float(bs_price(S0, K, T, r, sigma))}
fig, ax = plt.subplots()
ax.loglog(steps, stds, "o-", label="simulated std of hedging P&L")
ax.loglog(steps, theory, "k--", label=r"$\sqrt{\pi/4}\,\sigma\,\mathcal{V}/\sqrt{N}$")
ax.set_xlabel("number of rebalancing dates N")
ax.set_ylabel("std of discounted P&L")
ax.set_title(f"Hedging error vs rebalancing frequency (slope {slope:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "hedging_frequency.pdf")
plt.close(fig)

# ---------------------------------------------------------------- 5. Volatility misspecification
print("5. Volatility misspecification")
sig_imp, sig_real, n = 0.20, 0.25, 252
cases = {
    "hedge at implied vol": sig_imp,
    "hedge at realised vol": sig_real,
}
mis = {}
fig, ax = plt.subplots()
for label, sh in cases.items():
    pnl = hedge_pnl(S0, K, T, r, mu, sig_real, sig_imp, sh, n, n_paths, np.random.default_rng(40))
    mis[label] = {"mean": float(pnl.mean()), "std": float(pnl.std(ddof=1)),
                  "q05": float(np.quantile(pnl, 0.05)), "q95": float(np.quantile(pnl, 0.95))}
    ax.hist(pnl, bins=120, alpha=0.55, density=True, label=label)
mis["theory_mean_loss"] = float(bs_price(S0, K, T, r, sig_imp) - bs_price(S0, K, T, r, sig_real))
results["misspecification"] = mis
ax.axvline(mis["theory_mean_loss"], color="k", ls="--", lw=1, label="price difference")
ax.set_xlabel("discounted P&L of the short call")
ax.set_title(r"Sold at $\sigma_{imp}=20\%$, realised $\sigma=25\%$, daily hedge")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "vol_misspecification.pdf")
plt.close(fig)

(ROOT / "experiments" / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: results[k] for k in ["pde", "hedging_frequency", "misspecification"]}, indent=1)[:3000])
print(json.dumps(results["mc_strikes"], indent=1))
print(json.dumps(results["delta"], indent=1))
