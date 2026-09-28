# Pricing and hedging European options

Three ways to price the same Black–Scholes call (closed form, Monte Carlo with variance reduction, Crank–Nicolson PDE), a comparison of their accuracy against their cost, Monte Carlo estimators of delta, and a simulation study of discrete delta hedging.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Main results

| Question | Result |
|---|---|
| Variance reduction of the control variate (10⁶ paths) | ×49.5 for K = 80, ×5.9 at the money, ×2.1 for K = 120 |
| PDE accuracy | error 3.6 × 10⁻⁵ in 69 ms, measured order of convergence 2.00 |
| Gamma near the strike (T = 0.1, 10 time steps) | max error 10.7 with plain Crank–Nicolson, 1.6 × 10⁻⁴ with Rannacher start-up |
| Hedging error vs number of rebalancing dates N | std ∝ N^−0.489, within 8 % of the Derman–Kamal approximation |
| Call sold at 20 % vol, market realises 25 % | same mean loss (−1.93) with either hedge; hedging at realised vol halves the std (0.53 vs 0.98) |

## Layout

```
src/bsph/
  analytic.py     closed-form prices and Greeks
  montecarlo.py   plain, antithetic and control-variate estimators with standard errors
  pde.py          Crank–Nicolson solver in log-price, optional Rannacher smoothing, grid gamma
  greeks.py       pathwise, likelihood-ratio and bump-and-revalue deltas
  hedging.py      discrete delta hedging of a short call
experiments/run_experiments.py   produces every figure and number of the report
tests/                           each numerical method checked against its closed form
report/                          LaTeX source, figures and PDF
```

## Reproduce

```bash
pip install -e ".[test]"
pytest -q
python experiments/run_experiments.py      # ~1 min, writes report/figures and experiments/results.json
cd report && pdflatex report.tex && pdflatex report.tex
```

All random numbers come from seeded `numpy.random.Generator` objects, so the numbers in the report are reproducible.
