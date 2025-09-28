# Impact of Copula Choice (VaR / TVaR & Exceedance Rates)

## Why Copula Choice Matters

In multivariate risk modeling, copulas allow us to represent the dependence structure between random variables (e.g. demand, lead time, cost shocks) separately from marginal distributions. The choice of copula is critical because it affects joint tail behavior — in other words, how extreme events in one dimension may co-occur with extremes in another. This, in turn, strongly influences risk measures such as Value at Risk (VaR), Tail Value at Risk (TVaR), and observed exceedance rates.

## How It Strengthens Our Project

- Better tail risk estimation: By comparing multiple candidate copulas, we determined which dependence structure best captures extreme joint behavior. This yields more realistic VaR/TVaR estimates under stress scenarios.
- Model validation via exceedance calibration: We backtested exceedance frequencies (i.e., how often losses actually exceed the VaR threshold) under different copulas. The copula whose simulated exceedance rates align closest with empirical data was preferred.
- Sensitivity analysis & robustness: We benchmarked how much VaR or TVaR changes when one swaps copulas (keeping marginals fixed). This helps us assess model risk due to copula misspecification.
- Scenario stress testing across dependence regimes: We used “worst-case” copulas (with high tail dependence) to stress the system, showing that under certain dependence structures, losses could be significantly worse than under more benign structures. This provides insights into resilience under extreme correlation shocks.


## Where in the Project We Applied It

- In the risk simulation notebook(s), we implemented multiple copulas (e.g. Gaussian, t) to model joint residuals / shocks among dimensions (demand, delay, cost).
- For each copula candidate, we computed VaR and TVaR at several quantile levels (e.g. 95 %, 99 %), using Monte Carlo simulation of the joint model.
- We also simulated exceedance rates over a hold-out period (i.e. count how many simulated losses exceed the VaR) and compared these to empirical exceedances to calibrate the copula fit.
- We documented and plotted how VaR and TVaR shift across copula choices (e.g. a plot of VaR vs copula type) and summarized differences in exceedance calibration.
- In stress / scenario analyses, we used copulas with stronger tail dependence to stress test supply chain risk under more adverse dependence.
