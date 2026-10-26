# Retail Sales Forecasting: Model Comparison and Backtest

Finds the most accurate, cost-effective model for forecasting daily store sales six weeks ahead, validated with a 6-window rolling backtest instead of one train/test split.

![MAPE per model, per fold](docs/images/mape_by_fold.png)

**Problem.** Retailers plan staffing and stock roughly six weeks ahead. Pick the wrong method and you over- or understaff — and it isn't obvious upfront which approach (classical statistics, Prophet, gradient-boosted trees) actually holds up over time, not just once.

**Approach.** Backtested five methods across six rolling 42-day windows, each retrained on all prior data — not one split. Fed every model the **full calendar, including closed days**, plus known-in-advance `Promo`/`SchoolHoliday`/`StateHoliday` regressors — this alone flipped AutoETS from losing every window to beating every window. Forecast XGBoost **recursively**: lag features come only from its own prior predictions, never a fold's real future sales.

**Result.**

| Model | Mean MAPE | Beats baseline | Mean fit time/store |
|---|---|---|---|
| XGBoost | 11.14% | 6 of 6 windows | 1.8s |
| Prophet | 12.04% | 6 of 6 windows | 0.4s |
| AutoARIMA | 17.96% | 6 of 6 windows | 37.9s |
| Seasonal Naive (baseline) | 25.25% | — | ~0s |

XGBoost wins on accuracy; Prophet is the cost-conscious alternative, ~4x cheaper to retrain. Full result, uncertainty check and recommendation: [Findings](#findings).

## Findings

**Accuracy.** Every model beats the seasonal-naive baseline in all 6 of 6 windows, including AutoETS — which lost to the baseline every window before the full-calendar + exogenous-feature fix above. WAPE mostly confirms MAPE, except the Christmas/New Year window, where it's worse (mis-sized peak, not noisy low-value days). **XGBoost wins** (11.14 MAPE); **Prophet is the cheaper runner-up** (12.04 MAPE, ~4x faster to fit); **AutoARIMA isn't worth its cost** (17.96 MAPE, barely beats AutoETS's 18.02, for ~100x more fit time).

**Uncertainty.** A point forecast alone doesn't say how much safety stock or staffing buffer to hold for the busy end of the range, so each model's 80%/95% prediction intervals were scored the same way — open days, all 6 folds — for coverage, width, and Winkler (interval) score:

| Model | 80% coverage | 95% coverage | Mean width (80%) | Winkler score (80%) |
|---|---|---|---|---|
| Prophet | 72% | 90% | 1,536 | 2,887 |
| XGBoost | 77% | 87% | 1,871 | 3,745 |
| AutoETS | 84% | 92% | 3,136 | 4,740 |
| AutoARIMA | 67% | 80% | 1,980 | 4,836 |
| Seasonal Naive | 94% | 100% | 7,531 | 7,987 |

**How to read it:** coverage near the target means the band can be trusted; well below means the model is overconfident (band too narrow); well above means it's wider than it needs to be. For safety stock or staffing, size against the **upper bound** — understating it matters in practice. Example: XGBoost's 95% interval should miss the actual value only 1 day in 20; in the backtest it missed roughly 1 day in 8 (87% vs. 95% coverage) — stock sized to its 95% upper bound would run short far more often than planned.

**AutoARIMA is also the worst-calibrated model** — 13–15pp undercoverage at both levels, on top of being slowest and least accurate; its parametric (Gaussian) interval assumes steadier residual behaviour than promo/holiday-driven demand swings actually produce. **AutoETS is the best-calibrated.** XGBoost's conformal interval (calibrated on prior folds' own backtest errors, bucketed by horizon week) holds at 80% but undercovers at 95%, where its calibration pool is thin. **Prophet wins on Winkler score outright** — but that's tight bands outscoring a real coverage gap, not evidence of good calibration.

**Recommendation:** use **XGBoost** as the default — it wins on accuracy and full-window consistency. For a 95% safety-stock buffer specifically, trust **AutoETS**'s interval over XGBoost's own. Skip **AutoARIMA** — it loses on cost, accuracy, and calibration alike.

Full per-fold breakdown, interval methodology, and the fair-testing protocol (no leakage, fixed seeds, matched train/test windows across all five models): see [`single_store_forecast.ipynb`](single_store_forecast.ipynb), section 5.

## Limitations & next steps

- **One store.** This backtest alone may not generalise across store types and locations. Next: extend the comparison to all 1,115 stores, weighing XGBoost's accuracy against its retraining cost at scale.
- **Six test windows** is enough to show every model now beats the baseline consistently, not from one lucky window — but still too few folds for a rigorous confidence interval.
- **Promo/holiday schedules are assumed fully known at forecast time.** True in this backtest (Rossmann's historical flags); in a real deployment, promotions scheduled weeks ahead can still change.
- **Prophet can predict slightly negative sales on closed days** (additive regression, no floor at 0) — harmless here since closed days are excluded from scoring, but would need clamping to 0 if used directly.
- **Training-time figures are single-machine, single-run wall-clock measurements**, a directional signal rather than a precise benchmark.

## Run it

```bash
git clone https://github.com/Ampofowaa/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting
pip install -r requirements.txt
# Download train.csv, test.csv, store.csv from Kaggle into data/rossmann/ (not redistributed here)
jupyter nbconvert --to notebook --execute single_store_forecast.ipynb --output single_store_forecast.ipynb
```

Prophet's `cmdstanpy` backend needs a one-time compiled-model build on first install, which `pip install` alone doesn't trigger — see [cmdstanpy's install docs](https://mc-stan.org/cmdstanpy/installation.html) if `import prophet` fails. Takes roughly 6–8 minutes end to end, dominated by AutoARIMA's order search across 6 folds. Saved model files (`.pkl`) are excluded from version control; rerun the notebook to regenerate them.

## Project structure

```
.
├── single_store_forecast.ipynb   # main analysis: model comparison & backtest
├── docs/images/                  # chart exported from the notebook, used above
├── requirements.txt
├── LICENSE
└── data/rossmann/                # Kaggle data, not included — see Run it
```

## Tech stack

Python · pandas · `statsforecast` (AutoARIMA, AutoETS) · Prophet · XGBoost · `utilsforecast` · matplotlib · Jupyter

## Data

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle): daily sales for 1,115 German drugstores. This comparison uses Store 1's full calendar — 942 days (1 Jan 2013 – 31 Jul 2015), including the 161 closed days — scored on the 781 open days per window, since MAPE is undefined at zero sales (the same convention as Rossmann's own RMSPE metric). Raw data files aren't redistributed here per Kaggle's terms — see Run it. Code in this repo is MIT-licensed (`LICENSE`); the dataset itself is third-party and not covered by it.

[GitHub](https://github.com/Ampofowaa) · [LinkedIn](https://www.linkedin.com/in/richlove-frimpong)
