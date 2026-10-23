# Retail Sales Forecasting: Model Comparison and Backtest

Which forecasting method should a retailer trust to plan staffing and stock six weeks ahead? This project compares four common approaches against a simple baseline on daily store sales, tests whether the best-looking result holds up over time, and then works through what changes when the same decision has to scale to 1,115 stores at once.

**What this demonstrates:** rolling-origin backtesting (not a single train/test split), fair multi-model comparison with exogenous/regressor features, leakage-safe recursive forecasting, and model selection that weighs accuracy against compute cost rather than accuracy alone — using `statsforecast`, `Prophet`, `XGBoost`, `pandas`, and `utilsforecast`.

## Key findings

- **Every model now beats the seasonal-naive baseline in all 6 of 6 test windows**, including AutoETS — which lost to the baseline in every window in an earlier version of this comparison. The fix: feeding every model the full calendar (including the ~161 days Store 1 is closed) and known-in-advance `Promo`/`SchoolHoliday`/`IsStateHoliday`/`Open` flags, instead of silently dropping closed days and ignoring promotions and holidays.
- **XGBoost is the most accurate** (11.14 mean MAPE, a 52% cut vs. baseline), with **Prophet a close second** (12.04 mean MAPE, a 47% cut) — and for the first time, both are fully consistent, beating the baseline in every single window rather than XGBoost's edge coming from one lucky window.
- **WAPE (total error ÷ total actual sales) mostly confirms the MAPE story**, which is reassuring — but it also shows MAPE wasn't inflated by low-sales days the way we expected. The hardest window (Christmas/New Year) actually scores *worse* under WAPE for most models, because it weights errors by sales value, and that window's problem is under/over-shooting the size of the December peak, not noisy low-value days.
- **Accuracy alone doesn't settle it: training cost does.** Prophet fits markedly faster than XGBoost per store per retrain. That's irrelevant for one store, but matters once this scales to nightly retraining across all 1,115 stores (see [Scaling to all 1,115 stores](#scaling-to-all-1115-stores-in-progress) below) — and these are single-machine, single-run wall-clock numbers, a directional signal rather than a precise benchmark.
- **AutoARIMA isn't worth its cost.** Its accuracy (17.96 MAPE) is barely better than AutoETS's (18.02), but it takes roughly two orders of magnitude longer to fit.

**Recommendation:** use **XGBoost** as the default model for this store — it now wins on both accuracy and full-window consistency, which a smaller, less complete backtest couldn't establish. Treat **Prophet** as the cost-conscious alternative: nearly as accurate, a tighter worst-case error, and markedly cheaper to retrain — worth it if nightly compute cost across many stores becomes a constraint. Skip AutoARIMA; AutoETS is the better `statsforecast` option if a fast, low-dependency model is ever needed.

## Data

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle): daily sales for 1,115 German drugstores. This comparison uses **Store 1, every calendar day**: 942 days from 1 January 2013 to 31 July 2015, including the 161 days the store is closed (`Sales == 0` on those days by construction). **Scoring is restricted to the 781 open days** within each test window, since MAPE is undefined when actual sales are 0 — the same convention Rossmann's own competition metric (RMSPE) uses.

## Method

**Models compared**

| Model | Approach |
|---|---|
| Seasonal naive (baseline) | Forecast = sales on the same day last week (via `statsforecast`'s `SeasonalNaive`) |
| Prophet | Trend plus weekly and yearly seasonality, plus `Promo`/`SchoolHoliday`/`IsStateHoliday`/`Open` as regressors |
| XGBoost | Gradient-boosted trees on lag (1, 2, 3, 7 days), rolling mean and spread (7 days) and calendar features, plus the same four exogenous flags; forecasts recursively day by day |
| AutoARIMA | Automatically selected ARIMA model (`statsforecast`), season length 7, same four exogenous flags |
| AutoETS | Automatically selected exponential smoothing model (`statsforecast`), season length 7 — no exogenous support, stays univariate |

**Fair testing**

- Every model is trained and scored on exactly the same data and test windows.
- **Rolling-origin backtest:** six consecutive test windows of 42 calendar days (six weeks) each, with each model retrained on all data before its window.
- Scored with MAE, RMSE, MAPE and WAPE, computed on each window's open days only (closed-day sales are always 0, which makes MAPE/WAPE undefined there — the same convention Rossmann's own competition metric, RMSPE, uses).
- **No leakage:** XGBoost builds its lag and rolling features only from past sales and its own earlier forecasts; `Promo`/`SchoolHoliday`/`IsStateHoliday`/`Open` are known ahead of the forecast horizon in a real deployment (schedules and calendars, not forecasts), so passing their real future values isn't leakage.
- Each model's fit time and forecast-generation time are measured separately, since a nightly retrain pipeline treats them as two different operational costs.
- Random seeds are fixed so results can be reproduced.

## Results

**Summary across six windows (lower is better; fit/predict are mean seconds per store per fold, single-machine wall-clock)**

| Model | Mean MAPE | Mean WAPE | Best window (MAPE) | Worst window (MAPE) | Mean change vs baseline | Windows beating baseline | Mean fit time | Mean predict time |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 11.14 | 11.32 | 7.40 | 20.44 | −52.1% | 6 of 6 | 3.2s | 0.7s |
| Prophet | 12.04 | 11.70 | 7.07 | 17.19 | −47.5% | 6 of 6 | 0.6s | 0.1s |
| AutoARIMA | 17.96 | 17.61 | 6.88 | 33.18 | −30.7% | 6 of 6 | 65.2s | 0.05s |
| AutoETS | 18.02 | 18.02 | 12.41 | 30.32 | −22.6% | 6 of 6 | 0.6s | 0.01s |
| Seasonal naive | 25.25 | 25.25 | 14.77 | 45.41 | — | 0 of 6 | ~0s | ~0s |

**By test window (MAPE, %)**

| Window | Seasonal naive | Prophet | XGBoost | AutoARIMA | AutoETS |
|---|---|---|---|---|---|
| 22 Nov 2014 – 2 Jan 2015 | 32.51 | 17.19 | 20.44 | 28.23 | 30.32 |
| 3 Jan – 13 Feb 2015 | 45.41 | 12.79 | 9.43 | 33.18 | 17.60 |
| 14 Feb – 27 Mar 2015 | 14.77 | 12.08 | 8.41 | 13.95 | 14.02 |
| 28 Mar – 8 May 2015 | 24.11 | 12.77 | 10.81 | 14.85 | 18.70 |
| 9 May – 19 Jun 2015 | 18.53 | 10.31 | 10.37 | 10.65 | 12.41 |
| 20 Jun – 31 Jul 2015 | 16.15 | 7.07 | 7.40 | 6.88 | 15.04 |

The first window (22 Nov – 2 Jan, spanning the December peak and the Christmas Eve/New Year's Eve crashes) was hardest for every model. WAPE confirms this isn't a low-sales-day artifact inflating a percentage average — it's models under- or over-shooting the size of the December peak itself, which a sales-weighted metric like WAPE picks up even more than MAPE does.

### When to use each method

| Method | Handles seasonality and holidays | Needs manual features | Data needed | Typical fit time (this backtest) | Typical use |
|---|---|---|---|---|---|
| Prophet | Built in, plus explicit regressors here | No | Low to medium | ~0.6s/store/retrain | Business series with known seasonality and holidays |
| XGBoost | Through engineered features | Yes | Medium | ~3.2s/store/retrain | Structured data with external drivers such as promotions |
| AutoETS (`statsforecast`) | Depends on the model; no exogenous support | No | Low | ~0.6s/store/retrain | Fast baselines, large numbers of series |
| AutoARIMA (`statsforecast`) | Depends on the model | No | Low | ~65s/store/retrain | Not recommended here — order-search cost doesn't pay off vs. AutoETS |

## How this would be used in a business

In production, forecasting runs as a scheduled process rather than a notebook:

1. **Scheduled batch job.** Each night or week, the pipeline pulls the latest sales, refreshes the model and forecasts the next six weeks for every store. Extrapolating this backtest's single-store fit times linearly across all 1,115 stores (a rough estimate, not a measured one) puts XGBoost at roughly an hour of nightly fit time versus Prophet's ~10 minutes — worth validating against the actual multi-store run before it shapes the batch window.
2. **Forecast table.** Results are written to a database table (store, date, forecast, lower and upper range). This table is the product that every downstream user reads from.
3. **Downstream use.**
   - *Operational systems:* forecasts feed stock ordering, staff rotas and budgets, with planners reviewing exceptions rather than every number.
   - *Dashboard:* store managers, planners and finance see the next six weeks with the uncertainty range, and forecast vs actual for recent weeks.
4. **Accuracy monitoring.** Each week, last week's forecasts are compared with actual sales. If error rises above an agreed threshold, the model is reviewed or retrained.

The uncertainty range matters as much as the forecast itself: managers staff for the busy end of the range, and stock planners set safety stock from it.

## Limitations

- **One store.** Results may not generalise across store types and locations.
- **Six test windows.** Up from four, and enough to establish that every model now beats the baseline consistently rather than in one lucky window — but still too few folds for a rigorous confidence interval.
- **Promo and holiday schedules are assumed fully known at forecast time.** Rossmann's historical flags make this true in a backtest; in a real deployment, promotions scheduled weeks ahead can still change (state holidays rarely do).
- **Prophet can predict slightly negative sales on closed days** (an artifact of its additive regression having no floor at 0). This doesn't affect scoring, since closed days are excluded from it, but would need clamping to 0 if these raw forecasts were ever used directly.
- **Training-time figures are single-machine, single-run wall-clock measurements**, not a controlled benchmark — useful as a directional signal (which model costs roughly how much more than another), not as a precise cost estimate.

## Scaling to all 1,115 stores (in progress)

The single-store result above raises a real multi-store question: compute cost
(AutoARIMA's ~65s/store fit, XGBoost's ~3.2s/store retrain) stops being a
rounding error once it's multiplied by 1,115 stores, nightly. `multi_store_forecast.ipynb`
works through that, tracked step by step in `MULTISTORE_PLAN.md`:

- Reuses the same full-calendar + exogenous-feature approach, validated against
  real data-quality issues found along the way (a genuine 184-day gap shared by
  181 stores, not just routine closed-day handling).
- Three-level EDA (group aggregate, per-store distributions, representative
  spot-checks) to check whether grouping stores by `StoreType`/`Assortment` is
  actually justified by the data, rather than assumed.
- Extends the single-store comparison to seven candidates: the four classical
  models above, Prophet (per-store), and XGBoost in three pooling strategies —
  global (one model, all stores, store ID as a feature), per-group, and
  per-store — backtested with the same rolling-origin CV protocol and ranked by
  RMSPE, Rossmann's own competition metric, with relative compute cost reported
  alongside accuracy rather than as an afterthought.
- Deliberately scoped to the model-selection decision, not a full production
  deployment (see `MULTISTORE_PLAN.md`'s decision log for the reasoning) — a
  separate project already demonstrates end-to-end deployment.

See `MULTISTORE_PLAN.md` for the current step and the reasoning behind each
decision along the way.

## Repository

| File | Contents |
|---|---|
| `single_store_forecast.ipynb` | Main analysis above: shared split, all five models (with exogenous promo/holiday features), rolling backtest, fit/predict timing, results tables and plots |
| `multi_store_forecast.ipynb` | Scaling the comparison to all 1,115 stores — see [Scaling to all 1,115 stores](#scaling-to-all-1115-stores-in-progress) above |
| `MULTISTORE_PLAN.md` | Step-by-step decision log for the multi-store work — what was decided, why, and what's still open |
| `data/rossmann/` | Kaggle data (large raw files excluded from version control — see Reproduce below) |

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute single_store_forecast.ipynb --output single_store_forecast.ipynb
```

Download the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) data from Kaggle into `data/rossmann/` first (`train.csv`, `test.csv`, `store.csv`) — these are excluded from version control as large third-party files, not redistributed here. Takes roughly 6–8 minutes, dominated by AutoARIMA's order search across 6 folds. Saved model files (`.pkl`) are excluded from version control; rerun the notebook to regenerate them.
