# Multi-Store Forecasting: Implementation Plan & Decision Log

Tracking doc for extending the single-store model comparison (`single_store_forecast.ipynb`,
Store 1 only) to a decision about what to run in production across all 1,115 Rossmann
stores. Builds on the groundwork already started in `multi_store_forecast.ipynb`
(store grouping, group-level EDA, per-store CV — currently statsforecast models only).

Discussed and decided one step at a time; each step is reviewed before moving to the next.

## Plan

- [x] **1. Scope and decision criteria** — what granularity are we deciding at, what
      metric(s) decide it, how does compute budget factor in?
- [x] **2. Data prep for the multi-store comparison** — reuse/validate the full-calendar +
      exogenous-feature long-format dataframe and per-store split already built.
- [x] **3. Group-level EDA** — review/validate what's already built in
      `multi_store_forecast.ipynb` (data quality, seasonality/trend strength,
      stationarity, cross-sectional group comparison, feature relationships).
- [ ] **4. Bring Prophet and XGBoost into the multi-store comparison** — currently the
      multi-store notebook only tests AutoARIMA/AutoETS/Naive/SeasonalNaive. Add
      **Prophet, per-store only** (no pooling variant exists for it), and **XGBoost in
      three pooling variants** — global (store ID + static attributes as features),
      per-group, and per-store — each evaluated as a separate candidate.
- [ ] **5. Model selection via CV on a sampled subset** — rolling-origin CV across all
      candidate models and pooling strategies (global vs. per-group), ranked by the
      primary metric, with *relative* fit/predict time measured per candidate.
- [x] ~~6. Apply the compute-budget filter~~ — **removed as a separate step,
      2026-10-23**. Folded into step 7 (weighed alongside accuracy, using relative
      compute cost from step 5) and step 8 (where compute feasibility gets a
      *concrete* empirical answer on the actual deployment infra). See decision log.
- [ ] **7. Decide the final model + pooling strategy** — weighs accuracy *and*
      relative compute cost from step 5 together (no separate hard pre-filter).
- [ ] **8. Full-scale fit of the winner** across all 1,115 stores on the chosen
      deployment infra, evaluated once on the untouched final holdout (`test_all`) —
      also where compute feasibility against the deployment environment gets
      confirmed concretely (absorbs old step 6's role). **Deprioritized — see scope
      decision below.**
- [ ] **9. Production considerations** — retraining cadence, forecast table output,
      monitoring/accuracy tracking, what triggers a re-evaluation of the model choice.
      **Deprioritized — see scope decision below.**

## Decisions Log

### Step 1: Scope and decision criteria — 2026-10-22

- **Granularity**: compare (a) one pooled global model vs. (b) one model per
  `StoreType`/`Assortment` group, decided via CV — not per-individual-store (1,115
  separate decisions) unless (a)/(b) show large within-group variance that justifies it.
- **This is not p-hacking** as long as the comparison rule is fixed in advance (compare
  via CV using the agreed primary metric, whichever wins goes to the full-fleet fit) and
  the final holdout (`test_all`) is touched exactly once, after the decision is locked
  in — not iterated on.
- **Metrics**: RMSPE is the primary (ranking) metric — Rossmann's official competition
  metric, scale-invariant across stores of very different sales volumes, and reasonably
  explainable to the business ("average percentage error, weighted more heavily toward
  big misses"). MAPE/WAPE are reported alongside as more intuitive business-facing
  context. A stockout-rate business metric is deferred to a later phase (requires
  simulating an inventory policy against forecast errors) — not part of initial model
  selection.
- **Compute budget**: treated as a hard SLA filter, not just a tiebreaker. Fit/predict
  time is measured per candidate during CV (as already done for the single-store
  comparison); anything that can't plausibly retrain within the nightly/weekly window at
  1,115-store scale is disqualified regardless of accuracy. This is also the reason
  step 4 specifically calls for a **global XGBoost** (one parallelizable training job)
  rather than only per-series/per-group classical models (AutoARIMA doesn't scale to
  1,115 series).

### Step 2: Data prep — _planned, not yet executed_

Gaps to close in `df_long_all`/`train_all`/`test_all` before relying on them for EDA or
modeling (execution deferred — this is the task list to run later):

1. **Merge store static attributes** (`StoreType`, `Assortment`, `CompetitionDistance`,
   `Promo2` enrollment flag, etc. from `store.csv`) into the modeling dataframe. Needed
   only for **global XGBoost** (store ID + attributes as features) — classical models
   and Prophet are fit strictly per-store, so a feature that's constant across a given
   store's own rows has zero within-series variance and literally cannot have an
   estimable effect in a per-store fit (it's collinear with that model's intercept).
   This isn't a capability gap to work around — it's a structural fact about per-series
   fitting, same reasoning as the earlier exogenous-regressor fairness discussion.
2. **Derive time-varying versions of two nominally-static columns**, since they're not
   *purely* static: whether a `Promo2` continuation promo is actually active on a given
   date (from `Promo2SinceWeek`/`Promo2SinceYear` + `PromoInterval`), and whether a
   nearby competitor already existed on a given date (from
   `CompetitionOpenSinceYear`/`Month`). Both vary within a single store's own series, so
   — unlike the raw static values — these two derived flags should go into the
   exogenous set for **every** model, not just global XGBoost.
3. **Validate the "full calendar" assumption per store** — confirm every store has one
   row per calendar day with no genuinely missing dates (as opposed to expected
   `Open==0` rows), since a silent gap would break the `season_length=7`
   day-of-week assumption the notebook's own markdown already warns about.
4. **Check minimum history length per store** — `horizon=42` assumes every store has
   enough history left after holding out the last 42 days; flag any store with short
   total history before it causes a CV crash or a near-empty training set in step 5.
5. **Decide handling for data-quality issues** (duplicates, negative values,
   closed-but-nonzero-sales rows) surfaced by step 3's EDA — drop, clip, or impute —
   so step 3 and step 4 aren't blocked on an undecided call.

### Step 2: Data prep — executed and closed out — 2026-10-23

All five gaps implemented in `multi_store_forecast.ipynb` (new "Step 2: Data Prep —
Closing the Gaps" section, right after `df_long_all` is first built, before the
per-store split), and validated by actually executing that section of the notebook
against the real data:

1. **Static attributes merged** (`StoreType`, `Assortment`, `CompetitionDistance`,
   `CompetitionOpenSinceMonth/Year`, `Promo2`, `Promo2SinceWeek/Year`,
   `PromoInterval`) into `df_long_all`, kept as separate columns and **not** added to
   `exog_cols` — reserved for global XGBoost (step 4B) only, per the original
   reasoning (collinear with a per-store model's intercept otherwise).
2. **Two time-varying flags derived** and added to `exog_cols` for every model:
   `promo2_active` (Promo2 enrolled, continuation start date reached, current month
   in `PromoInterval`) and `competitor_exists` (distance known, and either the
   recorded opening date has passed or no opening date was recorded — treated as
   predating the dataset). Fleet-wide means: `promo2_active` ≈ 14.6%,
   `competitor_exists` ≈ 91.6%.
3. **Full-calendar assumption: found violated, then fixed.** 181 of 1,115 stores
   (not just the 180 found during the single-store-era EDA pass) were missing the
   *same* contiguous 184-day block (2014-07-01 to 2014-12-31) — not random noise, a
   real structural gap (`store.csv`/Kaggle-forum-documented, consistent with a
   known refurbishment-closure period). Fixed by reindexing every store onto one
   shared full daily calendar (global min/max date), filling the gap rows as an
   unreported closure (`Open=0`, `y=0`, all other flags 0) — the same treatment
   already given to every other `Open==0` day. `df_long_all` grew from 1,017,209 to
   1,050,330 rows; every store now has exactly 942 rows.
4. **Minimum history check**: not a concern on this dataset — every store has the
   same 942-row history after the calendar fix, comfortably above the
   `horizon x (n_windows+1) = 168`-row minimum the CV protocol needs. The
   conditional "cold-start policy" item from the Step 3 EDA triage below is
   confirmed **not needed as an implementation** — stays as documented policy only.
5. **Other data-quality issues**: checked directly against the raw data — zero
   duplicates, zero negative Sales/Customers, zero `Open==0`-with-`Sales>0` rows.
   54 rows (41 stores) have `Open==1` with `Sales==0`; kept as-is, since the
   `rmspe()` metric (defined later in the notebook) already excludes actual-zero
   days from the percentage-error calculation.

**Side effect on existing cells**: the four places that previously built model
input via `.drop(columns=['store_group'])` (the per-group CV loop, the pooled CV
cell, the per-group forecast loop, the global forecast cell) were changed to
explicit `['unique_id', 'ds', 'y'] + exog_cols` selection — otherwise the newly
merged static columns would've leaked into the classical models' `X_df` as
collinear regressors. `exog_cols` itself is unchanged in meaning, just extended
with the two derived flags from item 2.

### Step 3: EDA — expanded to three levels — 2026-10-22

Group-level EDA alone isn't sufficient: classical models (AutoARIMA, AutoETS) and
Prophet all fit **per-store**, but group-level EDA looks at an *aggregated* series
(mean sales across stores in a group). Averaging smooths out individual-store noise,
which can make a group look more stationary / more cleanly seasonal than any
individual store actually is — so conclusions from aggregate EDA don't necessarily
describe what a per-store model will actually see. Three complementary levels,
not one:

1. **Group-level (aggregate)** — already built in
   `multi_store_forecast.ipynb` (cells 16–36): grouping by
   `StoreType`/`Assortment`, data quality checks, STL decomposition + seasonal/trend
   strength, ADF/KPSS stationarity, ACF/PACF, cross-sectional comparison (validates
   whether the grouping itself is justified), feature relationships (Promo/holiday
   effect, `CompetitionDistance` correlation). Purpose: validate the segmentation
   decision and support business-facing narrative — **not** a proxy for per-store
   model behavior.
2. **Per-store distributional (feature-based) EDA** — new. Compute the same lightweight
   diagnostics (stationarity result, seasonal/trend strength, promo/holiday effect
   size, intermittency, history length) for **every store individually**, then look at
   the *distribution* of those per-store numbers (histograms, overall and split by
   group) rather than a single aggregate value. Matches standard large-scale
   forecasting practice (Hyndman's feature-based time series analysis / `tsfeatures`,
   underlying the FFORMS/FFORMA model-selection literature). This is also where step
   2's per-store data-quality and minimum-history-length checks naturally live as
   distributions, and where the two newly-derived time-varying flags (Promo2-active,
   competitor-exists) get the same effect-size check as the existing exogenous
   features, before step 4/5 spends compute on them.
3. **Individual representative-store spot-checks** — a handful of stores spanning
   different groups/behaviors (not just Store 1, already covered thoroughly in
   `single_store_forecast.ipynb`) to sanity-check that the distributional/group-level
   conclusions actually hold up on real individual series, at the granularity
   classical/Prophet models actually operate at.

**Additional industry-standard EDA items — feasibility triaged 2026-10-22:**

Committed (implementable now, no new tools/data needed, low-moderate effort):
- **Demand-pattern classification** (intermittent vs. continuous demand, via average
  demand interval + CV² — Syntetos-Boylan). Likely N/A here since Rossmann's
  zero-sales days are known scheduled closures (`Open` flag), not random intermittent
  demand — but run the actual check rather than assume.
- **Fleet-wide automated outlier/anomaly detection** (STL-residual or IQR-based, run
  per store, flagged/counted rather than reviewed individually — reuses the STL
  decomposition already used at group level).
- **Train-vs-test distributional drift (covariate shift) check** — whether sales
  levels or driver frequency (e.g. promo frequency) differ meaningfully between
  older training history and the most recent period.
- **Data limitation to document** (not code, just write-up): `StateHoliday` isn't
  store-specific to each store's actual German federal state, and `store.csv` has no
  `State` column — no way to know if a given holiday flag even applies to a given
  store. Didn't matter for the single-store comparison; does matter across 1,115
  multi-state stores. Document as an acknowledged limitation, not fixable with
  available data.

Conditional (depends on an upstream check before committing effort):
- **Cold-start policy for short-history stores** — only worth building/testing if
  step 2's per-store history-length check actually finds stores with meaningfully
  short history. If every store has close to the full ~2.5-year window, this becomes
  a documented production policy (design it, but can't validate it against this
  dataset) rather than something to implement and test now.

Deferred to future work, not pursued now (no confirmed downstream requirement):
- **Hierarchical reconciliation** (per-store forecasts summing consistently to
  regional/company-wide totals, via Nixtla's `hierarchicalforecast` package, which
  plugs directly into `statsforecast`'s output format). Technically implementable,
  but speculative without a confirmed need for aggregate-level forecasts — document
  as the recommended approach *if* that requirement ever appears, rather than
  building it against an unconfirmed need.

### Step 3: consistency fix after Step 2, then full three-level expansion — 2026-10-23

First pass (scoped deliberately small): the consistency fix only, deferring the
full three-level EDA expansion. Second pass (same day, after the user asked to
implement the rest): the full expansion — per-store distributional EDA,
representative spot-checks, demand-pattern classification, fleet-wide outlier
detection, train/test drift check, documented StateHoliday limitation. All of it
now implemented in `multi_store_forecast.ipynb` and validated by executing the
notebook end-to-end through this section (no errors; ~450s total, dominated by the
per-store ADF+STL loop over 1,115 stores).

**Consistency fix (first pass):**

- **Feature Relationships section** (group-level EDA, item 6) now also checks the two
  flags derived in Step 2 (`promo2_active`, `competitor_exists`) with the same
  effect-size-by-group treatment as `Promo`/`StateHoliday`/`SchoolHoliday`, and
  includes both in the exogenous correlation matrix — this was a concrete Step 3
  requirement from the plan ("these two derived flags... get the same effect-size
  check as the existing exogenous features") that wasn't implemented until now.
  Validated by executing that section of the notebook: effect sizes vary a lot by
  group (e.g. `promo2_active` uplift ranges from +20% in `b_a` to -25% in `b_b`;
  `competitor_exists` effect ranges from +8% in `a_c` to -41% in `c_c`) — real
  signal, not noise, worth keeping as exogenous regressors.
- **Data-quality takeaway note added**: the EDA Takeaways cell now explicitly says
  the missing-calendar-days finding from EDA section 1 was resolved in Step 2 (the
  184-day/181-store gap, fixed by reindexing), so it doesn't read as an unresolved
  open question sitting ahead of its own fix later in the notebook.
- **No cell reordering**: EDA (section 1) still runs on the raw, pre-fix data before
  `df_long_all`/Step 2 are built later in the notebook. This was reconsidered and
  kept intentionally — the plan's own Step 2 item 5 already frames it this way
  ("issues surfaced by step 3's EDA" are then decided in step 2), and since every
  EDA section downstream of section 1 filters to `Open==1` anyway, the calendar-gap
  fix (which only adds `Open==0` rows) doesn't change any of those numbers whether
  computed before or after the fix.

**Full three-level expansion (second pass)**, added as a new "Per-Store &
Fleet-Wide (Levels 2 & 3)" section right after Step 2, before the per-store split,
plus a drift-check section after the split:

- **Per-store distributional EDA** (Level 2): the same diagnostics as group-level
  (stationarity via a fixed-maxlag ADF test, seasonal/trend strength via
  `stl_strength()`, promo/holiday effect size, intermittency, history length),
  computed for all 1,115 stores individually and viewed as distributions (overall
  + by group). Findings: seasonal/trend/promo strength genuinely vary
  store-to-store (wide histograms, substantial within-group spread too) — the
  group-level EDA is a reasonable summary, not a guarantee every store behaves
  identically. ~95% of stores classify stationary, ~5% don't. History length is a
  single spike at 942 days for every store (confirms the Step 2 fix held).
- **Fleet-wide outlier detection** (IQR-based on each store's own open-day sales,
  computed in the same per-store pass): 16,847 flagged days fleet-wide (2.00% of
  all open-store-days); only 4/1,115 stores have zero flagged outliers. Treated as
  expected variability, not a data-quality problem to fix.
- **Demand-pattern classification** (Syntetos-Boylan ADI/CV², on open-day sales):
  **all 1,115 stores classify as `smooth`** — confirms the plan's hypothesis that
  Rossmann's zero-sales days are scheduled closures, not random intermittent
  demand. No intermittent-demand-specific modeling approach is needed.
- **Representative per-group spot-checks** (Level 3): one store per group, chosen
  as closest to that group's median `seasonal_strength` (not arbitrary) —
  store IDs: `{'a_a': '44', 'a_c': '447', 'b_a': '682', 'b_b': '1097', 'b_c': '562',
  'c_a': '944', 'c_c': '63', 'd_a': '309', 'd_c': '1078'}`. Confirms the
  group-level/distributional read on real series.
- **Train-vs-test drift check**: mean sales on open days is essentially unchanged
  between `train_all` and `test_all` (+0.35%) — the core forecast target isn't
  drifting. `pct_closed` (-29.6%), `school_holiday_freq` (+69.9%), and
  `state_holiday_freq` (train 3.1% → test **0%**) differ more, but this is a
  calendar-window artifact (the 42-day holdout just doesn't happen to contain a
  state holiday), not genuine covariate shift. Flagged for later: the final
  holdout (Step 8) can't validate state-holiday handling at all, since it contains
  zero holiday examples — that's only ever tested via the CV windows.
- **Holiday effect size confirmed as a per-store dead end**: only 156/1,115 stores
  ever open on a state holiday, so a per-store holiday-uplift number is undefined
  for most stores — the group-level check is the reliable read for this feature.
- **Cold-start policy**: confirmed not needed as an implementation (no
  short-history stores exist in this dataset) — documented as policy only (fall
  back to group-level/pooled model for any future short-history store).
- **StateHoliday state-mismatch limitation**: documented in the notebook as an
  acknowledged ceiling on the holiday features' signal, not fixable with available
  data (no `State` column to match holidays to the right federal state).
- **Hierarchical reconciliation**: remains deferred, not pursued — still no
  confirmed need for aggregate-level forecasts.

### Step 4/5: Pooling strategy, mixed-model policy, and fair comparison — 2026-10-22

- **Pooled vs. grouped vs. per-store is not a universal choice — it only applies to
  models capable of sharing learned parameters across series.** AutoARIMA, AutoETS,
  SeasonalNaive, and Prophet all fit strictly per-store regardless of how calls are
  batched; there is no pooled/grouped variant for them. Only **XGBoost** has a real
  three-way choice: (a) global (one model, all stores' data, store ID + attributes as
  features), (b) per-group (one model per `StoreType`/`Assortment`), (c) per-store (one
  model per store). All three XGBoost variants are evaluated as separate candidates.
- **Mixed model types across groups in production**: default to **one model/pooling
  strategy for the entire fleet**, matching standard industry practice — teams generally
  don't maintain multiple model codebases/retrain pipelines for a marginal accuracy gain.
  The exception that would justify splitting by model type is a **large, consistent CV
  gap tied to a real structural difference in the data** (e.g. intermittent vs.
  continuous demand — not the case here, since all groups are regular retail stores),
  not a small CV score difference. Step 5's CV results are what actually decide this,
  not an assumption going in.
- **"Apples to oranges" resolved**: fairness requires the same CV protocol (sampled
  stores, rolling windows, horizon, primary metric RMSPE) applied identically to every
  candidate's forecasts vs. the same actuals — not structural symmetry between models.
  Models that structurally can't use exogenous regressors (e.g. AutoETS) aren't
  artificially equalized; the backtest is specifically testing whether exogenous access
  earns a real advantage. Every (model type, pooling variant) pair is one row in a
  single leaderboard, ranked together by RMSPE, with MAPE/WAPE as supporting context.
- **Compute cost is compared in a consistent unit** — total wall-clock to produce
  forecasts for the full 1,115-store fleet — not per-model-object, so "one global
  XGBoost" isn't trivially favored against "1,115 AutoARIMA fits" just by unit choice.
  This is confirmed as standard practice (same approach used in M5-competition-style
  backtests and production forecasting at scale, e.g. Uber/Walmart-style pipelines).

### Step 4: Execution plan — 2026-10-22

Concrete implementation plan for the decisions already logged under "Step 4/5" above
(this is the *how*, not new *what*/*why* decisions):

**A. Prophet — per-store only**
- Loop over the sampled stores (`cv_sample_df`), fitting one Prophet model per store
  per CV window, with the same regressors as `single_store_forecast.ipynb` (`Promo`,
  `SchoolHoliday`, `StateHoliday` flag, `Open`) plus the two derived time-varying flags
  from step 2 (Promo2-active, competitor-exists).
- No native multi-series engine, so the loop needs to be parallelized (e.g. `joblib`)
  to keep wall-clock reasonable across sampled stores x CV windows.
- Output must match the same long-format schema (`unique_id`, `ds`, `cutoff`, model
  columns) that `statsforecast`'s `cross_validation()` already produces, so it merges
  into one `utilsforecast.evaluate()` call alongside everything else.

**B. XGBoost — three pooling variants**
- Lag(1,2,3,7) and rolling mean/std(7) features must be computed **within each
  store's own series** (`groupby('unique_id')` before `.shift()`/`.rolling()`) —
  getting this wrong is the easiest place to silently leak data across stores.
- **Global**: one model, all sampled stores pooled, store ID + static attributes
  (`StoreType`, `Assortment`, `CompetitionDistance`, `Promo2` enrollment) as features.
  Store ID encoding — **not locked in, treated as an empirical question for step 5**:
  default starting point is native **categorical** (`enable_categorical`), not raw
  numeric (raw numeric is ruled out outright — it implies a false ordinal
  relationship between store IDs). One-hot is ruled out too (too sparse at ~1,115
  categories, weaker splits than native categorical). If native categorical
  underperforms per-group/per-store in CV, the documented fallback to test is
  **target/mean encoding** (leakage-safe, expanding-window store-level statistic —
  what several top Rossmann Kaggle solutions used for this exact problem), not a
  full redo. Entity embeddings (the actual 3rd-place Rossmann solution's approach)
  noted as the more sophisticated option, likely not worth the effort here.
- **Per-group**: one model per `StoreType`/`Assortment` group.
- **Per-store**: same approach as the existing single-store notebook, replicated
  across the sample.
- All three need **recursive day-by-day forecasting** through the horizon (each day's
  lag features depend on the model's own prior forecasts, not true future actuals) —
  same technique as `single_store_forecast.ipynb`, implemented correctly per variant.

**C. One unified leaderboard**
- Merge Prophet + the 3 XGBoost variants + existing AutoARIMA/AutoETS/Naive/
  SeasonalNaive CV results into a single table, ranked by mean RMSPE, with MAPE/WAPE
  and fit+predict wall-clock (fleet-wide unit, per the Step 4/5 decision above)
  reported alongside.

**Prophet runtime estimate — computed 2026-10-22** (using real group counts from
`store.csv`, no notebook execution): `StoreType`/`Assortment` actually has **9**
groups, not the 12 that 4x3 would suggest (sizes: a_a=381, a_c=221, d_c=220, d_a=128,
c_a=77, c_c=71, b_b=9, b_a=7, b_c=1). With `n_per_group=20` (already set in the
notebook), total sampled stores = **137**. At `n_windows=3` (already set), that's
411 individual Prophet fits; at ~0.7s/fit+predict (measured in
`single_store_forecast.ipynb`'s single-store backtest), **~288s serial (~5 min), ~1–2 min
parallelized**. Conclusion: Prophet is cheap at this sample size — the earlier
concern that it could be the CV bottleneck was overstated and is now resolved.

**New finding from the same calculation — the actual bottleneck is AutoARIMA, not
Prophet.** At 65.2s/fit (same single-store measurement), 137 x 3 x 65.2s ≈ 26,800s
serial (**~7.4 hours**), or **~1.9 hours even with 4-way parallelism** — for a model
the single-store backtest already showed barely beats AutoETS (17.96 vs 18.02 MAPE)
while costing ~100x more. **Decided 2026-10-23: (b) — shrink AutoARIMA's sample specifically** (e.g. 5
stores/group instead of 20; every other candidate keeps the full `n_per_group=20`
sample), rather than dropping it outright or running it at full sample. Rationale:
standard "staged/successive elimination" practice (same principle behind
Hyperband/ASHA-style hyperparameter search) — a cheap pilot (single-store evidence)
already suggests AutoARIMA is weak and expensive, but extrapolating that to "drop
entirely" from n=1 risks exactly the hasty generalization the sample-based CV
methodology (step 1) exists to avoid; running it at full sample wastes ~1.9hrs of
parallelized compute on a candidate already expected to lose. A small confirmatory
run is the cheaper way to get real multi-store evidence either way. This is also the
more rigorous version of the step 1 principle "compute budget as a hard filter
before accuracy breaks the tie" — applied during CV screening itself, not only at
the final full-fleet stage.

### Step 5: Model selection via CV — execution plan — 2026-10-23

Ties together everything decided in Steps 1–4 into one CV run.

**Candidates (each a separate leaderboard row):**

| # | Candidate | Sample |
|---|---|---|
| 1 | SeasonalNaive (baseline) | per-store, full sample (137 stores) |
| 2 | AutoETS | per-store, full sample |
| 3 | AutoARIMA | per-store, **reduced** sample (5/group, per Step 4's decision) |
| 4 | Prophet | per-store, full sample |
| 5 | XGBoost — global | full sample, store ID (categorical default) + static attributes |
| 6 | XGBoost — per-group | full sample |
| 7 | XGBoost — per-store | full sample |

**Protocol** (already fixed by earlier steps): rolling-origin CV, `n_windows=3`,
`step_size=horizon=42`, expanding training window; RMSPE as the primary ranking
metric, MAPE/WAPE as secondary business-facing context (`utilsforecast.evaluate()`);
fit+predict wall-clock measured per candidate, reported in the fleet-wide unit
(extrapolated time to cover all 1,115 stores), per the Step 4/5 compute-cost-
consistency decision.

**Output**: one leaderboard ranked by mean RMSPE, plus the per-group breakdown
(`cv_group_summary`, already planned in the existing notebook) to catch cases where
the best model differs by group — feeds the "mixed model types?" policy decision
from Step 4/5.

**Two caveats flagged, not yet resolved:**
1. AutoARIMA's row isn't on the same sample as everyone else (~45 stores vs. 137) —
   its RMSPE carries more sampling noise than the other rows; annotate in the
   leaderboard rather than treating it as directly comparable at the same confidence
   level as the full-sample candidates.
2. Step 5 depends on Step 3's EDA items (demand-pattern classification, outlier
   detection, drift check) having actually run first — a data-quality issue caught
   late would undermine these CV results. Confirm Step 3 is complete before reading
   Step 5's output as final.

### Step 6: Compute-budget filter — _pending_

### Step 7: Final model decision — _pending_

### Step 6 removed, folded into Steps 7/8 — 2026-10-23

**This revises a principle locked in under Step 1** ("compute budget treated as a
hard SLA filter, not just a tiebreaker, applied *before* accuracy breaks the tie").
That principle assumed a precise, known SLA number to filter against. In practice
we never had one — Step 6's own discussion established there's no real company SLA
here, just a choice between AWS credits (expiring January 2027) and GitHub Actions.
A precise pre-filter was always somewhat artificial without that number.

**Revised approach**: Step 7 (final model decision) weighs accuracy *and* **relative**
compute cost together — the relative comparisons we already have (e.g. AutoARIMA's
~100x cost vs. AutoETS for marginal accuracy gain, which already drove the step 4
decision to shrink its CV sample) are enough to inform the decision without a hard
SLA gate. Step 8 (full-scale fit, on the chosen deployment infra) is where compute
feasibility gets a *concrete*, empirical answer, replacing the old step 6's role —
rather than guessing a threshold upfront, we see whether the chosen model actually
fits a reasonable schedule once it's really deployed.

### Step 6/8/9: Deployment environment — 2026-10-23

- **Primary**: AWS Batch or ECS Fargate, scheduled via EventBridge Scheduler,
  orchestrated/observed via Prefect Cloud's free workspace. Paid for with the user's
  AWS Free Plan credits (a real dollar balance, not the quota-based Free Tier — does
  cover Fargate/Batch billed compute). **Credits expire January 2027** — real
  schedule constraint, ~2.5 months from this decision (2026-10-23). Set a billing
  alert early so an always-on resource doesn't silently drain the balance.
- **Fallback**: GitHub Actions (repo kept public), confirmed free + unlimited
  minutes + 4 vCPUs on standard `ubuntu-latest` runners — zero cost, no expiration,
  used if AWS setup takes longer than the credit window allows.

### Step 8: Full-scale fit — _pending_

### Step 9: Production considerations — _pending_

### Scope decision: stop after Step 5 + a lightweight Step 7 — 2026-10-23

**This project's goal is a CV/portfolio writeup, not an actual production
deployment** — a separate project (the churn project) already demonstrates
deployment, so duplicating that here isn't needed. Decided: execute through
**Step 5** (CV leaderboard across all candidates) plus a **lightweight Step 7**
(write up which model + pooling strategy wins, weighing Step 5's accuracy
ranking against the relative compute-cost numbers already measured there —
no new experiments needed, just synthesis of numbers Step 4/5 already
produce). This gives a complete, decided narrative ("evaluated N candidates
via CV, selected X for Y reason") instead of stopping at a bare leaderboard.

**Steps 6/8/9 (and the AWS Batch/Fargate vs. GitHub Actions deployment
environment decision logged above) stay documented as a productionization
roadmap in the README, not executed.** Not forgotten, just out of scope for
this project's purpose — revisit only if this project's goal changes (e.g. an
actual deployment becomes wanted later).
