# Model Evaluation

The [`Modeling_Tool.Eval`](../api/eval.md) subpackage turns scores and labels into the artifacts a model review expects:
KS/AUC summaries across train, test, and out-of-time samples, decile **Gains tables**, challenger-versus-champion
comparisons, cross-risk matrices, slice-by-slice evaluation, and ROC/KDE/percentile/gain figures.

| I want to... | Use | Import |
|---|---|---|
| Summarize KS / AUC / top-decile bad rate across train, test, OOT | `PerformanceEvaluator` | `from Modeling_Tool import PerformanceEvaluator` |
| Do the same with a one-call function | `get_perf_summary` | `from Modeling_Tool import get_perf_summary` |
| Build a score-band (decile) Gains table | `GainsTableCalculator`, `get_gains_table` | `from Modeling_Tool import GainsTableCalculator, get_gains_table` |
| Add business columns to a Gains table | `calculate(add_func=...)`, `get_gains_table_by_cust_metrics` | `from Modeling_Tool import get_gains_table_by_cust_metrics` |
| Compare a challenger score with the champion | `Model_Evaluation_Tool` | `from Modeling_Tool import Model_Evaluation_Tool` |
| Cross two scores into a risk matrix | `cross_risk` | `from Modeling_Tool import cross_risk` |
| Evaluate every month, channel, or segment | `EvaluationPipeline` | `from Modeling_Tool import EvaluationPipeline` |
| Draw ROC / KDE / percentile / gain figures | `evaluate_performance`, `comparison_performance` | `from Modeling_Tool import evaluate_performance, comparison_performance` |
| Find score cut-offs for target lift multiples | `calc_lift_apt` | `from Modeling_Tool import calc_lift_apt` |
| Build curves yourself | `calc_roc`, `calc_pr`, `calc_equid_dist`, `calc_equid_pct`, `calc_fixed_pct` | `from Modeling_Tool import calc_roc, calc_pr, ...` |

The `plot_*` helpers, `evaluate_distribution`, `summarize_pr`, `summarize_pct`, `tie_score_rate`, and `score_unique_rate` are
not exported at the top level; import them from `Modeling_Tool.Eval`. `summarize_roc` lives in
`Modeling_Tool.Eval.evaluate_model`.

!!! warning "Score direction"

    Every evaluator assumes that **a higher score means a higher chance of the bad outcome**, as with a predicted
    probability of default. AUC is computed as `roc_auc_score(y_true, y_score)`, and the Gains table's
    `RANK_ORDER_BUMP` flags bins where the bad rate moves against that direction. If your score runs the other way (a
    scorecard where more points mean lower risk), evaluate a flipped copy such as `df["risk"] = df["points"].max() - df["points"]`.

## Example data

Every snippet on this page runs top to bottom in one Python session. The setup below creates a synthetic portfolio, a
time-based train / test / out-of-time split, a LightGBM model, and the score columns used later.

```python
import os

import numpy as np
import pandas as pd

from Modeling_Tool import GradientBoostingModel

rng = np.random.default_rng(42)
n = 8000
months = [f"2025-{m:02d}" for m in range(1, 10)]

df = pd.DataFrame({
    "flow_id":     np.arange(n),                       # unique application id
    "apply_month": rng.choice(months, n),
    "city_grade":  rng.choice(["A", "B", "C", "D"], n),
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
    "sample_wgt":  rng.uniform(0.5, 2.0, n),           # e.g. balance or correction weight
})
logit = -2.2 - 0.02 * (df["score_b"] - 600) + 0.5 * df["n_overdue"] - 0.8 * df["utilization"]
df["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
df["bad_dpd30"] = df["bad_flag"] * rng.binomial(1, 0.6, n)    # a second, stricter label

features = ["age", "income", "score_b", "utilization", "n_overdue"]
train = df[df["apply_month"] <= "2025-06"].copy()
test = df[df["apply_month"] == "2025-07"].copy()
oot = df[df["apply_month"] >= "2025-08"].copy()

gbm = GradientBoostingModel(
    "lgb",
    params={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
            "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1},
)
gbm.fit(train[features], train["bad_flag"], test[features], test["bad_flag"])

for frame in (train, test, oot):
    frame["prob"] = gbm.predict(frame[features])                                # challenger score
    frame["score_old"] = 1 / (1 + np.exp((frame["score_b"] - 600) / 60 + 2))    # an older, simpler score

holdout = pd.concat([test, oot], ignore_index=True)    # three months of unseen data
os.makedirs("output", exist_ok=True)
```

## Sample-Weighted Evaluation

Pass `weight_col` (a column name) to `PerformanceEvaluator`, `GainsTableCalculator`, `get_gains_table`,
`Model_Evaluation_Tool`, or `cross_risk`, or `sample_weight` (an array) to the low-level functions, and the metrics are
aggregated by weight. **Without weights the behavior is the historical, unweighted one.**

### N and N_RAW

Weighted tables report sample size two ways:

| Column | Meaning | Typical use |
|---|---|---|
| `N` | **Sum of weights** in the bin or dataset (effective sample size) | Weighted bad rate, Lift, capture rate |
| `N_RAW` | **Number of rows** | Auditing, reconciling record counts |

If each row is an account and the weight is the loan balance, `N_RAW` is the number of accounts and `N` is the total
balance. The weighted bad rate is `sum(y * w) / sum(w)`, and Lift is the bin's weighted bad rate divided by the overall
weighted bad rate. If all weights are 1, `N == N_RAW`.

### Weighted metric semantics

| Metric | Weighted behavior |
|---|---|
| `AUC` | scikit-learn `roc_auc_score(..., sample_weight=...)` |
| `KS` | Maximum gap between the weighted cumulative bad and good shares (`abs(TPR - FPR)` on the weighted ROC curve) |
| `avgTrue`, `avgScore` | Weighted averages |
| Gains `AVG_BAD`, `LIFT`, `WOE`, `IV` | Computed from the weighted bad and good totals in each bin |

A weighted `PerformanceEvaluator.evaluate()` returns a narrower table than the unweighted one:
`index, dataset, DATASET, AUC, KS, LIFT, IV, N, N_RAW, avgTrue, avgScore`, where `LIFT` is the largest bin lift and `IV` is the
Gains-table IV. The `Top10%_*` / `Btm10%_*`, `*_Shift`, and other Gains-summary columns exist only in the unweighted table.

### Where weights are applied

| API | Weights are applied | Weights are silently **not** applied |
|---|---|---|
| `PerformanceEvaluator.evaluate` | `weight_col` on the constructor, on `evaluate`, or per dataset on `add_dataset` | When `oot_grp_name` or `benchmark_dataset` is passed |
| `get_perf_summary` | `weight_col`, when `oot_grp_name` is `None`. Requires `scr_name`: with `model` and `feature_cols` the weighted call raises `KeyError`, so add a score column first | When `oot_grp_name` is passed |
| `get_gains_table`, `GainsTableCalculator` | `weight_col` | When `grp_name` is passed. On the weighted path `add_func` and `withSummary` are ignored, and the bins are always equal-weight (`weighted_binning` does not change them in 0.8.2) |
| `Model_Evaluation_Tool` | `weight_col`, in `model_perf_compare` and `get_gains_summary` | With `grp_name`; `get_cross_risk_summary`, `cross_perf_eval`, and `multi_dim_eval` are unweighted |
| `cross_risk` | `weight_col` or `sample_weight` | |
| `evaluate_performance` | The `'sample_weight'` key of each dataset (or the function-level `sample_weight`) | |
| `calc_roc`, `calc_pr`, `calc_equid_dist`, `calc_equid_pct`, `calc_fixed_pct`, `calc_lift_apt` | `sample_weight` | |
| `get_gains_table_by_cust_metrics`, `comparison_performance` | Not supported (no weight argument) | |

Use the **same weights in training and evaluation**, otherwise the metrics describe a different population than the one the
model was fitted on. When weights are applied to figures, the panel titles are marked `Weighted`.

## 1. Multi-dataset summary: `PerformanceEvaluator`

Register each dataset once, then call `evaluate()`. The model must expose `predict_proba` (a `GradientBoostingModel`
or any scikit-learn-style classifier); alternatively pass `scr_name` to evaluate an existing score column.

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=features)
perf = (
    evaluator.add_dataset("train", train)
    .add_dataset("test", test)
    .add_dataset("oot", oot)
    .evaluate(display=False)        # display=True (the default) renders the table with IPython
)
print(perf[["index", "N", "KS", "AUC", "Top10%_TargetRate", "Top10%_Lift"]])
```

!!! note "`display`"

    `evaluate()` calls `IPython.display.display` when `display=True`, which is the default. Outside a notebook (or when
    IPython is not installed) pass `display=False`. The table is returned either way.

### Output columns (unweighted)

| Column | Meaning |
|---|---|
| `index` | The dataset name given to `add_dataset` |
| `N`, `avgTrue`, `avgScore` | Row count, overall bad rate, mean score |
| `KS`, `AUC` | Maximum TPR − FPR, and the area under the ROC curve |
| `Btm10%_TargetRate`, `Top10%_TargetRate` | Bad rate in the lowest- and highest-scored band. The percentage is `100 / pct_bins`: `pct_bins=5` gives `Btm20%_*` and `Top20%_*` |
| `Btm10%_Lift`, `Top10%_Lift` | Band bad rate divided by `avgTrue` |
| `AUC_Shift`, `KS_Shift` | Previous row's metric divided by this row's, minus 1: a positive value means this dataset is weaker than the one above it |
| `N_BUMP`, `MIN_RISK_DEP`, `MAX_RISK_DEP`, `KS_IN_GAINS`, `LIFT_IN_GAINS`, `IV`, `N_BINS` | Summary of the dataset's Gains table: rank-order breaks, smallest and largest bin-to-bin bad-rate change, KS, largest lift, total IV, and number of bins |

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `tgt_name` | required | Target column, or a list/tuple of target columns (see [several labels](#several-target-labels)) |
| `scr_name` | `None` | Score column to evaluate. Use it instead of `model` and `feature_cols` |
| `model`, `feature_cols` | `None` | A fitted classifier and the columns passed to its `predict_proba` |
| `dist_bins` | `20` | Equal-width score bins in the distribution panels |
| `pct_bins` | `10` | Equal-frequency bins for the percentile/gain panels, Gains tables, and Top/Btm bands |
| `weight_col` | `None` | Default weight column; `add_dataset` can override it per dataset |
| `spec_values` | `None` | Sentinel scores (for example `-1`) excluded from the ranking metrics (`N`, `KS`, `AUC`, Top/Btm bands) |
| `ascending` | `None` | `None` keeps each panel's historical direction; a bool applies one direction to the Gains tables and figures |
| `precision`, `min_bin_prop`, `include_missing`, `equal_freq`, `chi2_method`, `chi2_p`, `init_equi_bins`, `tree_binning`, `random_state` | `5`, `0.05`, `False`, `True`, `False`, `0.9`, `1000`, `False`, `42` | Binning controls forwarded to the Gains table |

`add_dataset(name, data, weight_col=None, overwrite=False)` returns the evaluator, so calls chain. Registering an existing
name raises `KeyError` unless `overwrite=True`, so build a new evaluator when re-running a notebook cell.

`evaluate(oot_grp_name=None, min_data_size=100, grp_colname=None, fig_save_path=None, rpt_save_path=None, to_show=False, display=True, gains_table=False, benchmark_dataset=None, weight_col=None)`:

| Argument | Meaning |
|---|---|
| `oot_grp_name` | Evaluate each dataset separately for every value of this column; groups smaller than `min_data_size` are skipped. The result gains a column named `grp_colname` (default: the same name) |
| `fig_save_path` | **File** path for the figure, for example `output/perf.png`; the folder must exist |
| `rpt_save_path` | CSV path for the summary table |
| `to_show` | Display the figures (notebooks). Default `False` |
| `gains_table` | Build the percentile and gain panels from the Gains-table binning |
| `benchmark_dataset` | Name of a registered dataset (or a DataFrame) whose score bins are reused for every dataset, so bins are comparable |
| `weight_col` | Overrides the constructor's weight column for this call |

### Evaluate by group

```python
by_month = (
    PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=features)
    .add_dataset("oot", oot)
    .evaluate(oot_grp_name="apply_month", min_data_size=100, display=False)
)
print(by_month[["index", "apply_month", "N", "KS", "AUC"]])
```

### Fixed bin edges from a benchmark dataset

```python
fixed = (
    PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=features)
    .add_dataset("train", train)
    .add_dataset("test", test)
    .add_dataset("oot", oot)
    .evaluate(benchmark_dataset="train", display=False)    # train's score bins for every dataset
)
print(fixed[["index", "N", "KS", "AUC", "IV", "N_BINS"]])
```

### Several target labels

`tgt_name` accepts a list or tuple. Each label is evaluated separately and the results are stacked, with a leading
`tgt_name` column. Each label gets its own figure, and `fig_save_path` receives the label as a suffix
(`output/perf.png` becomes `output/perf_bad_dpd30.png`). A single `str` behaves as before, with no `tgt_name` column.

```python
multi = (
    PerformanceEvaluator(tgt_name=["bad_flag", "bad_dpd30"], model=gbm, feature_cols=features)
    .add_dataset("train", train)
    .add_dataset("oot", oot)
    .evaluate(display=False, fig_save_path="output/perf.png")
)
print(multi[["tgt_name", "index", "KS", "AUC"]])
```

### Weighted evaluation

```python
weighted = (
    PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=features, weight_col="sample_wgt")
    .add_dataset("train", train)
    .add_dataset("test", test)
    .evaluate(display=False)
)
print(weighted[["index", "N", "N_RAW", "KS", "AUC", "LIFT", "IV"]])
```

Figures are produced on the weighted path only when `fig_save_path` is set. Datasets can use different weight columns
(`add_dataset(name, data, weight_col=...)`). A dataset registered without one falls back to the evaluator's `weight_col`,
and if that is not set either it is evaluated with unit weights.

### Function form: `get_perf_summary`

`get_perf_summary(train, validation, oot, tgt_name, ...)` evaluates up to three frames in one call (pass `None` for a missing
one). The rows are labeled `ins`, `oos`, and `oot`. It accepts the same `scr_name`, `model`, `feature_cols`, `oot_grp_name`,
`weight_col`, and binning arguments as `PerformanceEvaluator`. For a weighted summary pass a score column through
`scr_name` (see [Where weights are applied](#where-weights-are-applied)).

```python
from Modeling_Tool import get_perf_summary

summary = get_perf_summary(
    train, test, oot, tgt_name="bad_flag", model=gbm, feature_cols=features, display=False,
)
print(summary[["index", "N", "KS", "AUC"]])
```

## 2. Gains tables: `GainsTableCalculator`

A Gains table bins rows by score and reports, for each bin, how many rows, how many bads, the bad rate, the lift, and the
cumulative capture. It is the report form most familiar to business stakeholders.

```python
from Modeling_Tool import GainsTableCalculator

calc = GainsTableCalculator(data=test, dep="bad_flag", score="prob", nbins=10)
gains = calc.calculate()
print(gains[["MIN", "MAX", "N", "PROP", "AVG_BAD", "LIFT", "CUM_BAD_PCT", "KS_PER_BIN", "RANK_ORDER_BUMP"]])
```

The index is `(_bin_num, _bin_range)`. With the default `ascending=False`, bin 0 holds the **highest** scores. Instead of
`score`, you can pass `model` and `varlist` (the feature columns) and the calculator scores the data itself.

| Column | Meaning |
|---|---|
| `MIN`, `MAX` | Lowest and highest score in the bin |
| `N`, `PROP` | Rows in the bin, and their share of all rows |
| `PERF_CNT` | Rows with an observed target |
| `AVG_SCORE`, `UNIQUE_SCORE` | Mean score, and the number of distinct score values |
| `AVG_BAD`, `AVG_GOOD`, `N_BAD`, `N_GOOD` | Bad rate, good rate, and the bad and good counts |
| `LIFT` | `AVG_BAD` divided by the overall bad rate |
| `BAD_PCT_IN_EACH_BIN`, `GOOD_PCT_IN_EACH_BIN` | Share of all bads (goods) that fall in this bin (the capture rate) |
| `N_CUM_BAD`, `N_CUM_GOOD`, `CUM_BAD_PCT`, `CUM_GOOD_PCT` | Cumulative counts and shares down the table |
| `KS_PER_BIN` | `abs(CUM_BAD_PCT - CUM_GOOD_PCT)`; its maximum is the KS statistic |
| `TRUE_BAD_SHIFT` | Relative bad-rate change versus the previous bin |
| `RANK_ORDER_BUMP` | `1` where the bad rate moves against the expected direction (a rank-ordering break) |
| `WOE`, `IV` | Weight of evidence and information value of the bin |

`GainsTableCalculator(data, dep, nbins=10, precision=5, min_bin_prop=0.05, include_missing=True, score=None, model=None, varlist=None, equal_freq=True, chi2_method=False, chi2_p=0.95, init_equi_bins=100, fillna=-999999, spec_values=[], tree_binning=False, random_state=42, ascending=False, weight_col=None, weighted_binning=None)`
holds the configuration, and
`calculate(grp_name=None, min_data_size=100, grp_colname=None, sync_range=True, retSummary=False, withSummary=False, wholeGroup=False, add_func=None, weight_col=None)`
produces the table:

| `calculate` argument | Meaning |
|---|---|
| `grp_name` | Build one Gains table per value of this column and stack them, with the group value in a new column (`grp_colname`, default: `grp_name`). Groups with fewer than `min_data_size` rows are skipped |
| `sync_range` | With `grp_name`, reuse the first group's bin edges for every group so the bins line up (default `True`) |
| `wholeGroup` | With `grp_name`, `sync_range=True`, and `retSummary=True`, take the shared bin edges from all rows instead of the first group |
| `withSummary` | Append a `Grand Summary` row (unweighted tables without `grp_name`) |
| `retSummary` | Return a one-row summary (`N_BUMP`, `MIN_RISK_DEP`, `MAX_RISK_DEP`, `KS_IN_GAINS`, `LIFT_IN_GAINS`, `IV`, `N_BINS`) instead of the table |
| `add_func` | Custom per-bin statistics; see [Custom columns](#custom-columns-with-add_func) |
| `weight_col` | Overrides the calculator's `weight_col` for this call |

The function form `get_gains_table(data, dep, nbins=10, ..., score=None, model=None, varlist=None, ..., grp_name=None, ..., add_func=None, weight_col=None, weighted_binning=None)`
takes the same arguments.

```python
from Modeling_Tool import get_gains_table

by_grade = get_gains_table(
    test, dep="bad_flag", score="prob", nbins=5, grp_name="city_grade", min_data_size=100,
)
print(by_grade[["MIN", "MAX", "N", "AVG_BAD", "LIFT", "city_grade"]].head(8))

summary_row = get_gains_table(test, dep="bad_flag", score="prob", nbins=10, retSummary=True)
print(summary_row)
```

### Weighted Gains table

```python
weighted_gains = GainsTableCalculator(
    data=test, dep="bad_flag", score="prob", weight_col="sample_wgt", nbins=10,
).calculate()
print(weighted_gains[["MIN", "MAX", "N", "N_RAW", "AVG_BAD", "LIFT", "KS", "AUC"]])
```

The weighted table has bins `1` to `nbins` (bin 1 holds the highest scores), each carrying about `1 / nbins` of the total
weight, and extra `N_RAW`, `KS`, and `AUC` columns. See [Where weights are applied](#where-weights-are-applied) for what the
weighted path ignores.

### Custom columns with `add_func`

`add_func` receives the rows of one bin as a DataFrame (all columns of your data, plus the bin columns `_bin_num` and
`_bin_range`) and returns a `Series`. Its values become extra columns on the Gains table. It is applied on the unweighted
path only.

```python
def bin_profile(group):
    return pd.Series({
        "mean_income": group["income"].mean(),
        "overdue_rate": (group["n_overdue"] > 0).mean(),
    })


gains_plus = GainsTableCalculator(data=test, dep="bad_flag", score="prob", nbins=10).calculate(add_func=bin_profile)
print(gains_plus[["N", "AVG_BAD", "mean_income", "overdue_rate"]])
```

`get_gains_table_by_cust_metrics` is the column-name version: it adds the aggregate (`metric_agg_func`, default `"mean"`) of
each column in `eval_metrics` to a compact Gains table.

```python
from Modeling_Tool import get_gains_table_by_cust_metrics

cust = get_gains_table_by_cust_metrics(
    data=test, dep="bad_flag", score="prob", nbins=10,
    eval_metrics=["income", "n_overdue"], metric_agg_func="mean",
)
print(cust)
```

!!! warning "Defaults tied to a legacy dataset"

    `eval_metrics` defaults to `["age", "monthly_income", "education"]`, so always pass it explicitly. This function has no
    `weight_col`, and its default `ascending=True` differs from `GainsTableCalculator` (`False`): bin 0 holds the lowest
    scores.

## 3. Challenger versus champion: `Model_Evaluation_Tool`

`Model_Evaluation_Tool` evaluates several score columns on the same frame. `base_score` names the champion and
`comp_scrlist` lists the challengers; the comparison methods report one block per score, labeled in a `score_name` column.

```python
from Modeling_Tool import Model_Evaluation_Tool

m_eval = Model_Evaluation_Tool(
    data=holdout,
    dep="bad_flag",
    base_score="score_old",
    comp_scrlist=["prob"],
)

compare = m_eval.model_perf_compare()
print(compare[["score_name", "index", "N", "KS", "AUC", "Top10%_TargetRate"]])

print(m_eval.get_score_correlation())     # long format: base, compare, corr

gains_by_score = m_eval.get_gains_summary(disp=False)
print(gains_by_score[["score_name", "_bin_num", "N", "AVG_BAD", "LIFT"]].head())
```

!!! note "Positive scores only"

    `model_perf_compare` keeps only rows whose scores are greater than 0 when `positive_score_only=True` (the default);
    pass `False` if your scores can legitimately be zero or negative. `get_score_correlation`, `get_cross_risk_summary`,
    and `cross_perf_eval` always drop rows with a score of 0 or below, because 0 and negative values conventionally mark
    "no score".

| Method | Returns |
|---|---|
| `model_perf_compare(data=None, grp_name=None, dist_bins=100, pct_bins=10, min_data_size=50, sync_data_size=True, min_bin_prop=None, include_missing=None, equal_freq=None, sample_name=None)` | One performance row per score (the `PerformanceEvaluator` columns plus `score_name`); `sample_name` labels the `index` column (default `'all'`); `grp_name` adds a per-group evaluation |
| `get_gains_summary(data=None, grp_name=None, disp=True, grp_disp_metric=None, grp_nbins=5, withSummary=True, add_func=None, sync_range=True, spec_values=None, include_missing=None, fillna=None)` | Gains tables for every score, stacked, with `score_name`. Without `add_func`, only the columns in `gains_display_metric_list` are kept; with `add_func`, all columns plus yours |
| `get_score_correlation(score_list=None, method='pearson')` | Long-format correlation table with `base`, `compare`, `corr` |
| `get_base_score(scorename='_base_model_score_', disp=False)` | Returns `self` after adding the fitted `model`'s probability as the base-score column (the model needs `feature_names_in_`) |
| `get_cross_risk_summary(cross_agg_dict=None, nbins=5, equal_freq=None, disp=True, spec_values=None, binning_numeric=None)` | Base-score × challenger cross matrices (see below) |
| `cross_perf_eval(bad_list, scr_list, data=None, eval_metric=None, melt=True)` | `AUC`, `KS`, `N` for every label × score pair |
| `run_variable_analysis_summary(varlist, ...)` | Per-variable KS, lift, IV, and missing rate |
| `multi_subset_wrapper`, `multi_ylabel_wrapper`, `multi_group_wrapper`, `multi_dim_eval` | Repeat an evaluation over subsets, labels, and groups |
| `pipe(data=None)` | An [`EvaluationPipeline`](#5-slice-by-slice-evaluation-evaluationpipeline) on this object |

```python
cross_perf = m_eval.cross_perf_eval(bad_list=["bad_flag", "bad_dpd30"], scr_list=["score_old", "prob"])
print(cross_perf)       # columns: score_name, variable (e.g. AUC_bad_flag), value
```

### Configure the legacy defaults

Several constructor defaults come from the library author's production data. **Override them whenever you use the
methods that read them:**

| Constructor argument | Legacy default | Read by |
|---|---|---|
| `cross_agg_dict` | Columns `is_dpd7`, `credit_limit`, `monthlyincome`, `education` | `get_cross_risk_summary` |
| `subset_condition_dict` | Queries on `aprvvrsn_2` | `multi_subset_wrapper`, `multi_dim_eval` |
| `eval_ylabels` | `['is_dpd7']` | `multi_ylabel_wrapper`, `multi_dim_eval` |
| `grp_namelist` | `['sample_ind_fnl', 'aprvvrsn_2', 'week_start_date']` | `multi_dim_eval` |

`get_cross_risk_summary` additionally needs a unique-id column named **`flow_id`**: it appends a count of `flow_id` to every
cell.

```python
cross = m_eval.get_cross_risk_summary(
    cross_agg_dict={"bad_flag": ["count", lambda x: x.mean()]},    # {column: aggregation(s)}
    nbins=5,
    disp=False,
)
print(cross.head())    # long format: base_scr_range, eval_metric, score_name, comp_scr_range, value

grid = Model_Evaluation_Tool(
    data=holdout,
    dep="bad_flag",
    base_score="score_old",
    comp_scrlist=["prob"],
    subset_condition_dict={"Overall": "", "Top cities": "city_grade in ['A', 'B']"},   # label -> query ("" = all rows)
    eval_ylabels=["bad_flag", "bad_dpd30"],
    grp_namelist=["apply_month"],
    min_data_size=50,
)
multi_dim = grid.multi_dim_eval()
print(multi_dim[["score_name", "eval_subset", "eval_ylabel", "group_name", "group_value", "N", "AUC", "KS"]].head())
```

### Weighted comparison

```python
weighted_eval = Model_Evaluation_Tool(
    data=holdout, dep="bad_flag", base_score="score_old", comp_scrlist=["prob"], weight_col="sample_wgt",
)
print(weighted_eval.model_perf_compare()[["score_name", "N", "N_RAW", "KS", "AUC"]])
```

## 4. Cross-risk matrix: `cross_risk`

`cross_risk(data, score_list, dep, nbins, agg_col=None, ..., agg_func='mean', ..., weight_col=None, sample_weight=None)`
bins two scores and aggregates a column inside every cell. With the defaults, the cells hold the **bad rate** (the mean of
`dep`) and the margins are labeled `Total_Avg_Risk`. This is the usual way to see where a new score disagrees with an old one.

```python
from Modeling_Tool import cross_risk

risk_matrix = cross_risk(
    data=holdout,
    score_list=["score_old", "prob"],     # exactly two score columns
    dep="bad_flag",
    nbins=5,                              # an int, or a list with one value per score
)
print(risk_matrix)

# Any other column and aggregation: here, the mean income per cell, with weights
income_matrix = cross_risk(
    data=holdout, score_list=["score_old", "prob"], dep="bad_flag", nbins=5,
    agg_col="income", agg_func="mean", weight_col="sample_wgt",
)
```

`agg_func` accepts `'mean'`, `'sum'`, `'count'`, or a callable, as in `pandas.crosstab`. It also supports a ratio of two
columns: `agg_col=(numerator, denominator), agg_func='ratio'`.

## 5. Slice-by-slice evaluation: `EvaluationPipeline`

`EvaluationPipeline(m_eval, data=None)` runs a function once per slice. Chain `group_by` and `subset_by`, then call
`apply(func)`:

- `group_by(group_name, min_size=100, group_var_name=None)` splits by the values of a column.
- `subset_by(condition_dict, name='eval_subset', min_size=100)` filters by `{label: query}`, where each query is a
  `DataFrame.query` string and `""` keeps every row.
- Slices with fewer than `min_size` rows are skipped, and the slice labels are appended to the result as columns.

`func` either takes a `current_data` argument and receives the slice, or takes no data argument and is a method of your
`Model_Evaluation_Tool`, which the pipeline points at the slice while it runs.

```python
from Modeling_Tool import EvaluationPipeline

pipeline = (
    EvaluationPipeline(m_eval)
    .group_by("apply_month", min_size=100)
    .subset_by({"All cities": "", "Top cities": "city_grade in ['A', 'B']"}, name="city_subset", min_size=100)
)


def slice_metrics(current_data):
    return pd.DataFrame({
        "n": [len(current_data)],
        "bad_rate": [current_data["bad_flag"].mean()],
        "avg_score": [current_data["prob"].mean()],
    })


by_slice = pipeline.apply(slice_metrics)
print(by_slice)

# Reuse a method of m_eval on every month
monthly_perf = EvaluationPipeline(m_eval).group_by("apply_month", min_size=100).apply(m_eval.model_perf_compare)
print(monthly_perf[["apply_month", "score_name", "N", "KS", "AUC"]])
```

!!! warning "Errors inside `func` are swallowed"

    An exception raised by `func` is logged at INFO level and that slice yields an empty result. If the output is
    unexpectedly empty, call `func` on a single slice to see the real error.

## 6. Figures: `evaluate_performance` and `comparison_performance`

Both functions take a dictionary of datasets and return a summary DataFrame. `save_path` is the **image file** to write
(for example `output/perf.png`; the folder must exist). Pass `to_show=False` in scripts.

```python
from Modeling_Tool import evaluate_performance, comparison_performance

# One score, several datasets: ROC, score distribution, percentile, and gain panels per dataset
perf_summary = evaluate_performance(
    datasets={
        "test": {"y_true": test["bad_flag"], "y_score": test["prob"]},
        "oot": {"y_true": oot["bad_flag"], "y_score": oot["prob"]},
    },
    to_show=False,
    save_path="output/perf.png",
)
print(perf_summary)       # columns: index, N, avgTrue, avgScore, KS, AUC, Btm10%_TargetRate, Top10%_TargetRate

# Weighted: add a 'sample_weight' key to each dataset
weighted_summary = evaluate_performance(
    datasets={"test": {"y_true": test["bad_flag"], "y_score": test["prob"], "sample_weight": test["sample_wgt"]}},
    to_show=False,
    save_path="output/perf_weighted.png",
)

# Several scores on the same labels: y_score_dict maps a model name to its scores (unweighted only)
comparison = comparison_performance(
    datasets={
        "test": {"y_true": test["bad_flag"], "y_score_dict": {"gbm": test["prob"], "old": test["score_old"]}},
        "oot": {"y_true": oot["bad_flag"], "y_score_dict": {"gbm": oot["prob"], "old": oot["score_old"]}},
    },
    to_show=False,
    save_path="output/perf_compare.png",
)
print(comparison)         # columns: model, KS, AUC, Btm10%_TargetRate, Top10%_TargetRate, N, avgTrue, dataset
```

`evaluate_performance(datasets, dist_bins=20, pct_bins=10, square_figsize=5, fontdicts=..., to_show=True, save_path=None, gains_table=True, equal_freq=True, pct_bin_edges=None, sample_weight=None, ascending=None)`
draws one row of four panels per dataset. `comparison_performance(datasets, pct_bins=10, square_figsize=5, fontdicts=..., to_show=True, save_path=None)`
draws one row of three panels (ROC, percentile bad rate, cumulative bad rate) per dataset.

### Building blocks

The curve tables behind the figures are public. Each takes `sample_weight=` for the weighted version.

```python
from Modeling_Tool import calc_roc, calc_pr, calc_equid_pct
from Modeling_Tool.Eval import summarize_pct
from Modeling_Tool.Eval.evaluate_model import summarize_roc

roc = calc_roc(test["bad_flag"], test["prob"])                      # fpr, tpr, thresholds, thresholds_percentile
print(summarize_roc(roc))                                           # {'auc': ..., 'ks_index': ..., 'ks_threshold': ..., 'ks': ...}

pr = calc_pr(test["bad_flag"], test["prob"])                        # precision, recall, thresholds

deciles = calc_equid_pct(test["bad_flag"], test["prob"], bins=10)   # equal-frequency table with lift and gain
print(deciles[["min_score", "max_score", "n", "avg_true", "lift", "gain"]].head(3))
print(summarize_pct(deciles))     # {'pct_bins': ..., 'pct_interval': ..., 'pct_top_avgTrue': ..., 'pct_btm_avgTrue': ...}
```

| Function | Returns |
|---|---|
| `calc_roc(y_true, y_score, sample_weight=None)` | ROC table: `fpr`, `tpr`, `thresholds`, `thresholds_percentile` (the weighted version adds `FPR`, `TPR`, and `KS`) |
| `calc_pr(y_true, y_score, sample_weight=None)` | Precision-recall table: `precision`, `recall`, `thresholds` |
| `calc_equid_dist(y_true, y_score, y_group=None, bins=10, sample_weight=None)` | Equal-width score-bin table |
| `calc_equid_pct(y_true, y_score, y_group=None, bins=10, ascending=True, sample_weight=None)` | Equal-frequency table with `lift` and `gain`; row 0 holds the lowest scores when `ascending=True` |
| `calc_fixed_pct(y_true, y_score, y_group=None, bin_edges=None, ascending=True, sample_weight=None)` | Table over the bin edges you supply |

The weighted versions of the three binning functions return a Gains-style table (`N`, `N_RAW`, `AVG_BAD`, `LIFT`, ...) instead.
`summarize_roc` (in `Modeling_Tool.Eval.evaluate_model`), `summarize_pr`, and `summarize_pct` (in `Modeling_Tool.Eval`) reduce
these tables to a dictionary of headline numbers. The `plot_roc_curve`, `plot_ks_curve`, `plot_pr_curve`, `plot_kde_curve`,
`plot_dist_curve`, `plot_cumdist_curve`, `plot_pct_curve`, `plot_cumpct_curve`, and `plot_gain_curve` functions draw
individual panels from the tables; import them from `Modeling_Tool.Eval`.

## 7. Lift cut-offs: `calc_lift_apt`

`calc_lift_apt(y_true, y_score, start, stop, step, score_ascending=True, sample_weight=None)` answers "what score cut-off
captures a lift of at least 1.5×, 2×, 2.5×, 3×?". `start`, `stop`, and `step` are required.

```python
from Modeling_Tool import calc_lift_apt

lift_table = calc_lift_apt(test["bad_flag"], test["prob"], start=1.5, stop=3.0, step=0.5)
print(lift_table)
```

Each row is a target `lift`, the `lift_actual` achieved, the score cut-off (`lower_limit`: rows scoring at least this
value form the group), and that group's cumulative size (`cumsum_n`, `cumsum_proportion`), bad count (`cumsum_true`), and
bad rate (`cumavg_true`).

!!! note "Weights"

    With integer-valued weights the rows are replicated and the same table is returned. With fractional weights the function
    returns a one-dimensional NumPy array holding, for each target lift, the closest lift the weighted Gains table can
    achieve, not a table.

## Troubleshooting

??? question "`ModuleNotFoundError: No module named 'IPython'` from `evaluate()`"

    `display=True` is the default and imports `IPython.display`. Pass `display=False` outside notebooks.

??? question "`KeyError: 'is_dpd7'` or `'flow_id'` from `Model_Evaluation_Tool`"

    You used a method that reads a legacy default. Pass `cross_agg_dict`, `subset_condition_dict`, `eval_ylabels`, and
    `grp_namelist` for your own columns, and make sure the frame has a `flow_id` column for `get_cross_risk_summary`. See
    [Configure the legacy defaults](#configure-the-legacy-defaults).

??? question "My weighted results look unweighted (no `N_RAW`)"

    Weights are not applied together with `oot_grp_name`, `benchmark_dataset`, or `grp_name`. See
    [Where weights are applied](#where-weights-are-applied).

??? question "AUC is below 0.5"

    The score runs the wrong way: a higher score must mean a higher chance of the bad outcome. See the score-direction
    note at the top of this page.

??? question "AUC and KS disagree (high AUC, low KS)"

    AUC summarizes the whole ranking, while KS is the single largest gap between the cumulative bad and good shares. They can
    diverge when the separation is concentrated in a narrow score range. Check `KS_PER_BIN` and `LIFT` in the Gains table to
    see where the separation comes from.

??? question "OOT AUC is far lower than on the test set"

    Check for population drift first: compute the PSI of the model's inputs between the development and OOT samples
    (see [Feature Screening](feature.md)); a PSI above 0.25 is conventionally treated as significant drift. If the inputs
    are stable, the relationship itself may have changed and the model needs a more recent training window.

??? question "`N` and `N_RAW` differ a lot. Which one should I report?"

    Business metrics (bad rate, lift, capture rate) are based on `N`, the sum of weights. `N_RAW` is the number of rows and
    answers "how many accounts are in this bin?". If every weight is 1, `N == N_RAW`.

??? question "I trained with weights but forgot to pass them at evaluation"

    Evaluation then treats every row equally, which no longer matches the population the model was fitted on, and AUC, KS,
    and Lift can look different from training time. Pass the same weights to every evaluation entry point.
