# Model Evaluation

SuperModelingFactory provides tools in the [`Eval`](../api/eval.md) subpackage for **Gains tables / performance summaries / ROC-KS-PR plots / chained evaluation pipelines**.

## Sample-Weighted Evaluation

After you pass `weight_col` (or `sample_weight` for the low-level functions), the Gains table, performance summaries, and ROC/KS/Lift
metrics are all aggregated by weight. **Without weights, behavior is exactly the same as before** (backward compatible).

!!! info "Native API"

    Since the main repository merged [PR #25](https://github.com/Kyle-J-Sun/SuperModelingFactory/pull/25),
    the weighting logic has been built into `Modeling_Tool.Eval`, with no runtime patch needed.

### N and N_RAW

In Gains tables and binned aggregation outputs, sample size is counted two ways:

| Column | Meaning | Typical use |
|----|------|----------|
| `N` | **Sum of weights** in the bin (effective sample size) | Weighted bad rate, Lift, cumulative capture rate |
| `N_RAW` | **Number of rows** in the bin (raw record count) | Auditing, reconciling raw account/transaction counts |

For example: when each row is an account and the weight is the loan balance, `N_RAW` is the number of accounts and `N` is the total balance.
In the weighted case, bad rate = `sum(y * w) / sum(w)`, and Lift = the bin's weighted bad rate / the overall weighted bad rate.

### Weighted Metric Semantics

| Metric | Weighted behavior |
|------|----------|
| `AUC` | Passed through to sklearn `roc_auc_score(..., sample_weight=...)` |
| `KS` | Maximum TPR−FPR on the weighted ROC curve |
| `Gini` | `2 * AUC - 1` (based on the weighted AUC) |
| `Top10%_TargetRate` | **Weighted** bad-sample share within the top 10% score band |
| `Top10%_Lift` | **Weighted** lift multiple of the top 10% score band |
| Gains bad rate / WOE / IV | Aggregated by `N` (sum of weights) |

### Weighted Evaluation Images

When `fig_save_path` is set, the images and the summary table use exactly the same weighting basis. Each dataset reads
the weight it registered through `add_dataset(..., weight_col=...)`; the ROC, score distribution/KDE,
Percentile, and Gain panels are all computed with that weight, and the title is marked `Weighted`.
If the weights of an OOS/OOT dataset are all 1, the weighted and unweighted images have identical values.

## 1. Multi-Dataset Evaluation — `PerformanceEvaluator`

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
    weight_col="sample_wgt",   # optional: the DataFrame of each add_dataset must have this column
)

perf = (
    evaluator.add_dataset("train", train_woe)
              .add_dataset("test",  test_woe)
              .add_dataset("oot",   oot_woe)
              .evaluate()
)

print(perf[["index", "KS", "AUC", "Top10%_TargetRate"]])
```

### Output Metrics

| Metric | Meaning |
|------|------|
| `KS` | Kolmogorov-Smirnov (maximum TPR-FPR) |
| `AUC` | Area under the ROC curve |
| `Gini` | `2*AUC - 1` |
| `Top10%_TargetRate` | Bad-sample share within the top 10% score band |
| `Top10%_Lift` | Lift multiple of the top 10% score band |
| `AvgScore` | Average score |

### Multi-y-Label Comparison

`tgt_name` accepts a **list/tuple of multiple y labels**, evaluating each label separately:

- **Table**: a new `tgt_name` column is added, and the results for each label are **stacked vertically** into one table;
- **Images**: **one image per label**, displayed in a loop when `to_show=True`; `fig_save_path` automatically gets a label suffix (`perf.png` → `perf_<label>.png`).

```python
perf = (
    PerformanceEvaluator(
        tgt_name=["bad_dpd7", "bad_dpd30"],   # multiple y labels
        model=gbm._model.model,
        feature_cols=woe_features,
        weight_col="sample_wgt",
    )
    .add_dataset("train", train_woe)
    .add_dataset("oot",   oot_woe)
    .evaluate(to_show=True)                    # one image per label
)

# The output contains a tgt_name column, with each label stacked vertically
print(perf[["tgt_name", "index", "KS", "AUC"]])
```

> Passing a single `str` (such as `tgt_name="bad_flag"`) behaves as before, and the output has no `tgt_name` column (backward compatible).

`Model_Evaluation_Tool.model_perf_compare()` is a high-level wrapper around `PerformanceEvaluator` that compares OOT performance in batch using `base_score + comp_scrlist` and appends a `score_name` column.

## 2. Gains Table — `GainsTableCalculator`

A gains table binned by score, the report form most familiar to business stakeholders.

```python
from Modeling_Tool import GainsTableCalculator

gains = GainsTableCalculator(
    data=test_woe,
    score="prob",       # score column
    dep="bad_flag",
    weight_col="sample_wgt",     # optional: weighted Gains
    weighted_binning=True,       # True = equal-frequency bins by cumulative weight; False = by row count (default)
    nbins=10,
)
gains_table = gains.calculate()
print(gains_table)   # contains N (sum of weights) and N_RAW (row count)
```

`get_gains_table` / `get_gains_table_by_cust_metrics` also accept `weight_col`;
when weighted, the bad/good counts, bad rate, Lift, KS, WOE, and IV are all computed by weight.

`Model_Evaluation_Tool.get_gains_summary()` uses `GainsTableCalculator` internally to generate gains tables for multiple scores in batch, and supports injecting custom columns through `add_func`.

## 3. Custom Metrics — `add_func`

`Model_Evaluation_Tool.get_gains_summary(add_func=...)` and `GainsTableCalculator.calculate(add_func=...)` share the same extension point: append custom statistics columns on the binned grouped data.

```python
from Modeling_Tool import Model_Evaluation_Tool, GainsTableCalculator

def mean_income(group):
    return pd.Series({"mean_income": group["income"].mean()})

m_eval = Model_Evaluation_Tool(
    data=test_woe,
    dep="bad_flag",
    base_score="prob",
    comp_scrlist=["prob"],
)
gains = m_eval.get_gains_summary(add_func=mean_income)

# Or call the calculator directly
gains = GainsTableCalculator(
    data=test_woe,
    score="prob",
    dep="bad_flag",
    nbins=10,
).calculate(add_func=mean_income)
```

The function-level API `get_gains_table_by_cust_metrics` can still be used for batch aggregation by column name:

```python
from Modeling_Tool import get_gains_table_by_cust_metrics

gains = get_gains_table_by_cust_metrics(
    data=test_woe,
    score="prob",
    dep="bad_flag",
    weight_col="sample_wgt",
    nbins=10,
    eval_metrics=["income", "n_overdue"],
    metric_agg_func="mean",
)
```

## 4. Cross-Risk Matrix — `cross_risk`

Joint risk evaluation by **two score binnings** (typical scenario: comparing a new model against an old model).

```python
from Modeling_Tool import cross_risk

risk_matrix = cross_risk(
    data=test_woe,
    score_list=["score_old", "score_new"],
    dep="bad_flag",
    weight_col="sample_wgt",
    nbins=5,
)
print(risk_matrix)
```

## 5. Chained Evaluation Pipeline — `EvaluationPipeline`

Run a custom function after grouping/subsetting by conditions:

```python
from Modeling_Tool import EvaluationPipeline, Model_Evaluation_Tool

m_eval = Model_Evaluation_Tool(
    data=test_woe,
    dep="bad_flag",
    comp_scrlist=["prob"],
    weight_col="sample_wgt",
)

pipeline = (
    EvaluationPipeline(m_eval)
    .group_by("apply_month", min_size=100)        # group by month
    .subset_by({"city_grade": ["A", "B"]}, name="top_city")  # filter to the top cities
)

def per_group_metrics(current_data):
    return current_data.groupby("apply_month").agg(
        bad_rate=("bad_flag", "mean"),
        avg_score=("prob", "mean"),
        n=("bad_flag", "size"),
    )

result = pipeline.apply(per_group_metrics)
```

## 6. ROC / KS / PR / KDE Plots — `evaluate_model.py`

Draw images directly to local disk:

```python
from Modeling_Tool import evaluate_performance, comparison_performance

# Single model: datasets = {dataset name: {'y_true':, 'y_score':, 'sample_weight': (optional)}}
evaluate_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score": test_woe["prob"],
            "sample_weight": test_woe["sample_wgt"],   # optional
        }
    },
    to_show=False,
    save_path="./output/perf/",
)

# Multi-model comparison: in each dataset, use y_score_dict to hold the scores of several models
comparison_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score_dict": {"lgb": test_woe["prob_lgb"], "lr": test_woe["prob_lr"]},
            "sample_weight": test_woe["sample_wgt"],
        },
        "oot": {
            "y_true": oot_woe["bad_flag"],
            "y_score_dict": {"lgb": oot_woe["prob_lgb"], "lr": oot_woe["prob_lr"]},
            "sample_weight": oot_woe["sample_wgt"],
        },
    },
    to_show=False,
    save_path="./output/perf_compare/",
)
```

`calc_pr` / `calc_roc` also accept `sample_weight`, passed through to sklearn.
Binned aggregations (`calc_equid_dist` / `calc_equid_pct` / `calc_fixed_pct`) output two columns when weights are provided:
`n` (sum of weights) and `n_raw` (row count).

The generated plots include:

- ROC curve (with AUC)
- KS curve
- PR curve
- Score KDE distribution
- Cumulative distribution plot
- Gain / Lift plots

## 7. Lift Table — `calc_lift_apt`

```python
from Modeling_Tool import calc_lift_apt

lift_table = calc_lift_apt(
    y_true=test_woe["bad_flag"],
    y_score=test_woe["prob"],
    sample_weight=test_woe["sample_wgt"],   # optional
    start=1.5, stop=3.0, step=0.5,   # lift threshold range and step
)
print(lift_table)
```

## Complete Evaluation Pipeline

```python
from Modeling_Tool import (
    PerformanceEvaluator, GainsTableCalculator,
    get_gains_table_by_cust_metrics, evaluate_performance,
)

# 1) Multi-dataset summary
perf = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
    weight_col="sample_wgt",
).add_dataset("train", train_woe) \
 .add_dataset("test",  test_woe).evaluate()

# 2) Gains table + custom metrics
gains = get_gains_table_by_cust_metrics(
    test_woe, score="prob", dep="bad_flag", nbins=10,
    weight_col="sample_wgt",
    eval_metrics=["income"], metric_agg_func="mean",
)

# 3) Output plots
evaluate_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score": test_woe["prob"],
            "sample_weight": test_woe["sample_wgt"],
        }
    },
    to_show=False,
    save_path="./output/perf/",
)
```

## FAQ

??? question "AUC and KS disagree (high AUC, low KS)"

    This usually means **scores are concentrated in the middle range**. Check how many bad samples the top 10% score band covers.

??? question "OOT AUC is far lower than the test set"

    PSI > 0.25 indicates significant distribution drift, so you need to:

    1. Check the PSI of the model input variables
    2. Retrain if necessary (using samples from a more recent window)

??? question "N and N_RAW differ a lot in the Gains table — which one should I look at?"

    Business metrics (bad rate, Lift, capture rate) are based on `N` (sum of weights).
    `N_RAW` reflects only the raw row count, and is used to check "how many accounts/transactions are in this bin".
    If each row is one account and all weights are 1, then `N == N_RAW`.

??? question "What happens if I trained with weights but forget to pass them at evaluation?"

    Evaluation defaults to equal weights (weight 1 per row), which is inconsistent with the target distribution of the weighted-trained model
    and may leave you with a skewed picture of AUC/KS/Lift compared with training time. Pass the same `weight_col` consistently to `PerformanceEvaluator`,
    `GainsTableCalculator`, and similar entry points.
