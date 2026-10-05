# Model Explainability

`ModelExplainer` explains a fitted credit model with six methods behind one object: SHAP, Owen value, PDP, ICE, ALE, and
LIME. It lives in [`Modeling_Tool.Explainability`](../api/explainability.md) and is also importable from `Modeling_Tool`.

Use it to answer the three questions a model review asks: which variables drive the score (global importance), why one
applicant received a score (local reason codes), and how the score responds when a variable changes (effect curves).

!!! note "Install the optional dependencies"

    SHAP, Owen value, and LIME need the `explain` extra. PDP, ICE, and ALE need nothing beyond SMF's core dependencies.

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

    `shap` and `lime` are imported lazily, when the first SHAP, Owen, or LIME call runs. If a package is missing, that call
    raises an `ImportError` that names the extra.

## Methods at a glance

| Method | Scope | Question it answers | Needs |
|---|---|---|---|
| SHAP | Global and local attribution | Which variables contribute most? Why did this applicant score high? | `shap` |
| Owen value | Group attribution | How much do business modules such as delinquency, multi-lending, and affordability contribute? | `shap` |
| PDP | Global average effect | When a variable increases, how does the average predicted risk change? | none |
| ICE | Individual response curves | Do different applicants respond to a variable in the same way? | none |
| ALE | Accumulated local effects | What is the effect of a variable when features are strongly correlated? | none |
| LIME | Local surrogate model | Near one applicant, which variables explain the score? | `lime` |

## Supported models

`ModelExplainer` accepts a `GradientBoostingModel`, an `LRMaster`, or a fitted estimator with `predict_proba`. It picks the
SHAP explainer from the model family:

| Model | SHAP explainer | Unit of SHAP values | `background_data` |
|---|---|---|---|
| LightGBM, XGBoost | `TreeExplainer` | log-odds | optional |
| `LRMaster`, `LogisticRegression` | `LinearExplainer` | log-odds | required |
| Anything else (CatBoost, random forest, ...) | Model-agnostic `shap.Explainer` over `predict_proba` | probability | required; slow with many features |

PDP, ICE, ALE, and LIME call `predict_proba`, so their outputs are in probability units for every model.

!!! warning "`LRMaster(standardize=True)`"

    The explainer unwraps the scikit-learn model inside `LRMaster` and skips its scaler, so unscaled inputs produce wrong
    explanations. Pass frames scaled with the fitted scaler for both `background_data` and `X`, for example
    `pd.DataFrame(lr.standardizer.transform(df[features]), columns=features)`. Models trained without `standardize` need
    no such step.

## Example data

Every snippet on this page runs top to bottom in one Python session. The setup below creates numeric features in three
correlated groups (delinquency, multi-lending, affordability) plus one independent variable, and trains a LightGBM model.
Any numeric feature frame works the same way, for example the `train_woe[woe_features]` columns from the
[Quickstart](../quickstart.md).

```python
import numpy as np
import pandas as pd

from Modeling_Tool import GradientBoostingModel, ModelExplainer, SampleSplitter, build_coalition_structure

rng = np.random.default_rng(42)
n = 5000
dpd, loans, capacity = rng.normal(size=(3, n))      # three latent risk drivers


def noisy(base):
    return base + 0.35 * rng.normal(size=n)


data = pd.DataFrame({
    "max_dpd_12m": noisy(dpd), "dpd_cnt_6m": noisy(dpd), "ever_dpd30": noisy(dpd),
    "inquiries_3m": noisy(loans), "inquiries_6m": noisy(loans), "active_loans": noisy(loans),
    "monthly_income": noisy(capacity), "debt_to_income": noisy(capacity), "monthly_obligation": noisy(capacity),
    "age": rng.normal(size=n),
})
logit = -1.6 + 0.9 * dpd + 0.5 * loans - 0.6 * capacity + 0.2 * data["age"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = [c for c in data.columns if c != "bad_flag"]
train, test = SampleSplitter(test_size=0.3, random_state=42, stratify=True).split_df(data, target="bad_flag")

gbm = GradientBoostingModel("lgb", {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "early_stopping_rounds": 20,
    "eval_metric": "auc",
    "verbose": -1,
    "n_jobs": 1,                  # one thread is fastest on a sample this small
})
gbm.fit(train[features], train["bad_flag"], test[features], test["bad_flag"])
```

## 1. Create an Explainer

```python
explainer = ModelExplainer(gbm, background_data=train[features].sample(100, random_state=42))
```

`ModelExplainer(model, feature_names=None, model_type=None, background_data=None)`:

| Argument | Meaning |
|---|---|
| `model` | A `GradientBoostingModel`, an `LRMaster`, or a fitted estimator |
| `feature_names` | Column order the model expects. Inferred when the model records its feature names; pass it for estimators that do not |
| `model_type` | `"lgb"`, `"xgb"`, `"lr"`, or any other string. Detected automatically; override it only if detection is wrong |
| `background_data` | A representative sample of the training features; a few hundred rows is enough. Optional for LightGBM and XGBoost SHAP, required for LR, Owen, and the model-agnostic SHAP path. It is also the default `X_train` for LIME and the default data for the coalition structure |

The frames you pass to later methods may contain extra columns: the explainer selects the model's features by name.

## 2. SHAP Attribution

```python
explainer.explain(test[features])                      # computes and caches SHAP values

importance = explainer.feature_importance(normalize=True)
print(importance[["feature", "mean_abs_shap", "importance_pct"]].head(10))

local = explainer.explain_instance(test[features].iloc[[0]])
print(local)
print("base value:", local.attrs["base_value"])

explainer.summary_plot(show=False, save_path="shap_summary.png")
explainer.summary_plot(plot_type="bar", show=False, save_path="shap_bar.png")
explainer.dependence_plot("max_dpd_12m", show=False, save_path="shap_dependence.png")
```

| Method | Returns |
|---|---|
| `explain(X)` | A `shap.Explanation`. It also caches `shap_values_`, `expected_value_`, and `explanation_` on the explainer |
| `feature_importance(X=None, normalize=False)` | `feature`, `mean_abs_shap`, sorted descending. `normalize=True` adds `importance_pct`, each feature's share of the total (the shares sum to 1, not 100). With `X` it recomputes the SHAP values; without `X` it reads the cache and raises `RuntimeError` if there is none |
| `explain_instance(x_row)` | `feature`, `value`, `shap_value` for one applicant, sorted by absolute SHAP value. `x_row` is a one-row DataFrame, a `Series`, or a `dict`; only the first row is used. `.attrs["base_value"]` holds the baseline |
| `summary_plot(X=None, max_display=20, plot_type="dot", show=True, save_path=None, random_state=0)` | A `matplotlib` figure. `plot_type` is for example `"dot"` (beeswarm, the default) or `"bar"`; `random_state` seeds the point jitter |
| `dependence_plot(feature, X=None, interaction_index="auto", show=True, save_path=None)` | A `matplotlib` figure for one feature |

For tree and linear models the SHAP values are additive in the model's log-odds: `base_value + sum(shap_value)` equals the
model's raw score for that row. Plot methods write the figure with `dpi=150` to `save_path`, whose folder must already
exist. `show=False` closes the figure after saving, which is what you want in scripts and jobs.

## 3. Owen Value: Group Attribution

Plain SHAP treats every feature as an independent player. When several variables carry the same risk signal, each one
looks less important than the signal really is. The Owen value first places features in coalitions and then splits each
coalition's credit among its members, so the result reads at the business-module level: "delinquency pushed the score up
by 0.4 log-odds" rather than three small contributions.

### 3.1 Build the Coalition Structure

```python
prior_groups = {
    "delinquency": ["max_dpd_12m", "dpd_cnt_6m", "ever_dpd30"],
    "multi_lending": ["inquiries_3m", "inquiries_6m", "active_loans"],
    "affordability": ["monthly_income", "debt_to_income", "monthly_obligation"],
}

coalitions = build_coalition_structure(
    train[features],
    prior_groups=prior_groups,
    threshold=0.35,
    method="complete",
    corr_method="spearman",
)
print(coalitions["summary"][["n_features", "mean_abs_corr", "max_abs_corr"]])
```

`build_coalition_structure(X, prior_groups=None, threshold=0.35, method="complete", corr_method="spearman", min_group_size=1, intra_dist=0.01, inter_dist=0.99)`
combines your business groups with data-driven clustering:

1. Business priors win. `prior_groups` maps a group name to its features. Names that are not columns of `X` are ignored,
   but a feature that is in `X` cannot appear in two groups (`ValueError`).
2. Features outside every prior group fall back to automatic clustering on the distance `1 - |association|`. The leftover
   features of an automatic cluster form a group named `residual_auto_cluster_<k>`. Without priors, the groups are named
   `auto_cluster_<k>`.

| Argument | Meaning |
|---|---|
| `X` | Numeric `DataFrame` of the model features |
| `prior_groups` | `{group: [features]}`, or `None` for fully automatic groups. `CREDIT_PRIOR_GROUPS` (also exported from `Modeling_Tool`) is a ready-made example for common credit variable names |
| `threshold` | Cut height on the distance `1 - \|association\|`, between 0 and 1. With `method="complete"`, every pair in an automatic group has `\|corr\| >= 1 - threshold` (0.65 at the default). Raise it for looser, larger groups; lower it for tighter ones |
| `method` | Linkage: `"complete"`, `"average"`, or `"single"` |
| `corr_method` | `"spearman"` (default), `"pearson"`, `"kendall"`, or `"MIC"` (case-insensitive) |
| `min_group_size` | Automatic groups smaller than this are merged into one group named `auto_singleton` |
| `intra_dist`, `inter_dist` | Distances inside and between groups in the partition tree handed to SHAP. The defaults rarely need changing |

It returns a dictionary:

| Key | Content |
|---|---|
| `groups` | Final `{group: [features]}` mapping |
| `auto_groups` | The automatic clusters before priors were applied |
| `summary` | A `DataFrame` indexed by `group` with `n_features`, `mean_abs_corr`, `max_abs_corr`, and `features`. Single-feature groups report `NaN` correlations |
| `features`, `threshold`, `method`, `corr_method` | The inputs that produced the structure |
| `corr_lnk`, `shap_lnk` | SciPy linkage matrices: the correlation dendrogram and the one SHAP uses |

`ModelExplainer.build_coalition_structure(X=None, prior_groups=None, ...)` takes the same arguments, builds the structure
from `X` (or `background_data` when `X` is omitted), and caches it as `explainer.coalition_structure_`.

#### Capturing Nonlinear Association with MIC

Spearman is a rank correlation and suits monotonic relationships. If variables are related nonlinearly (for example `x`
and `x²`), switch to the **Maximal Information Coefficient (MIC)**:

```python
# check: skip   (needs the optional minepy package)
coalitions_mic = build_coalition_structure(train[features], threshold=0.35, corr_method="MIC")
```

!!! note "MIC needs `minepy` (Python 3.10 only)"

    MIC uses the optional `minepy` package, which the `mic` extra installs **only on Python 3.10**. The extra is skipped on
    Python 3.11 and later, where `corr_method="MIC"` raises an `ImportError` that points you back to `"spearman"` unless
    you install `minepy` yourself.

    ```bash
    pip install 'supermodelingfactory[mic]'
    ```

    - `threshold` still applies to the distance `1 - MIC`: the higher the MIC, the more likely two features share a group.
    - In `summary`, `mean_abs_corr` and `max_abs_corr` then hold the within-group mean and maximum MIC.
    - MIC is computed for every pair of features, so it is much slower than Spearman on wide frames.

### 3.2 Compute Owen Values with `ModelExplainer`

```python
owen_x = test[features].head(10)          # Owen values are slow: explain a small batch

explainer.explain_owen(
    owen_x,
    coalition_structure=coalitions,
    model_output="log_odds",
    max_evals=200,
    silent=True,                          # hide the SHAP progress bar
)

# Group-level global importance over the batch
group_importance = explainer.owen_group_importance(normalize=True)
print(group_importance[["group", "n_features", "mean_abs_owen", "importance_pct"]])

# Group-level reason codes for the first row of the batch
reason_codes = explainer.owen_explain_instance()
print(reason_codes[["group", "owen_value", "abs_owen_value", "features"]])
print("base value:", reason_codes.attrs["base_value"])
```

`explain_owen(X, coalition_structure=None, prior_groups=None, threshold=0.35, method="complete", corr_method="spearman", background_data=None, model_output="probability", rebuild=False, **explain_kwargs)`:

| Argument | Meaning |
|---|---|
| `coalition_structure` | The dictionary from 3.1. If omitted, the explainer reuses its cached structure or builds one from `background_data` (or from `X` when there is none) |
| `prior_groups`, `threshold`, `method`, `corr_method` | Used only when the structure is built here. Passing `prior_groups` always rebuilds the structure, even if you also pass `coalition_structure` |
| `background_data` | Background sample for building the structure and the SHAP explainer; defaults to the explainer's own. A background whose values differ from those the cached explainer was built with makes it rebuild |
| `model_output` | `"probability"` (default) or `"log_odds"`. Use `"log_odds"` for reason codes: group contributions then add up on the log-odds scale |
| `rebuild` | Force a new SHAP `PartitionExplainer`. It is rebuilt automatically when the coalition structure (its features and linkage), the background values, or `model_output` differ from those it was built with, and reused otherwise, for example when only `X` changes |
| `**explain_kwargs` | Passed to the SHAP call, for example `max_evals` (default 500) and `silent` |

It returns a `shap.Explanation` and caches the results as `owen_values_`, `owen_expected_value_`, and `owen_explanation_`.
Owen values are additive: `base_value + sum(owen_value)` equals the model output (log-odds with `model_output="log_odds"`)
for each row.

| Method | Returns |
|---|---|
| `owen_group_importance(X=None, normalize=False)` | `group`, `n_features`, `features`, `mean_owen`, `mean_abs_owen`, sorted by `mean_abs_owen`; `normalize=True` adds `importance_pct` (a share that sums to 1). Group values are the sum of the members' Owen values per row |
| `owen_feature_importance(X=None, normalize=False)` | `feature`, `mean_abs_owen` (and `importance_pct`) per feature |
| `owen_explain_instance(x_row=None, aggregate_groups=True)` | Reason codes for one row: `group`, `n_features`, `features`, `owen_value`, `abs_owen_value`, sorted by absolute value, with `.attrs["base_value"]` and `.attrs["model_output"]`. `aggregate_groups=False` returns one row per feature instead (`feature`, `value`, `owen_value`, `abs_owen_value`). Without `x_row` it reads the first row of the cached batch |

!!! warning "`owen_explain_instance(x_row)` replaces the cached batch"

    Passing `x_row` re-runs the explainer for that single row **with `model_output="probability"`** and overwrites the
    cached results, so a later `owen_group_importance()` describes one row. To get log-odds reason codes for another
    applicant, call `explainer.explain_owen(row_frame, model_output="log_odds")` first, then `explainer.owen_explain_instance()`
    with no argument. Compute batch importance before you do either.

### 3.3 One-Step Call

If you do not need to inspect the grouping separately, pass the prior groups straight into `explain_owen()`. The
correlation clustering then runs on the explainer's `background_data` (or on `X` when there is none):

```python
explainer.explain_owen(
    owen_x,
    prior_groups=prior_groups,
    threshold=0.35,
    model_output="log_odds",
    max_evals=200,
    silent=True,
)
print(explainer.owen_group_importance().head())
print(explainer.coalition_structure_["groups"])
```

## 4. PDP: Average Marginal Effect

```python
pdp = explainer.partial_dependence(
    test[features],
    feature="max_dpd_12m",
    grid_resolution=50,
    sample_size=1000,
    random_state=42,
    prediction_batch_size=100000,
)
print(pdp.head())

explainer.pdp_plot(test[features], feature="max_dpd_12m", show=False, save_path="pdp_max_dpd.png")
```

`partial_dependence(X, feature, grid_resolution=50, percentiles=(0.05, 0.95), sample_size=None, random_state=None, prediction_batch_size=100000)`
sets the feature to each grid value for every row, predicts, and averages. The grid runs from the 5th to the 95th
percentile of the feature (`percentiles`). It returns `feature`, `grid_value`, and `average_prediction`.

## 5. ICE: Individual Response Curves

```python
ice = explainer.ice(
    test[features],
    feature="max_dpd_12m",
    grid_resolution=50,
    sample_size=200,
    random_state=42,
    prediction_batch_size=100000,
)
print(ice.head())

explainer.ice_plot(test[features], feature="max_dpd_12m", centered=True, show=False, save_path="ice_max_dpd.png")
```

`ice(X, feature, grid_resolution=50, percentiles=(0.05, 0.95), sample_size=200, random_state=None, centered=False, prediction_batch_size=100000)`
returns one curve per sampled row: `feature`, `sample_index` (the row label in `X`), `grid_value`, `prediction`. With
`centered=True`, each curve is shifted to start at 0 at the first grid point, which makes the shapes easy to compare.
`ice_plot` has the same arguments plus `show` and `save_path`, and its `sample_size` defaults to 100.

## 6. ALE: Accumulated Local Effects

```python
ale = explainer.ale(
    test[features],
    feature="inquiries_3m",
    bins=20,
    prediction_batch_size=100000,
)
print(ale.head())

explainer.ale_plot(test[features], feature="inquiries_3m", bins=20, show=False, save_path="ale_inquiries.png")
```

`ale(X, feature, bins=20, sample_size=None, random_state=None, prediction_batch_size=100000)` cuts the feature into `bins`
equal-count intervals, measures the average change in prediction across each interval using only rows inside it, and
accumulates those changes. The values are centered so that their count-weighted mean is 0. It returns `feature`,
`bin_left`, `bin_right`, `bin_center`, `ale_value`, and `n` (rows per bin). `ale_plot` takes the same arguments plus
`show` and `save_path`.

PDP, ICE, and ALE support one **numeric** feature at a time, and `feature` must be one of the model's input columns (a
name, or an integer position). To study a categorical variable, explain its WOE-encoded numeric column.

## 7. LIME: Local Surrogate Explanation

```python
lime_one = explainer.lime_explain_instance(
    x_row=test[features].iloc[0],
    X_train=train[features],
    num_features=10,
    num_samples=5000,
    random_state=42,
)
print(lime_one)
print(lime_one.attrs["score"])         # R² of the local linear surrogate

lime_global = explainer.lime_global_importance(
    X=test[features].sample(20, random_state=42),
    X_train=train[features],
    num_features=10,
    num_samples=500,
    random_state=42,
)
print(lime_global)
```

`lime_explain_instance(x_row, X_train=None, num_features=10, num_samples=5000, random_state=None, missing_strategy="median", **lime_kwargs)`
returns `feature`, `feature_rule`, `weight`, `abs_weight`, sorted by `abs_weight`. `x_row` is a `Series`, a `dict`, or a
one-row frame. `X_train` defaults to `background_data`. `attrs["intercept"]` (a dict keyed by class label) and
`attrs["score"]` describe the local linear model. Extra keyword arguments go to `LimeTabularExplainer`, for example
`discretize_continuous=False` to fit the surrogate on continuous features instead of binned ones.

`lime_global_importance(X, X_train=None, num_features=10, num_samples=2000, sample_size=100, random_state=None, missing_strategy="median", **lime_kwargs)`
explains up to `sample_size` sampled rows of `X` and returns `feature`, `mean_abs_lime_weight`, and `frequency` (in how many
of the explained rows the feature was among the top `num_features`). The mean is taken over those rows only.

`missing_strategy` handles NaN in `X_train` and `x_row`: `"median"` fills with training medians, `"drop"` removes the rows;
both issue a `UserWarning`.

!!! warning "Performance"

    Owen, PDP, ICE, ALE, and LIME all call the model many times. PDP, ICE, and ALE stack the grid rows and predict in
    batches of `prediction_batch_size` (default 100000, `None` for a single call); lower it to cap the memory of each predict
    call (the stacked grid itself is still built in memory; `sample_size` shrinks it). On large
    data, use `sample_size`, a small `background_data` sample, and a modest `max_evals`. Owen values are the slowest
    method: in our tests, explaining one row took about a second with 100 background rows and `max_evals=500`.

The one-click `CreditModelPipeline` can run these explainers for you; see its `explain_models`, `owen_enabled`, and
`business_prior_groups` options in [Top-Level Pipelines](../pipeline_one_click.md).

## FAQ

??? question "How are Owen value and SHAP related?"

    The Owen value is a Shapley value computed with a coalition structure. SMF uses `shap.PartitionExplainer`, which
    respects the grouping first and then splits credit within each group.

??? question "When should I use the Owen value?"

    Use it when variables are highly correlated, or when the business needs module-level reason codes. With several
    variables from one source (delinquency, multi-lending, debt capacity, device fraud), single-feature SHAP dilutes the
    attribution across them.

??? question "What does `threshold=0.35` mean?"

    Automatic clustering cuts the dendrogram at distance `1 - |association|`. With the default
    `corr_method="spearman"` and `method="complete"`, a threshold of 0.35 puts two features in one automatic group only if
    every pair in the group has `|corr| >= 0.65`. With `corr_method="MIC"`, the same threshold applies to `1 - MIC`. Use
    0.20 to 0.35 to group only strongly correlated variables, or about 0.50 to merge more loosely related ones.

??? question "When should I use `corr_method='MIC'`?"

    When variables have a clear nonlinear association that Spearman misses. MIC is slower and needs `minepy`, which the
    `mic` extra installs on Python 3.10 only. Elsewhere, keep `"spearman"`.

??? question "How do I choose between PDP and ALE?"

    With weakly correlated features, PDP is intuitive and easy to explain. With strongly correlated features, prefer ALE:
    PDP averages predictions at feature combinations that never occur in the data, while ALE changes the feature only inside
    local intervals.

??? question "Is LIME a global explanation?"

    No. LIME explains one applicant at a time. `lime_global_importance()` aggregates many local explanations, which gives
    only an approximate global ranking.

??? question "`ImportError` for `shap` or `lime`"

    Install the extra: `pip install 'supermodelingfactory[explain]'`.

??? question "My Owen reason codes are in probability although I asked for log-odds"

    You passed `x_row` to `owen_explain_instance()`. It recomputes that row with `model_output="probability"`. See the
    warning in [3.2](#32-compute-owen-values-with-modelexplainer).

??? question "SHAP values for my CatBoost or random forest model are in probability units and slow to compute"

    Only LightGBM, XGBoost, and logistic regression get a dedicated SHAP explainer. Every other model goes through the
    model-agnostic explainer on `predict_proba`, which works in probability units and evaluates the model many times per
    row. Use a small `background_data` sample and explain a subset of rows.

??? question "XGBoost with SHAP raises `could not convert string to float: '[5E-1]'`"

    This is a known incompatibility between XGBoost 3.1 and later and older SHAP releases. XGBoost writes the `base_score`
    of a single-target model as a one-element array string such as `"[5E-1]"`, which the older SHAP parser reads as a
    scalar.

    When `ModelExplainer` builds the XGBoost `TreeExplainer`, SMF patches SHAP in memory: it normalizes a single-element
    `base_score` while SHAP decodes the model, and it retries once through `XGBTreeModelLoader`. The patch applies only to
    the tree path for `model_type` `"xgb"` or `"xgboost"`; LightGBM, LR, Owen, PDP, ICE, ALE, and LIME are unaffected.

    If SHAP still rejects the model, SMF raises a `ValueError` that lists the detected versions. Then, in this order:

    1. Upgrade SMF to the latest release.
    2. On Python 3.11 or later, upgrade to `shap>=0.50.0`.
    3. Only if multi-class or multi-target models still fail, pin `xgboost<3.1` as a last resort.

    SMF never collapses a multi-element `base_score` vector to its first element, because that would misattribute
    multi-target explanations.
