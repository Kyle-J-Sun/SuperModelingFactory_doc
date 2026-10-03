# Model Explainability

SuperModelingFactory provides a unified explainer, `ModelExplainer`, in the [`Explainability`](../api/explainability.md) subpackage. It supports SHAP, Owen Value, PDP, ICE, ALE, and LIME, explaining SMF models or native sklearn / LightGBM / XGBoost models through a single entry point.

!!! note "Install the optional dependencies"

    SHAP, Owen Value, and LIME require the optional explainability dependencies:

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

    `ModelExplainer` uses lazy loading: `from Modeling_Tool import ModelExplainer` does not import `shap` or `lime` immediately; the dependencies are loaded only when the corresponding method is called.

## Method Comparison

| Method | Type | Questions it answers | Dependency |
|------|------|----------------|------|
| SHAP | Global + local attribution | Which variables contribute the most? Why is a single sample high-risk? | `shap` |
| Owen Value | Group attribution | What is the overall contribution of business modules such as delinquency, multi-lending, and debt? | `shap` |
| PDP | Global average effect | When a variable increases, how does the average predicted risk change? | Built in |
| ICE | Individual response curves | Do different samples respond consistently to the same variable? | Built in |
| ALE | Accumulated local effects | When features are strongly correlated, how can marginal impact be viewed more robustly? | Built in |
| LIME | Local surrogate model | Near a single sample, what is the local linear explanation? | `lime` |

## 1. Create an Explainer

```python
from Modeling_Tool import GradientBoostingModel, ModelExplainer

gbm = GradientBoostingModel("lgb", {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "early_stopping_rounds": 30,
    "eval_metric": "auc",
})
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],
    test_woe[woe_features], test_woe["bad_flag"],
)

exp = ModelExplainer(
    gbm,
    background_data=train_woe[woe_features].sample(1000, random_state=42),
)
```

## 2. SHAP Attribution

```python
exp.explain(test_woe[woe_features])

fi = exp.feature_importance(normalize=True)
print(fi[["feature", "mean_abs_shap", "importance_pct"]].head(10))

exp.summary_plot(show=False, save_path="shap_summary.png")
exp.summary_plot(plot_type="bar", show=False, save_path="shap_bar.png")
exp.dependence_plot("age_woe", show=False, save_path="shap_dep_age.png")

contrib = exp.explain_instance(test_woe[woe_features].iloc[[0]])
print(contrib)
print("base value:", contrib.attrs["base_value"])
```

## 3. Owen Value: Group Attribution

Plain SHAP treats every feature as an independent player. Strongly correlated variables share the same risk signal, so, for example, the single-feature contribution of each of several delinquency variables looks "not prominent enough". Owen Value first puts features into coalitions and then distributes contribution within each coalition, which makes it better suited to producing business-module-level reason codes.

### 3.1 Build the Coalition Structure

```python
from Modeling_Tool import build_coalition_structure

prior_groups = {
    "delinquency": ["max_dpd_12m_woe", "dpd_cnt_6m_woe", "ever_dpd30_woe"],
    "multi_lending": ["inquiries_3m_woe", "inquiries_6m_woe", "active_loans_woe"],
    "affordability": ["monthly_income_woe", "debt_to_income_woe", "monthly_obligation_woe"],
}

cs = build_coalition_structure(
    train_woe[woe_features],
    prior_groups=prior_groups,
    threshold=0.35,
    method="complete",
    corr_method="spearman",
)

print(cs["summary"][["n_features", "mean_abs_corr", "max_abs_corr"]])
```

The fusion logic is: business priors take precedence, and the remaining features fall back to automatic clustering on Spearman correlation distance. `prior_groups` may contain variable names that do not exist in the current data (they are ignored automatically), but one effective feature cannot appear in more than one prior group.

#### Capturing Nonlinear Association with MIC

The default `corr_method="spearman"` is based on rank correlation and suits monotonic relationships. If there is clear nonlinear association between variables (for example `x` and `x²`), switch to the **Maximal Information Coefficient (MIC)**:

```python
cs = build_coalition_structure(
    train_woe[woe_features],
    threshold=0.35,
    corr_method="MIC",
)
```

!!! note "MIC optional dependency"

    MIC requires the extra `minepy` package (GPLv3, with C extensions):

    ```bash
    pip install 'supermodelingfactory[mic]'
    ```

    - `corr_method="MIC"` is case-insensitive (`"mic"` also works).
    - `threshold` still applies to the `1 - MIC` distance: the higher the MIC, the more easily features are placed in the same group.
    - In MIC mode, `mean_abs_corr` / `max_abs_corr` in `summary` mean the within-group average/maximum MIC.
    - MIC must be computed for every pair of features, so its cost scales with the number of feature pairs and is usually much slower than Spearman.
    - `minepy` currently has a known build problem on Python 3.11+; if installation fails, use a Python 3.10 environment, or fall back to `corr_method="spearman"`.

### 3.2 Compute Owen Value with ModelExplainer

```python
owen_exp = exp.explain_owen(
    X=test_woe[woe_features],
    coalition_structure=cs,
    model_output="log_odds",
    max_evals=500,
)

# Group-level global importance
owen_group = exp.owen_group_importance(normalize=True)
print(owen_group[["group", "n_features", "mean_abs_owen", "importance_pct"]])

# Module-level reason code for a single sample
local_reason = exp.owen_explain_instance(test_woe[woe_features].iloc[0])
print(local_reason[["group", "owen_value", "abs_owen_value", "features"]])
```

`model_output="log_odds"` suits credit reason codes, because a group's contribution can be read directly as an additive effect on the log-odds; if you want to explain the positive-class probability, use the default `model_output="probability"`.

### 3.3 One-Step Call

If you don't need to inspect the grouping separately, you can also pass the prior groups directly into `explain_owen()`:

```python
exp.explain_owen(
    test_woe[woe_features],
    prior_groups=prior_groups,
    threshold=0.35,
    model_output="log_odds",
)
print(exp.owen_group_importance().head())
```

## 4. PDP: Average Marginal Effect

```python
pdp = exp.partial_dependence(
    X=test_woe[woe_features],
    feature="age_woe",
    grid_resolution=50,
    sample_size=2000,
    random_state=42,
    prediction_batch_size=100000,
)
print(pdp.head())

exp.pdp_plot(test_woe[woe_features], feature="age_woe", show=False, save_path="pdp_age.png")
```

## 5. ICE: Individual Response Curves

```python
ice = exp.ice(
    X=test_woe[woe_features],
    feature="age_woe",
    grid_resolution=50,
    sample_size=200,
    random_state=42,
    prediction_batch_size=100000,
)
print(ice.head())

exp.ice_plot(test_woe[woe_features], feature="age_woe", centered=True, show=False, save_path="ice_age.png")
```

## 6. ALE: Accumulated Local Effects

```python
ale = exp.ale(
    X=test_woe[woe_features],
    feature="income_woe",
    bins=20,
    prediction_batch_size=100000,
)
print(ale.head())

exp.ale_plot(test_woe[woe_features], feature="income_woe", bins=20, show=False, save_path="ale_income.png")
```

ALE currently supports a single numeric feature. For categorical variables, explain the WOE-encoded numeric column instead.

## 7. LIME: Local Surrogate Explanation

```python
lime_one = exp.lime_explain_instance(
    x_row=test_woe[woe_features].iloc[0],
    X_train=train_woe[woe_features],
    num_features=10,
    num_samples=5000,
    random_state=42,
)
print(lime_one)

lime_global = exp.lime_global_importance(
    X=test_woe[woe_features].sample(100, random_state=42),
    X_train=train_woe[woe_features],
    num_features=10,
    num_samples=2000,
)
print(lime_global)
```

!!! warning "Performance advice"

    PDP / ICE / ALE / LIME / Owen Value all call model prediction repeatedly. PDP, ICE, and ALE first stack the grid samples and then predict in batches of `prediction_batch_size`; the default is `100000`, and you can lower it to reduce peak memory. With large production data, also use `sample_size`, `background_data.sample(...)`, or `max_evals` to control the cost of explanation.

## FAQ

??? question "How are Owen Value and SHAP related?"

    Owen Value is a Shapley allocation with a coalition structure. The implementation uses `shap.PartitionExplainer`, which respects the feature grouping first and then distributes contribution within each group.

??? question "When should I use Owen Value?"

    Use it when variables are highly correlated, or when the business needs module-level reason codes. For example, with variables from the same source such as delinquency, multi-lending, debt capacity, and device fraud, single-feature SHAP easily dilutes the attribution.

??? question "What does threshold=0.35 mean?"

    Clustering uses the distance `1 - abs(association)`. With the default `corr_method="spearman"`, `0.35` roughly means features with `|corr| > 0.65` are more likely to be grouped together first. With `corr_method="MIC"`, the same threshold applies to `1 - MIC`. Use `0.20~0.35` for strongly correlated scenarios, or `0.50` to be looser.

??? question "When should I use corr_method='MIC'?"

    When variables have clear nonlinear association and Spearman has trouble grouping them together. MIC is slower and requires a separate install with `pip install 'supermodelingfactory[mic]'`; on Python 3.11+, if `minepy` fails to install, use Python 3.10 or keep using `spearman`.

??? question "How do I choose between PDP and ALE?"

    If correlation between features is weak, PDP is intuitive and easy to understand; if correlation is significant, ALE is usually more robust because it perturbs the feature only within local intervals.

??? question "Is LIME a global explanation?"

    LIME is inherently a local explanation. `lime_global_importance()` aggregates many local explanations and can serve only as an approximate global importance reference.

??? question "Why do LIME / SHAP / Owen report missing dependencies?"

    You need to install the explain extra:

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

??? question "XGBoost + SHAP raises `could not convert string to float: '[5E-1]'` — what should I do?"

    This is a known compatibility problem between XGBoost 3.1+ and older SHAP versions. XGBoost serializes the `base_score` of a single-target model as a single-element array string such as `"[5E-1]"`, while SHAP 0.49 and earlier still parse it as a scalar.

    SuperModelingFactory enables a compatibility layer automatically when `ModelExplainer` builds the XGBoost `TreeExplainer`:

    - During SHAP's UBJSON decoding stage, a single-element `base_score` is normalized to a scalar.
    - An XGBoost-specific fallback is added to `XGBTreeModelLoader`, with one retry on failure.
    - It applies only to the TreeSHAP path with `model_type in {"xgb", "xgboost"}`, and does not affect LightGBM, LR, Owen, or PDP/ICE/ALE/LIME.

    Recommended order:

    1. Install the latest SuperModelingFactory main branch or the latest release.
    2. On Python 3.11+, you can upgrade to `shap>=0.50.0` to get native upstream support.
    3. Only if multi-class/multi-target models still fail, consider temporarily pinning `xgboost<3.1` as a last resort.

    Note: SMF does not silently collapse a multi-element `base_score` vector to its first element, to avoid misinterpreting multi-target attribution.
