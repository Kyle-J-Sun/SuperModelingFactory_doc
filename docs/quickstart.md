# Quickstart

Run a complete scorecard training pipeline in 5 minutes. This section uses **synthetic data** and does not depend on any external data source.

## 0. Setup

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Assumes the current directory is the SuperModelingFactory repository root
```

To run the model explainability example, install the optional explainability dependencies:

```bash
pip install 'supermodelingfactory[explain]'
```

## 1. Create a Synthetic Sample

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

data = pd.DataFrame({
    "user_id":    np.arange(n),
    "age":        rng.normal(35, 8, n).clip(18, 70),
    "income":     rng.lognormal(10, 0.4, n),
    "score_b":    rng.normal(600, 60, n),
    "city_grade": rng.choice(["A", "B", "C", "D"], n),
    "n_overdue":  rng.poisson(0.3, n),
})

# Synthesize a bad rate that is correlated with the features
logit = -6 + 0.02 * (data["score_b"] - 600) + 0.001 * (data["income"] - data["income"].mean())
prob = 1 / (1 + np.exp(-logit))
data["bad_flag"] = rng.binomial(1, prob)

print(data.head())
print("Bad rate:", data["bad_flag"].mean())
```

## 2. Sample Splitting

```python
from Modeling_Tool import SampleSplitter

features = ["age", "income", "score_b", "city_grade", "n_overdue"]
splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
train_df, test_df = splitter.split_df(data, target="bad_flag")
print(f"train={len(train_df)}  test={len(test_df)}")
```

## 3. WOE Encoding

```python
from Modeling_Tool import WOE_Master

woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)

train_woe = woe.transform(train_df)
test_woe  = woe.transform(test_df)

print(train_woe[[f"{f}_woe" for f in features]].head())
```

## 4. Train a Logistic Regression

```python
from Modeling_Tool import LRMaster

woe_features = [f"{f}_woe" for f in features]

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
# fit takes (data, varlist, tgt_name)
lr.fit(train_woe, woe_features, "bad_flag")

coef = lr.get_statsmodel_summary()
print(coef)
```

## 5. Train LightGBM

```python
from Modeling_Tool import GradientBoostingModel

gbm = GradientBoostingModel(
    "lgb",
    params={
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "early_stopping_rounds": 20,
        "eval_metric": "auc",
    },
)
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],
    test_woe[woe_features],  test_woe["bad_flag"],
)
```

!!! tip "Switch to XGBoost / CatBoost in one line"

    `GradientBoostingModel` is a unified interface. Replace `"lgb"` with `"xgb"` or `"cat"` to train
    XGBoost / CatBoost without changing any other code, for example `GradientBoostingModel("cat", {"n_estimators": 200})`.

## 6. Model Evaluation

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
)
evaluator.add_dataset("train", train_woe).add_dataset("test", test_woe)
perf = evaluator.evaluate()

print(perf[["index", "KS", "AUC", "Top10%_TargetRate"]])
```

!!! tip "Weighted samples?"

    If the DataFrame contains a weight column such as `sample_wgt`, pass `weight_col="sample_wgt"` to
    `fit` / `PerformanceEvaluator` / `GainsTableCalculator` and similar entry points; use the same column
    name for training and evaluation. See
    [Model Training — Sample Weights](guides/model.md#sample-weights) and
    [Model Evaluation — Sample-Weighted Evaluation](guides/eval.md#sample-weighted-evaluation).

## 7. Model Explainability

```python
from Modeling_Tool import ModelExplainer

explain_x = test_woe[woe_features]
background_x = train_woe[woe_features].sample(n=min(1000, len(train_woe)), random_state=42)
focus_feature = woe_features[0]

explainer = ModelExplainer(gbm, background_data=background_x)

# SHAP
explainer.explain(explain_x)
shap_importance = explainer.feature_importance(normalize=True)
local_shap = explainer.explain_instance(explain_x.iloc[[0]])

# PDP / ICE / ALE
pdp_curve = explainer.partial_dependence(explain_x, focus_feature, sample_size=1000, random_state=42)
ice_curve = explainer.ice(explain_x, focus_feature, sample_size=100, centered=True, random_state=42)
ale_curve = explainer.ale(explain_x, focus_feature, bins=20)

# LIME
lime_local = explainer.lime_explain_instance(
    explain_x.iloc[0],
    X_train=background_x,
    num_features=10,
    num_samples=3000,
    random_state=42,
)
lime_global = explainer.lime_global_importance(
    explain_x,
    X_train=background_x,
    sample_size=50,
    num_features=10,
    num_samples=1000,
    random_state=42,
)

print(shap_importance.head(10))
print(pdp_curve.head())
print(ice_curve.head())
print(ale_curve.head())
print(lime_local.head(10))
print(lime_global.head(10))
```

## 8. Generate the Excel Report

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("Performance")

em.merge_col(ws, ncols=5, text="LightGBM Model Performance Summary")
em.write_dataframe(
    ws, perf,
    title="Performance Metrics",
    titleformat="BLUE_H2",
    headerformat="ORANGE_H4",
    valueformat="NUM%.4",
)
em.close_workbook()

print("Generated model_report.xlsx")
```

## Complete Script

Combining the snippets above:

```python
import numpy as np
import pandas as pd
from Modeling_Tool import (
    SampleSplitter, WOE_Master, LRMaster,
    GradientBoostingModel, PerformanceEvaluator, ModelExplainer,
)
from ExcelMaster.ExcelMaster import ExcelMaster

# 1) Data
rng = np.random.default_rng(42)
n = 5000
data = pd.DataFrame({
    "user_id":    np.arange(n),
    "age":        rng.normal(35, 8, n).clip(18, 70),
    "income":     rng.lognormal(10, 0.4, n),
    "score_b":    rng.normal(600, 60, n),
    "city_grade": rng.choice(["A", "B", "C", "D"], n),
    "n_overdue":  rng.poisson(0.3, n),
})
logit = -6 + 0.02 * (data["score_b"] - 600) + 0.001 * (data["income"] - data["income"].mean())
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "city_grade", "n_overdue"]

# 2) Split
train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True) \
                     .split_df(data, target="bad_flag")

# 3) WOE
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe, test_woe = woe.transform(train_df), woe.transform(test_df)
woe_features = [f"{f}_woe" for f in features]

# 4) LightGBM
gbm = GradientBoostingModel("lgb", {"n_estimators": 200, "learning_rate": 0.05})
gbm.fit(train_woe[woe_features], train_woe["bad_flag"],
        test_woe[woe_features],  test_woe["bad_flag"])

# 5) Evaluate
perf = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
).add_dataset("train", train_woe).add_dataset("test", test_woe).evaluate()

# 6) Explain
explain_x = test_woe[woe_features]
background_x = train_woe[woe_features].sample(n=min(1000, len(train_woe)), random_state=42)
focus_feature = woe_features[0]
explainer = ModelExplainer(gbm, background_data=background_x)

explainer.explain(explain_x)
shap_importance = explainer.feature_importance(normalize=True)
pdp_curve = explainer.partial_dependence(explain_x, focus_feature, sample_size=1000, random_state=42)
ice_curve = explainer.ice(explain_x, focus_feature, sample_size=100, centered=True, random_state=42)
ale_curve = explainer.ale(explain_x, focus_feature, bins=20)
lime_local = explainer.lime_explain_instance(explain_x.iloc[0], X_train=background_x, num_features=10)
lime_global = explainer.lime_global_importance(explain_x, X_train=background_x, sample_size=50)

# 7) Report
em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("Performance")
em.write_dataframe(ws, perf, title="Model Performance",
                   titleformat="BLUE_H2",
                   headerformat="ORANGE_H4",
                   valueformat="NUM%.4")
em.close_workbook()
```

## Next Steps

- For more options at each step, read [End-to-End Pipelines](pipeline.md)
- For a deeper look at explainability methods, read [Model Explainability](guides/explainability.md)
- For detailed descriptions of every public API, go to the [API Reference](api/index.md)
- To build deployable scripts, see the code snippets in each [User Guide](guides/index.md)
