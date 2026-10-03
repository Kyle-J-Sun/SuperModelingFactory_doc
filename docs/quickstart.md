# Quickstart

Run a complete scorecard workflow in about five minutes. The walkthrough uses **synthetic data**, so it needs no external
data source. Every snippet runs top to bottom in a single Python session, in order.

Before you start, [install SMF](installation.md). The explainability step also needs the optional extra:

```bash
pip install 'supermodelingfactory[explain]'
```

## 1. Create a synthetic sample

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 6000

data = pd.DataFrame({
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
})

# Generate a bad flag that depends on the features
logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"] - 0.8 * data["utilization"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "utilization", "n_overdue"]
print(data.head())
print("Bad rate:", data["bad_flag"].mean())
```

## 2. Split the sample

```python
from Modeling_Tool import SampleSplitter

splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
train_df, test_df = splitter.split_df(data, target="bad_flag")
print(f"train={len(train_df)}  test={len(test_df)}")
```

## 3. WOE encoding

```python
from Modeling_Tool import WOE_Master

woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)

train_woe = woe.transform(train_df)     # adds one `<feature>_woe` column per feature
test_woe = woe.transform(test_df)

woe_features = [f"{f}_woe" for f in features]
print(train_woe[woe_features].head())
```

!!! note "Categorical (string) features"

    `WOE_Master` bins **numeric** features. For string columns, or when you need monotone bins, use
    `MonotoneWOEBinner(feature_cols=..., target_col=..., cate_feats=[...])`, then `.fit(df)` and `.apply_woe(df)`.
    See the [WOE guide](guides/woe.md).

## 4. Train a logistic regression

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
lr.fit(train_woe, woe_features, "bad_flag")       # fit(data, varlist, tgt_name)

print(lr.get_statsmodel_summary())               # coef, std_err, z, p_value, confidence interval
```

## 5. Train LightGBM

```python
from Modeling_Tool import GradientBoostingModel

gbm = GradientBoostingModel(
    "lgb",
    params={
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "early_stopping_rounds": 20,   # required
        "eval_metric": "auc",
        "verbose": -1,
    },
)
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],      # training data
    test_woe[woe_features], test_woe["bad_flag"],        # validation data (used for early stopping)
)
print(gbm.get_feature_importance().head())
```

!!! tip "Switch to XGBoost or CatBoost"

    `GradientBoostingModel` is a unified interface. Replace `"lgb"` with `"xgb"` or `"cat"` and keep the rest of the code,
    for example `GradientBoostingModel("cat", {"n_estimators": 200, "early_stopping_rounds": 20, "eval_metric": "AUC"})`.

## 6. Evaluate

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=woe_features)
evaluator.add_dataset("train", train_woe).add_dataset("test", test_woe)
perf = evaluator.evaluate(display=False)    # display=True needs IPython (notebooks)

print(perf[["index", "KS", "AUC", "Top10%_TargetRate"]])
```

!!! tip "Weighted samples?"

    If your frame has a weight column such as `sample_wgt`, pass `weight_col="sample_wgt"` to `LRMaster.fit`,
    `PerformanceEvaluator`, and `GainsTableCalculator`, and use `sample_weight=` for `GradientBoostingModel.fit`. Use the same
    weights in training and evaluation. See [Model Training: Sample Weights](guides/model.md#sample-weights) and
    [Model Evaluation: Sample-Weighted Evaluation](guides/eval.md#sample-weighted-evaluation).

## 7. Explain the model

```python
from Modeling_Tool import ModelExplainer

explain_x = test_woe[woe_features]
background_x = train_woe[woe_features].sample(n=1000, random_state=42)
focus_feature = woe_features[2]

explainer = ModelExplainer(gbm, background_data=background_x)

# SHAP: global importance and one instance
explainer.explain(explain_x)
shap_importance = explainer.feature_importance(normalize=True)
local_shap = explainer.explain_instance(explain_x.iloc[[0]])

# PDP / ICE / ALE for one feature
pdp_curve = explainer.partial_dependence(explain_x, focus_feature, sample_size=1000, random_state=42)
ice_curve = explainer.ice(explain_x, focus_feature, sample_size=100, centered=True, random_state=42)
ale_curve = explainer.ale(explain_x, focus_feature, bins=20)

# LIME: one instance
lime_local = explainer.lime_explain_instance(
    explain_x.iloc[0], X_train=background_x, num_features=5, num_samples=1000, random_state=42,
)

print(shap_importance.head())
print(pdp_curve.head())
```

## 8. Generate an Excel report

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster("model_report.xlsx", verbose=False)      # `verbose` is required
ws = em.add_worksheet("Performance")

em.merge_col(ws, ncols=5, text="LightGBM performance summary", cformat="BLUE_H2")
em.write_dataframe(
    ws, perf,
    title="Performance metrics",
    titleformat="BLUE_H2",
    headerformat="ORANGE_H4",
    valueformat="----",
)
em.close_workbook()

print("Generated model_report.xlsx")
```

## Complete script

The same workflow without the commentary (steps 1 to 6 and 8):

```python
import numpy as np
import pandas as pd
from Modeling_Tool import SampleSplitter, WOE_Master, GradientBoostingModel, PerformanceEvaluator
from ExcelMaster.ExcelMaster import ExcelMaster

# 1) Data
rng = np.random.default_rng(42)
n = 6000
data = pd.DataFrame({
    "age": rng.normal(35, 8, n).clip(18, 70),
    "income": rng.lognormal(10, 0.4, n),
    "score_b": rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue": rng.poisson(0.3, n),
})
logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"] - 0.8 * data["utilization"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
features = ["age", "income", "score_b", "utilization", "n_overdue"]

# 2) Split
train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True).split_df(data, target="bad_flag")

# 3) WOE
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe, test_woe = woe.transform(train_df), woe.transform(test_df)
woe_features = [f"{f}_woe" for f in features]

# 4) LightGBM
gbm = GradientBoostingModel("lgb", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
                                    "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1})
gbm.fit(train_woe[woe_features], train_woe["bad_flag"], test_woe[woe_features], test_woe["bad_flag"])

# 5) Evaluate
perf = (PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=woe_features)
        .add_dataset("train", train_woe).add_dataset("test", test_woe).evaluate(display=False))

# 6) Report
em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("Performance")
em.write_dataframe(ws, perf, title="Model performance", titleformat="BLUE_H2",
                   headerformat="ORANGE_H4", valueformat="----")
em.close_workbook()
```

## Next steps

- For more options at each step, read [End-to-End Pipelines](pipeline.md)
- To go deeper on explainability, read [Model Explainability](guides/explainability.md)
- For detailed descriptions of every public API, go to the [API Reference](api/index.md)
- To run the whole workflow in one call, see the one-click pipelines in [Top-Level Pipelines](pipeline_one_click.md)
