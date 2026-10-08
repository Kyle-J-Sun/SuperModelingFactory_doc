# Model Training

The [`Model`](../api/model.md) subpackage trains the models used in scorecard work and adds a variable-elimination helper.

| Class or function | Use it for | Sample weights |
|---|---|---|
| `LRMaster` | Logistic regression on WOE features: statistical summary, stepwise selection, calibration, holdout grid search | `weight_col` |
| `GradientBoostingModel` | One interface over LightGBM (`"lgb"`), XGBoost (`"xgb"`), and CatBoost (`"cat"`): warm start, calibration, parameter search | `sample_weight`, `eval_sample_weight` |
| `LightGBMModel`, `XGBoostModel`, `CatBoostModel` | The single-backend classes behind `GradientBoostingModel` | see [Sample weights](#sample-weights) |
| `lgbm_quick_train`, `xgbm_quick_train`, `catboost_quick_train` | One-call training from DataFrames; they return the bare fitted estimator | `wgt_col`, `val_wgt_col` |
| `BackwardVariableEliminator` | Dropping weak features round after round by cumulative importance | `weight_col`, `validation_weight_col` |
| `save_model`, `load_model`, `scoring` | Persisting a model and scoring a DataFrame with it | |

`catboost_quick_train` is imported from `Modeling_Tool.Model`; everything else on this page is importable from `Modeling_Tool`.

## Example data

Every snippet on this page runs top to bottom in one Python session. The setup creates a synthetic portfolio, a time-based
train / validation / out-of-time split, and WOE-encoded copies of the frames for logistic regression.

```python
import os

import numpy as np
import pandas as pd

from Modeling_Tool import WOE_Master

rng = np.random.default_rng(42)
n = 8000
months = [f"2025-{m:02d}" for m in range(1, 10)]

df = pd.DataFrame({
    "apply_month": rng.choice(months, n),
    "city_grade":  rng.choice(["A", "B", "C", "D"], n),
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
    "sample_wgt":  rng.uniform(0.5, 2.0, n),
})
logit = -2.2 - 0.02 * (df["score_b"] - 600) + 0.5 * df["n_overdue"] - 0.8 * df["utilization"]
df["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "utilization", "n_overdue"]
train = df[df["apply_month"] <= "2025-06"].copy()
valid = df[df["apply_month"] == "2025-07"].copy()
oot = df[df["apply_month"] >= "2025-08"].copy()

# WOE encoding (numeric features) for logistic regression
woe = WOE_Master(train_data=train, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe, valid_woe, oot_woe = (woe.transform(frame) for frame in (train, valid, oot))
woe_features = [f"{f}_woe" for f in features]

os.makedirs("models", exist_ok=True)
```

## Sample weights

Training and evaluation take optional sample weights. **Without weights, behavior is the historical, unweighted one.**
Weights must be non-negative and finite. Use the same weights when you evaluate: see
[Model Evaluation](eval.md#sample-weighted-evaluation).

| API | How to pass weights |
|---|---|
| `LRMaster.fit`, `stepwise_selection`, `get_aic`, `get_bic`, `calibrate_model` | `weight_col="<column of the frame>"` (`calibrate_model` also takes `sample_weight=<array>`) |
| `LRMaster.grid_search_params` | `weight_col` for training, `eval_weight_col` for the AUC on each evaluation set |
| `GradientBoostingModel.fit` | `sample_weight=` (training rows) and `eval_sample_weight=` (validation rows). For XGBoost, `sample_weight_eval_set=[...]` is also accepted |
| `LightGBMModel.fit` | `sample_weight=` (alias `wgt=`) and `eval_sample_weight=` |
| `XGBoostModel.fit` | `sample_weight=` and `sample_weight_eval_set=[<validation weights>]` |
| `CatBoostModel.fit` | `sample_weight=` for training only |
| `lgbm_quick_train`, `xgbm_quick_train`, `catboost_quick_train` | `wgt_col="<column>"` for training and `val_wgt_col="<column>"` for validation |
| `BackwardVariableEliminator` | `weight_col`, `validation_weight_col` (defaults to `weight_col`) |

Validation weights change the metric used for early stopping in LightGBM and XGBoost. **CatBoost ignores validation weights**:
`GradientBoostingModel("cat", ...).fit(..., eval_sample_weight=...)` accepts the argument and does nothing with it.

## 1. Logistic regression: `LRMaster`

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
lr.fit(train_woe, woe_features, "bad_flag")           # fit(data, varlist, tgt_name); returns the LRMaster

valid_score = lr.predict_proba(valid_woe)[:, 1]       # probability of the bad class

print(lr.get_statsmodel_summary())    # index: Intercept + features; columns: coef, std_err, z, p_value, ci_lower, ci_upper
print(lr.get_variable_importance())   # columns: varlist, coef, importance (= abs(coef)), sorted by importance
print(lr.get_aic(), lr.get_bic())
```

`params` is passed to scikit-learn's `LogisticRegression(**params)`, so any of its arguments works (`C`, `penalty`, `solver`,
`max_iter`, `class_weight`, ...) and omitted ones keep the defaults of your installed scikit-learn. Training uses the data in
`fit`; the validation arguments (`val_data`, `val_varlist`, `val_tgt_name`) are accepted for reference only.

!!! warning "`predict` returns class labels"

    `predict_proba(data)` returns an `(n, 2)` array whose column 1 is the bad-class probability, exactly as in scikit-learn.
    `predict(data)` returns hard 0/1 labels, not scores. Use `predict_proba(data)[:, 1]` whenever you need a score.

`LRMaster(params=None, model=None, varlist=None, tgt_name=None, standardize=False, scaler=None)` can also wrap an existing
fitted `LogisticRegression` through `model=` (give `varlist` if the model has no `feature_names_in_`).

| Method | Returns |
|---|---|
| `fit(data, varlist, tgt_name, val_data=None, val_varlist=None, val_tgt_name=None, weight_col=None)` | The `LRMaster` itself |
| `predict(data, varlist=None, calibrated_model=False)` | Hard 0/1 labels |
| `predict_proba(data, varlist=None, calibrated_model=False)` | `(n, 2)` probabilities; `calibrated_model=True` uses the calibrated model |
| `get_statsmodel_summary(data=None, varlist=None, tgt_name=None)` | DataFrame of coefficients, standard errors, z, p-values, and 95% intervals (computed from the Fisher information) |
| `get_variable_importance()` | DataFrame `varlist`, `coef`, `importance` |
| `get_aic(data=None, varlist=None, tgt_name=None, weight_col=None)`, `get_bic(...)` | Float |
| `stepwise_selection(data, varlist, tgt_name, criterion='aic', direction='both', max_iter=100, verbose=True, weight_col=None)` | List of the selected variables |
| `calibrate_model(model=None, train_df=None, method='sigmoid', cv=5, weight_col=None, sample_weight=None)` | The `LRMaster` itself |
| `eval_calibrated_outcome(evalset, plot=False, weight_col=None, sample_weight=None)` | `None`; prints the raw and calibrated Brier scores |
| `grid_search_params(data, varlist, tgt_name, eval_sets, param_grid, ...)` | DataFrame of results |
| `clone()` | An unfitted copy with the same configuration |
| `set_data(data)` | The `LRMaster`; stores the frame later used for calibration |

### Sample weights

```python
lr_w = LRMaster(params={"C": 1.0, "max_iter": 1000})
lr_w.fit(train_woe, woe_features, "bad_flag", weight_col="sample_wgt")
```

`LRMaster.fit` accepts only `weight_col`, a column of the training frame. It has no `sample_weight` array argument.

### Stepwise variable selection

`stepwise_selection` adds and removes variables by AIC or BIC (`direction` is `'forward'`, `'backward'`, or `'both'`). It
returns the selected names and leaves the model fitted on them.

```python
stepper = LRMaster(params={"C": 1.0, "max_iter": 1000})
selected = stepper.stepwise_selection(
    train_woe, woe_features, "bad_flag",
    criterion="aic",            # 'aic' or 'bic'
    direction="both",           # 'forward', 'backward', or 'both'
    weight_col="sample_wgt",    # optional: weighted AIC/BIC
    verbose=False,
)
print(f"Stepwise selection kept {len(selected)} variables: {selected}")
```

### Standardization (optional)

`LRMaster` does not standardize by default. Pass `standardize=True` to fit a `StandardScaler` on the training features; the
same scaler is applied in every prediction and evaluation call. Pass `scaler=` **together with** `standardize=True` to use
another scaler (the instance is cloned, never modified).

```python
from sklearn.preprocessing import MinMaxScaler

scaled = LRMaster(params={"C": 1.0, "max_iter": 1000}, standardize=True)
scaled.fit(train_woe, woe_features, "bad_flag")
proba = scaled.predict_proba(valid_woe)             # inputs are scaled automatically

custom = LRMaster(params={"C": 1.0}, standardize=True, scaler=MinMaxScaler())
custom.fit(train_woe, woe_features, "bad_flag")
print(type(custom.standardizer).__name__)           # MinMaxScaler
```

| Aspect | Behavior |
|---|---|
| Default | `standardize=False`: no scaling |
| Custom scaler | `scaler=` is ignored unless `standardize=True` |
| When it is fitted | Once, on the training features, in `fit` / `stepwise_selection`; stored as `lr.standardizer` |
| Consistency | `predict`, `predict_proba`, `calibrate_model`, `get_statsmodel_summary`, `get_aic`, and `get_bic` transform their input with the same scaler |
| `stepwise_selection` | Selection runs in the standardized space; the scaler is refitted on the selected variables at the end |
| `clone()` | Copies the `standardize` switch and the scaler prototype, not the fitted scaler or model |

With standardization on, the coefficients from `get_variable_importance()` and `get_statsmodel_summary()` are in the
**standardized space**: comparable across features, but no longer the log-odds change per original unit.

### Calibration

```python
lr.calibrate_model(method="sigmoid", cv=5)                 # 'sigmoid' or 'isotonic'; cv an int or 'prefit'
calibrated = lr.predict_proba(valid_woe, calibrated_model=True)[:, 1]
lr.eval_calibrated_outcome(valid_woe)                      # prints raw and calibrated Brier scores
```

Without `train_df`, `calibrate_model` uses the frame from `fit`. Calibrating on the training data is rarely what you want:
pass a holdout frame as `train_df=...`, or use `cv='prefit'` with a separate calibration set.

### Hyperparameter grid search on holdouts

`grid_search_params` trains one candidate per point of the Cartesian product of `param_grid`, scores the AUC on every frame in
`eval_sets`, and picks the best candidate by `objective`. It uses your **in-sample / out-of-sample / out-of-time holdouts**
rather than k-fold cross-validation.

```python
tuner = LRMaster(params={"solver": "lbfgs", "max_iter": 1000})
results = tuner.grid_search_params(
    data=train_woe,                                          # frame the candidates are trained on
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": train_woe, "oos": valid_woe, "oot": oot_woe},   # ordered; scored by AUC
    param_grid={"C": np.logspace(-3, 2, 6)},                 # several keys form a Cartesian product
    objective="oot_gap_penalized",                           # default
    primary_set="oot",                                       # default: the last key of eval_sets
    gap_ref_sets=["ins", "oos"],                             # default: every set except primary_set
    refit=True,                                              # refit the LRMaster on `data` with the best parameters
    weight_col="sample_wgt",                                 # training weights
    eval_weight_col="sample_wgt",                            # weighted AUC on each evaluation set
    verbose=False,
)
print(results.head())              # parameter columns, AUC_<set> per eval set, gap, score
print(tuner.best_params_)          # {'C': ...}
print(tuner.search_results_)       # the same table as the return value
```

| `objective` | Selection criterion |
|---|---|
| `'oot_gap_penalized'` (default) | `AUC[primary] - abs(mean(AUC[gap_refs]) - AUC[primary])`: a high primary-set AUC with little gap to the other sets |
| `'max_primary'` | `AUC[primary]` |
| a callable | `f(auc_dict) -> float`, where `auc_dict` maps each eval-set name to its AUC |

The returned table is sorted by `score`, best first. The search also writes `best_params_` and `search_results_`, merges the
best parameters into `lr.params`, and, with `refit=True`, refits the model on `data`. Only `metric='auc'` is supported. If the
`LRMaster` was built with `standardize=True`, every candidate uses the same scaling configuration.

## 2. Gradient boosting: `GradientBoostingModel`

`GradientBoostingModel(model_type, params)` takes `model_type` `'lgb'`, `'xgb'`, or `'cat'` (`'catboost'` is an alias). The
`params` dictionary is forwarded to the backend's scikit-learn estimator (`LGBMClassifier`, `XGBClassifier`,
`CatBoostClassifier`), so the backend's own defaults apply to everything you leave out. The validation set passed to `fit` is
used for early stopping.

```python
from Modeling_Tool import GradientBoostingModel

gbm = GradientBoostingModel(
    "lgb",
    params={
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,     # required for LightGBM
        "eval_metric": "auc",
        "verbose": -1,
    },
)
gbm.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"])    # returns the model

proba = gbm.predict(oot[features])                  # 1-D probability of the bad class
print(gbm.get_feature_importance().head())          # columns: feature, importance
print(gbm.roc_auc(oot[features], oot["bad_flag"]), gbm.brier_score(oot[features], oot["bad_flag"]))
```

Unknown attributes are passed through to the fitted estimator, so `gbm.predict_proba(X)` (an `(n, 2)` array),
`gbm.feature_names_in_`, and `gbm.get_params()` work, and a `GradientBoostingModel` can be handed directly to
`PerformanceEvaluator`, `ModelExplainer`, `scoring`, and `save_model`. The fitted estimator itself is `gbm._model.model`.

### Parameters that need care

| Parameter | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| `early_stopping_rounds` | **Required**; a missing key raises `KeyError` | Optional | Optional |
| `eval_metric` | Used (`'auc'`; default `'auc'`) | **Ignored**: early stopping uses XGBoost's own binary default, `logloss` | Used (`'AUC'`) |
| `n_estimators`, `max_depth` | Native names | Native names | Mapped to CatBoost's `iterations` and `depth`; the native names also work. Do not pass both |
| `cat_features` | n/a (CatBoost-only key) | n/a (CatBoost-only key) | List of raw categorical column names |
| `init_score` in `fit` (warm start) | Supported | Supported (as `base_margin`) | `NotImplementedError` |

The validation set carries no weights unless you pass `eval_sample_weight`; see [Sample weights](#sample-weights).

!!! note "`get_feature_importance` ignores `importance_type`"

    The method accepts `importance_type` for API symmetry, but each backend returns one fixed kind in 0.8.2: **LightGBM**
    total gain, **XGBoost** split counts, **CatBoost** its default `PredictionValuesChange` (which sums to 100). The values are
    not normalized. For another kind use the estimator directly, for example `gbm.booster_.feature_importance(importance_type="split")`
    for LightGBM or `gbm.get_booster().get_score(importance_type="gain")` for XGBoost.

### Sample weights

```python
weighted_gbm = GradientBoostingModel("lgb", {"n_estimators": 100, "early_stopping_rounds": 20, "verbose": -1})
weighted_gbm.fit(
    train[features], train["bad_flag"],
    valid[features], valid["bad_flag"],
    sample_weight=train["sample_wgt"],
    eval_sample_weight=valid["sample_wgt"],
)
```

### XGBoost and CatBoost

The interface is identical; only `model_type` and the parameter names that differ per backend change.

```python
xgb = GradientBoostingModel("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "early_stopping_rounds": 20})
xgb.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"])

cat = GradientBoostingModel(
    "cat",
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "early_stopping_rounds": 20,
     "eval_metric": "AUC", "verbose": False},
)
cat.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"])
print(cat.get_feature_importance().head())
```

### CatBoost with raw categorical columns

CatBoost can encode string columns itself, so you can skip WOE for them: list the columns in `cat_features`, keep them in the
feature frame, and use the same frame layout for training, validation, and scoring.

```python
cat_columns = features + ["city_grade"]
cat_native = GradientBoostingModel(
    "cat",
    {"iterations": 200, "depth": 4, "learning_rate": 0.05, "early_stopping_rounds": 20,
     "eval_metric": "AUC", "cat_features": ["city_grade"], "verbose": False},
)
cat_native.fit(train[cat_columns], train["bad_flag"], valid[cat_columns], valid["bad_flag"])
print(cat_native.predict(oot[cat_columns])[:3])
```

!!! tip "Strings in LightGBM and XGBoost"

    LightGBM raises `ValueError: pandas dtypes must be int, float or bool` for `object` columns, and listing the column in
    `params["categorical_feature"]` does not change that. Convert it first: `df[col] = df[col].astype("category")`, with the
    same categories in every frame. After WOE encoding, all features are numeric and none of this applies.

### Warm start (incremental learning)

Continue learning on new data from an existing model: the old model's log-odds become the starting point (`init_score`) for
the new model, and scoring fuses both: `sigmoid(old_margin + new_raw_score)`. It works for `lgb` and `xgb`; CatBoost raises
`NotImplementedError`.

```python
base_model = GradientBoostingModel("lgb", {"n_estimators": 100, "early_stopping_rounds": 20, "verbose": -1})
base_model.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"])

# 1) Log-odds of the old model, on the rows used for incremental training
base_margin_train = base_model.get_base_margin(train[features])

# 2) Train the new model on top of it (the same call for "xgb")
increment = GradientBoostingModel("lgb", {"n_estimators": 50, "early_stopping_rounds": 10, "verbose": -1})
increment.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"], init_score=base_margin_train)

# 3) Fused prediction: sigmoid(old margin + new raw score)
base_margin_oot = base_model.get_base_margin(oot[features])
fused = increment.predict_with_base_margin(oot[features], base_margin_oot, return_prob=True)    # False returns log-odds
```

| Step | XGBoost | LightGBM |
|---|---|---|
| Raw margin | `predict(X, output_margin=True)` | `predict(X, raw_score=True)` |
| Training offset | `fit(..., base_margin=...)` | `fit(..., init_score=...)` |
| Offset at prediction time | Native | **Not supported** |

`GradientBoostingModel` hides these differences: you always pass `init_score`, and fused prediction always goes through
`predict_with_base_margin`, the one approach that behaves identically on both backends. By default the offset is applied to the
training set only; the validation set gets none, so early stopping measures the metric without the offset. Pass
`eval_init_score=` (the offsets of the validation rows; LightGBM `eval_init_score`, XGBoost `base_margin_eval_set`) to stop on the
combined model instead. CatBoost raises `NotImplementedError` for both.

A model that was pickled as a bare `LGBMClassifier` or `XGBClassifier` can be adapted without retraining:
`GradientBoostingModel.from_fitted(estimator)`; the backend is detected from the estimator, or pass `model_type=`.

### Calibration

```python
calibrated = GradientBoostingModel("xgb", {"n_estimators": 100, "early_stopping_rounds": 10})
calibrated.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"])
calibrated.calibrate(valid[features], valid["bad_flag"], method="isotonic")      # 'sigmoid' (default) or 'isotonic'
proba = calibrated.predict(oot[features])
```

`calibrate` wraps the fitted model with `CalibratedClassifierCV` and fits the calibrator on the data you pass; use a
holdout, not the training data. Keep the default `cv='prefit'`: an integer `cv` refits clones without a validation set and
fails whenever `early_stopping_rounds` is set. After calibrating, importance and other tree-specific attributes are no longer
available on that object.

### Save and load

```python
gbm.save("models/gbm_estimator.pkl")                                     # the fitted estimator only
restored = GradientBoostingModel("lgb", gbm.params).load("models/gbm_estimator.pkl")    # load returns the model
print(np.allclose(restored.predict(oot[features]), gbm.predict(oot[features])))
```

`save_model` / `load_model` store a model together with metadata; see [Persistence and scoring](#4-persistence-and-scoring) and
[Model Registry and Versioning](model_registry.md).

### Parameter search

`GradientBoostingModel.param_search(...)` runs a holdout grid or Optuna search; see
[GBM Hyperparameter Search](gbm_param_search.md).

### The single-backend classes and the quick-train functions

`LightGBMModel(params, model=None)`, `XGBoostModel(params, model=None)`, and `CatBoostModel(params, model=None)` expose the same
methods as `GradientBoostingModel` (`fit`, `predict`, `get_feature_importance`, `calibrate`, `roc_auc`, `brier_score`, `save`,
`load`) plus `calibration_curve(x, y, n_bins=10)`. Their `fit` signatures differ slightly, as listed in
[Sample weights](#sample-weights). The quick-train functions take DataFrames plus the feature and target names and return
the **bare estimator**, not a `GradientBoostingModel`:

```python
from Modeling_Tool import LightGBMModel, lgbm_quick_train, xgbm_quick_train
from Modeling_Tool.Model import catboost_quick_train

lgb_params = {"n_estimators": 100, "learning_rate": 0.1, "early_stopping_rounds": 10, "verbose": -1}

lgb_single = LightGBMModel(dict(lgb_params))
lgb_single.fit(train[features], train["bad_flag"], valid[features], valid["bad_flag"])
curve = lgb_single.calibration_curve(valid[features], valid["bad_flag"], n_bins=10)   # (fraction_of_positives, mean_predicted_value)

lgb_estimator = lgbm_quick_train(
    train, valid, features, "bad_flag", dict(lgb_params), wgt_col="sample_wgt", val_wgt_col="sample_wgt",
)     # LGBMClassifier
xgb_estimator = xgbm_quick_train(
    train, valid, features, "bad_flag", params={"n_estimators": 100, "early_stopping_rounds": 10},
    wgt_col="sample_wgt", val_wgt_col="sample_wgt",
)     # XGBClassifier
cat_estimator = catboost_quick_train(
    train, valid, features, "bad_flag", {"n_estimators": 100, "early_stopping_rounds": 10, "verbose": False},
)     # CatBoostClassifier
```

| Function | Signature | Returns |
|---|---|---|
| `lgbm_quick_train` | `(train_data, validation_data, x, y, params, wgt_col=None, val_wgt_col=None, cat_x_train=None)` | `LGBMClassifier` |
| `xgbm_quick_train` | `(train_data, validation_data, x, y, wgt_col=None, params=None, sample_weight_eval_set=None, val_wgt_col=None)` | `XGBClassifier` |
| `catboost_quick_train` | `(train_data, validation_data, x, y, params, wgt_col=None, val_wgt_col=None, cat_features=None)` | `CatBoostClassifier` |

`x` is the list of feature columns and `y` the target column name.

## 3. Backward variable elimination: `BackwardVariableEliminator`

The eliminator trains a boosted model on the current variables, ranks them by importance, **keeps the top variables whose
cumulative importance reaches `cum_importance_threshold`** (and at least `min_vars` of them), and repeats on the survivors for
`n_rounds` rounds or until `min_vars` is reached. It is a lightweight screening step for long candidate lists.

```python
from Modeling_Tool import BackwardVariableEliminator

# 5 informative features plus 15 noise features
noise_rng = np.random.default_rng(7)
noise_cols = [f"noise_{i}" for i in range(15)]
for frame in (train, valid, oot):
    for col in noise_cols:
        frame[col] = noise_rng.normal(size=len(frame))
candidates = features + noise_cols

eliminator = BackwardVariableEliminator(
    train_data=train,
    varlist=candidates,
    dep="bad_flag",
    model_type="lgbm",                          # 'lgbm' or 'xgbm'
    validation_data=valid,                      # used for early stopping and the performance summary
    test_data_dict={"oot": oot},                # more frames to summarize, by name
    weight_col="sample_wgt",
    validation_weight_col="sample_wgt",
)
rounds = eliminator.run(
    n_rounds=3,
    varreduct_params={"num_leaves": 7, "learning_rate": 0.1},     # LightGBM parameters
    num_boost_round=100,
    early_stopping_rounds=10,
    cum_importance_threshold=0.99,
    min_vars=5,
)

print(eliminator.get_summary())         # round, n_vars_in, n_vars_out, vars_removed
print(eliminator.get_final_vars())      # variables kept after the last round
```

`BackwardVariableEliminator(train_data, varlist, dep, model_type='lgbm', validation_data=None, test_data_dict=None, weight_col=None, validation_weight_col=None, wgt_col=None)`
is configured at construction; `run(n_rounds=5, varreduct_params=None, stopping_metric='auc', seed=42, num_boost_round=200, early_stopping_rounds=20, importance_type='gain', cum_importance_threshold=0.99, min_vars=10, ret_perf=True, nbins=10, **kwargs)`
does the work and returns one dictionary per round:

| Key | Content |
|---|---|
| `round`, `n_vars_in`, `n_vars_out` | Round number, and the variable counts before and after |
| `selected_vars` | The variables kept |
| `model` | The fitted booster (`lightgbm.Booster`, or the XGBoost booster) |
| `perf` | `{split name: get_perf_summary frame}` with the keys `'mdl'` (training), `'hd'` (validation), and your `test_data_dict` names; empty when `ret_perf=False` |

!!! warning "`model_type` values"

    Use exactly `'lgbm'` or `'xgbm'` (the default is `'lgbm'`). Any other string, including `'lgb'` and `'xgb'`, silently
    runs XGBoost.

`varreduct_params` holds the booster's parameters. For LightGBM, SMF fills in `metric` (= `stopping_metric`), `seed`,
`objective='binary'`, `boosting_type='gbdt'`, and `num_threads=8` for the keys you do not set. Without `validation_data`,
early stopping watches the training set. The functions `backward_lgbm(...)` and `backward_xgbm(...)` in
`Modeling_Tool.Model` run a single round and return `(selected_vars, model, perf)`.

`BackwardEliminationAnalyzer(results)` (in `Modeling_Tool.Model`) analyzes the list that `run` returns:

```python
from Modeling_Tool.Model import BackwardEliminationAnalyzer

analyzer = BackwardEliminationAnalyzer(rounds)
print(analyzer.get_stable_vars(top_n=5))                     # variables kept in every round
print(analyzer.get_perf_trend(dataset="hd", metric="AUC"))   # metric per round on one split
```

`plot_var_reduction(figsize=(8, 4), save_path=None)` plots the number of variables per round (axes *Elimination Round* and
*Number of Variables Kept*).

## 4. Persistence and scoring

`save_model(model, filename, metadata=None, feature_cols=None, woe_mapping_path=None, train_window=None, metrics=None, model_name=None, model_version=None, include_metadata=True)`
pickles any model object (a `GradientBoostingModel`, an `LRMaster`, or a bare estimator) and returns `0`. With
`include_metadata=True` it also records the SMF and Python versions, the creation time, the model class, and what you pass in
`feature_cols`, `metrics`, `model_name`, `model_version`, plus any custom keys in `metadata`. `load_model(model_path,
return_metadata=False)` returns the model, or `(model, metadata)`; `load_model_metadata(model_path)` returns only the metadata.

```python
from Modeling_Tool import load_model, load_model_metadata, save_model, scoring

save_model(
    gbm, "models/gbm_v1.pkl",
    feature_cols=features, metrics={"oot_auc": 0.77}, model_name="demo_gbm", model_version="1.0",
    metadata={"owner": "risk-modeling"},
)
loaded, metadata = load_model("models/gbm_v1.pkl", return_metadata=True)
print(metadata["smf_version"], metadata["feature_cols"], metadata["owner"])
print(load_model_metadata("models/gbm_v1.pkl")["model_name"])        # metadata only

scored = scoring(data=oot, model=loaded, varlist=features, scr_name="prob")
print(scored[["prob"]].describe())
```

`scoring(data, model, varlist, scr_name, keeplist=None, all_missing_spec_value=None)` adds the model's bad-class probability
as `scr_name` to a copy of `data` (any model with `predict_proba`, including `LRMaster`). With `keeplist`, only those columns and
the score are returned. Rows whose features are **all** missing get `all_missing_spec_value` instead of the model's score; any
falsy value (`None`, `0`) leaves the override off.

## Model comparison in practice

Fit each model on the same split, score the holdouts, and compare them with one evaluator.

```python
from Modeling_Tool import PerformanceEvaluator

boosting_params = {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "early_stopping_rounds": 20}
models = {
    "lr": LRMaster({"C": 1.0, "max_iter": 1000}),
    "lgb": GradientBoostingModel("lgb", {**boosting_params, "verbose": -1}),
    "xgb": GradientBoostingModel("xgb", dict(boosting_params)),
    "cat": GradientBoostingModel("cat", {**boosting_params, "verbose": False}),
}

for name, model in models.items():
    if name == "lr":
        model.fit(train_woe, woe_features, "bad_flag", weight_col="sample_wgt")
        for frame in (train_woe, valid_woe, oot_woe):
            frame["score_lr"] = model.predict_proba(frame)[:, 1]
    else:
        model.fit(
            train_woe[woe_features], train_woe["bad_flag"], valid_woe[woe_features], valid_woe["bad_flag"],
            sample_weight=train_woe["sample_wgt"],
            eval_sample_weight=valid_woe["sample_wgt"],
        )
        for frame in (train_woe, valid_woe, oot_woe):
            frame[f"score_{name}"] = model.predict(frame[woe_features])

for name in models:
    perf = (
        PerformanceEvaluator(tgt_name="bad_flag", scr_name=f"score_{name}", weight_col="sample_wgt")
        .add_dataset("train", train_woe)
        .add_dataset("valid", valid_woe)
        .add_dataset("oot", oot_woe)
        .evaluate(display=False)
    )
    print(name, perf.set_index("index")[["AUC", "KS"]].round(4).to_dict("index"))
```

## FAQ

??? question "LightGBM raises `ValueError: pandas dtypes must be int, float or bool`"

    A feature column has the `object` dtype. Convert string columns before training: `df[col] = df[col].astype("category")`,
    using the same categories in every frame (training, validation, scoring). Listing the column in `params["categorical_feature"]`
    does not help. Alternatively WOE-encode the feature, or use CatBoost with `cat_features`.

??? question "LightGBM raises `KeyError: 'early_stopping_rounds'`"

    LightGBM training in SMF reads `params["early_stopping_rounds"]` and needs a validation set: pass both. XGBoost and
    CatBoost do not require it.

??? question "CatBoost: `n_estimators` / `max_depth`, or `iterations` / `depth`?"

    `GradientBoostingModel("cat", ...)` and `CatBoostModel` accept both, so one configuration can serve all three backends:
    `n_estimators` maps to `iterations`, and `max_depth` maps to `depth`. Do not pass an alias and its native name together.
    Other parameters (`learning_rate`, `l2_leaf_reg`, `early_stopping_rounds`, `eval_metric`, ...) use CatBoost's native names.

??? question "How does CatBoost handle categorical features?"

    List raw categorical columns in `params["cat_features"]` (names or indices); CatBoost encodes them with ordered target
    statistics, so no WOE or one-hot step is needed. The training, validation, and scoring frames must have the same columns.
    After WOE encoding every feature is numeric and `cat_features` is unnecessary.

??? question "Feature importances do not sum to 1"

    They are not normalized. LightGBM returns total gain, XGBoost returns split counts, CatBoost returns
    `PredictionValuesChange` (summing to 100), and `importance_type` does not change that. Divide by the sum if you need
    shares. For `ModelExplainer.feature_importance(normalize=True)` see [Model Explainability](explainability.md).

??? question "`calibrate` fails with an early-stopping error"

    You passed an integer `cv`, which refits clones of the estimator without a validation set. Keep the default `cv='prefit'`
    and pass a separate calibration frame.

??? question "After turning on standardization the coefficients look very different"

    That is expected: with `standardize=True` the model lives in the standardized space, and `get_variable_importance()` and
    `get_statsmodel_summary()` report standardized coefficients. Use `standardize=False` (the default) for coefficients in the
    original units.

??? question "How do I run k-fold cross-validation or a hyperparameter search?"

    `LRMaster.grid_search_params` searches on holdouts (see above) and `GradientBoostingModel.param_search` searches GBM
    parameters (see [GBM Hyperparameter Search](gbm_param_search.md)). For k-fold cross-validation, build a fresh estimator
    **without** `early_stopping_rounds` and `eval_metric`, because scikit-learn refits without a validation set:

    ```python
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_score

    cv_params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "verbose": -1}
    scores = cross_val_score(LGBMClassifier(**cv_params), train[features], train["bad_flag"], cv=5, scoring="roc_auc")
    ```

??? question "When should I use sample weights, and what is the difference between `weight_col` and `sample_weight`?"

    Typical uses are correcting sampling bias (for example weighting after oversampling), weighting by balance or amount, and
    time-decay weighting. `weight_col` names a column of the DataFrame you pass (`LRMaster`, the evaluators, the Pipelines);
    `sample_weight` is an array aligned with the rows (`GradientBoostingModel` and the single-backend classes). See
    [Sample weights](#sample-weights) for the argument each API takes.

## LR p-value backward elimination in the Pipeline

`CreditModelPipeline` can drop logistic-regression variables whose coefficient p-values are too high. After the first fit it
repeatedly removes the variable with the largest p-value and refits, until every p-value is at most the threshold or a limit
is hit. The p-values come from the same Fisher-information computation as `LRMaster.get_statsmodel_summary()`.

```python
from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig

# the five real features plus six noise columns that elimination should remove
pipeline_df = df.assign(badflag=df["bad_flag"], oot_flag=(df["apply_month"] >= "2025-08").astype(int))
pipe_rng = np.random.default_rng(3)
pipe_noise = [f"noise_{i}" for i in range(6)]
for col in pipe_noise:
    pipeline_df[col] = pipe_rng.normal(size=len(pipeline_df))

config = CreditModelPipelineConfig(
    output_dir="output/lr_elimination",
    target_col="badflag",
    feature_cols=features + pipe_noise,
    oot_col="oot_flag",
    train_models=["lr"],
    lr_elimination_mode="pvalue",              # None (default) disables it
    lr_elimination_params={"pvalue_threshold": 0.05, "min_features": 1, "max_iterations": 20},
    backward_enabled=False, optuna_models=[], explain_models=[], owen_enabled=False,
    write_excel=False, plot_outputs=False,
)
result = CreditModelPipeline(config).run(pipeline_df)

print(result.feature_selection_summary["lr_elimination"])    # iteration, dropped_feature, p_value, n_remaining
lr_master, lr_estimator, lr_features = result.models["lr"]   # (LRMaster, LogisticRegression, final feature list)
print(lr_features)
```

| `lr_elimination_params` key | Default | Meaning |
|---|---|---|
| `pvalue_threshold` | `0.05` | Stop when the largest coefficient p-value is at most this |
| `min_features` | `1` | Never go below this many variables |
| `max_iterations` | `20` | Maximum number of drops |
| `tie_breaker` | | Only `"pvalue"` (or `None`) is accepted, and it changes nothing: equal p-values are resolved by column order. Any other value raises `ValueError` |

The trajectory (`iteration`, `dropped_feature`, `p_value`, `n_remaining`) is also written to `lr_pvalue_elimination.csv` in
`output_dir` when outputs are written, and evaluation and explanation use the reduced list. The reduced list is
`result.models["lr"][2]`; `result.selected_features` and `result.model_feature_sets` still show the features that entered the
first fit. Any other key in `lr_elimination_params` raises `ValueError`.
