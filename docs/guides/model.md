# Model Training

SuperModelingFactory wraps four families of models in the [`Model`](../api/model.md) subpackage — **logistic regression (the scorecard first choice), LightGBM, XGBoost, and CatBoost** — and provides a **backward variable elimination** helper.

!!! tip "Sample weights"

    Both training and evaluation support optional sample weights (`weight_col` / `sample_weight`).
    Without weights, behavior is exactly the same as in previous versions (backward compatible). A native API since the main repository's [PR #25](https://github.com/Kyle-J-Sun/SuperModelingFactory/pull/25).
    For the evaluation-side semantics, see [Model Evaluation — Sample Weights](eval.md#sample-weighted-evaluation).

## 1. Logistic Regression — `LRMaster`

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
# fit takes (data, varlist, tgt_name), not (X, y)
lr.fit(train_woe, woe_features, "bad_flag")

# Statistical summary: coefficients, standard errors, z, p-value, confidence intervals
summary = lr.get_statsmodel_summary()
print(summary)
```

### Sample Weights

`fit` supports passing weights as a DataFrame column name or an explicit array (**pick one**; the `wgt` / `wgt_col` aliases are also accepted):

```python
# Option 1: column name (recommended; same naming as the evaluation-side weight_col)
lr.fit(train_woe, woe_features, "bad_flag", weight_col="sample_wgt")

# Option 2: explicit array
lr.fit(train_woe, woe_features, "bad_flag", sample_weight=train_woe["sample_wgt"].values)
```

`stepwise_selection`, `calibrate_model`, and `get_aic` / `get_bic` pass the weights through as well.
Weights must be non-negative finite values.

### Key Parameters (passed through to sklearn)

| Parameter | Default | Description |
|------|-------|------|
| `C` | `1.0` | Inverse regularization strength; the smaller, the stronger the regularization |
| `penalty` | `"l2"` | `l1` / `l2` / `elasticnet` |
| `solver` | `"lbfgs"` | Optimization algorithm |
| `max_iter` | `100` | Maximum number of iterations |

### Variable Importance

```python
# LR coefficients (sorted by absolute value); columns are varlist / coef / importance
varimp = lr.get_variable_importance()
print(varimp[["varlist", "coef", "importance"]])
```

### Stepwise Variable Selection

`stepwise_selection(data, varlist, tgt_name, ...)` does forward / backward / bidirectional selection based on AIC/BIC:

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0})
selected = lr.stepwise_selection(
    train_woe, woe_features, "bad_flag",
    criterion="aic",        # 'aic' or 'bic'
    direction="both",       # 'forward' / 'backward' / 'both'
    weight_col="sample_wgt",  # optional: weighted AIC/BIC
)
print(f"Stepwise selection kept {len(selected)} variables: {selected}")
```

### Standardization (Optional)

By default, `LRMaster` **does not** standardize features (consistent with historical behavior). If you want to standardize the features before they enter the model,
turn on `standardize=True` at construction; `StandardScaler` is used by default:

```python
from Modeling_Tool import LRMaster

# Turn on standardization (StandardScaler by default)
lr = LRMaster(params={"C": 1.0, "max_iter": 1000}, standardize=True)
lr.fit(train_woe, woe_features, "bad_flag")

# At prediction time, the input is automatically transformed with the scaler fitted during fit; no manual standardization needed
proba = lr.predict_proba(test_woe)
```

You can also pass a custom scaler (**together with** `standardize=True`), such as `MinMaxScaler`:

```python
from sklearn.preprocessing import MinMaxScaler
from Modeling_Tool import LRMaster

lr = LRMaster(
    params={"C": 1.0},
    standardize=True,
    scaler=MinMaxScaler(),   # the instance passed in is cloned; the original object is not modified
)
lr.fit(train_woe, woe_features, "bad_flag")
```

#### Behavior Notes

| Aspect | Description |
|------|------|
| Default | `standardize=False`, no standardization at all (backward compatible) |
| Default scaler | `StandardScaler`; a custom one can be passed through `scaler=` (such as `MinMaxScaler()`) |
| When it is fitted | The scaler is fitted **only once, on the training features**, during `fit` / `stepwise_selection`, and stored in `lr.standardizer` |
| Inference consistency | `predict` / `predict_proba` / `calibrate_model` / `get_statsmodel_summary` / `get_aic` / `get_bic` all transform their input with the same scaler, avoiding a training / inference space mismatch |
| `stepwise_selection` | Selection happens in the standardized space; when it finishes, the scaler is refitted on the **finally selected variables** |
| `clone()` | Copies only the `standardize` switch and the scaler prototype; it does **not** copy the fitted scaler / model |

!!! note "Interpreting coefficients"

    With standardization on, the coefficients returned by `get_variable_importance()` and `get_statsmodel_summary()` are
    coefficients in the **standardized space** — the benefit is that coefficient sizes of features on different scales can be compared directly; but they no longer equal
    the log-odds change for "a 1-unit change in the predictor" in the original units.

!!! warning "A custom scaler requires standardization to be turned on explicitly"

    Passing only `scaler=...` without `standardize=True` does not enable standardization; a custom scaler must be used together with
    `standardize=True`.

### Hyperparameter Grid Search (Holdout)

`grid_search_params(...)` runs a hyperparameter grid search on **INS / OOS / OOT holdouts** (rather than k-fold cross-validation),
designed for the "in-sample / out-of-sample / out-of-time" scenario common in scorecards: it trains a candidate model for each point in the Cartesian product of `param_grid`,
computes AUC on every eval set, and then picks the best combination according to `objective`.

```python
import numpy as np
from Modeling_Tool import LRMaster

tuner = LRMaster(params={"solver": "lbfgs", "max_iter": 1000})
results = tuner.grid_search_params(
    data=ins_fit,                  # data used to train candidate models (usually the INS)
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},  # ordered, scored by AUC
    param_grid={"C": np.logspace(-3, 2, 31)},   # multiple keys are combined as a Cartesian product
    objective="oot_gap_penalized",  # default: maximize the primary-set AUC while penalizing the overfitting gap
    primary_set="oot",              # defaults to the last key of eval_sets
    gap_ref_sets=["ins", "oos"],    # defaults to all sets except primary_set
    refit=True,                     # after the search, refit self on data with the best parameters
    weight_col="sample_wgt",        # training-set weight column
    eval_weight_col="sample_wgt",   # weighted AUC scoring on each eval set
)

print(tuner.best_params_)     # best-parameter dict
print(tuner.search_results_)  # full results table (= return value)
```

#### Three Objectives

| objective | Selection criterion |
|---|---|
| `'oot_gap_penalized'` (default) | `AUC[primary] - |mean(AUC[gap_refs]) - AUC[primary]|`, i.e. raise the primary-set AUC while penalizing the AUC gap between training and holdout (overfitting) |
| `'max_primary'` | Directly maximize `AUC[primary]` |
| callable | Custom `f(auc_dict) -> float`, where `auc_dict` is `{set_name: AUC}` |

#### Return Value and Side Effects

- **Returns**: a results table sorted by `score` in descending order, with columns: the parameter columns + `AUC_<name>` for each eval set + `gap` (under the gap objective) + `score`.
- **Side effects**: writes `self.best_params_` and `self.search_results_`, merges the best combination into `self.params`; with `refit=True`, it also refits `self.model` on `data` with the best parameters.

!!! note "Holdout, not CV; only AUC is supported for now"

    This is a holdout search based on the `eval_sets` you provide explicitly (not k-fold cross-validation), which fits the risk-control
    INS/OOS/OOT practice better. `metric` currently supports only `'auc'`.

!!! tip "Standardization config is inherited automatically"

    If this `LRMaster` has `standardize=True`, every candidate inherits the same configuration (each fits its scaler on `data`),
    ensuring the search and the final model live in the same feature space.

## 2. Gradient Boosting Models — `GradientBoostingModel`

A unified interface to LightGBM / XGBoost / CatBoost.

```python
from Modeling_Tool import GradientBoostingModel

gbm = GradientBoostingModel(
    model_type="lgb",       # 'lgb' / 'xgb' / 'cat'
    params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
        "eval_metric": "auc",
    },
)
gbm.fit(
    train_X, train_y,
    val_X,   val_y,
)

# Variable importance
varimp = gbm.get_feature_importance()
print(varimp.head(15))

# Predict (returns the positive-class probability, 1-D)
proba = gbm.predict(test_X)

# Calibration (optional)
gbm.calibrate(val_X, val_y, method="isotonic")
```

### Sample Weights

`GradientBoostingModel.fit` and the underlying `LightGBMModel` / `XGBoostModel` / `CatBoostModel`
support training-set `sample_weight` (alias `wgt`) and validation-set `eval_sample_weight`:

```python
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],
    test_woe[woe_features],  test_woe["bad_flag"],
    sample_weight=train_woe["sample_wgt"],
    eval_sample_weight=test_woe["sample_wgt"],
)
```

CatBoost injects the weights through `Pool(weight=...)`. `calibrate` and the internal `roc_auc` / `brier_score`
evaluation accept `sample_weight` as well.

`lgbm_quick_train` / `xgbm_quick_train` can specify the validation-set weight column with `val_wgt_col`:

```python
from Modeling_Tool import lgbm_quick_train

model = lgbm_quick_train(
    train_X, train_y, val_X, val_y,
    params={"n_estimators": 200},
    val_wgt_col="sample_wgt",
)
```

#### CatBoost Example (`model_type="cat"`)

CatBoost uses the same interface; just change `model_type` to `"cat"`. The unified parameter names (`n_estimators` /
`max_depth`) are mapped automatically to CatBoost's native parameters (`iterations` / `depth`), with no change to the rest of the calling code;
if the data has raw categorical columns, you can hand them to CatBoost's native handling through `cat_features` (no need for WOE / one-hot first).

```python
from Modeling_Tool import GradientBoostingModel

cat = GradientBoostingModel(
    model_type="cat",       # CatBoost
    params={
        "n_estimators": 500,        # equivalent to CatBoost's iterations
        "learning_rate": 0.05,
        "max_depth": 6,             # equivalent to CatBoost's depth
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "early_stopping_rounds": 30,
        "eval_metric": "AUC",
        "cat_features": ["city_grade"],   # optional: pass raw categorical columns directly
    },
)
cat.fit(train_X, train_y, val_X, val_y)

# Downstream usage identical to lgb / xgb
varimp = cat.get_feature_importance()
proba = cat.predict(test_X)
```

### Key Parameters

| Parameter | Default | Description |
|------|-------|------|
| `model_type` | `"lgb"` | `"lgb"` / `"xgb"` / `"cat"` |
| `n_estimators` | `100` | Number of trees |
| `learning_rate` | `0.1` | Learning rate |
| `max_depth` | `-1` | Maximum depth (-1 = unlimited) |
| `early_stopping_rounds` | `None` | Early-stopping rounds |
| `eval_metric` | `"auc"` | Evaluation metric |

### Using LightGBMModel / XGBoostModel / CatBoostModel Directly

```python
from Modeling_Tool import LightGBMModel, XGBoostModel, CatBoostModel

lgb = LightGBMModel(params={"n_estimators": 200, "learning_rate": 0.05})
lgb.fit(train_X, train_y, val_X, val_y)

xgb = XGBoostModel(params={"n_estimators": 200, "max_depth": 6})
xgb.fit(train_X, train_y, val_X, val_y)

cat = CatBoostModel(params={"n_estimators": 200, "max_depth": 6})
cat.fit(train_X, train_y, val_X, val_y)
```

### Standard CatBoost Modeling — `CatBoostModel`

`CatBoostModel` can be used on its own, or called through the unified interface via `GradientBoostingModel("cat", ...)`.
Its biggest feature is **native handling of categorical features**: specify raw categorical columns (column names or indices) through `cat_features`,
and CatBoost encodes them internally with ordered target statistics, with no need for WOE / one-hot beforehand.

```python
from Modeling_Tool import CatBoostModel

cat = CatBoostModel(
    params={
        "n_estimators": 500,        # -> iterations
        "learning_rate": 0.05,
        "max_depth": 6,             # -> depth
        "l2_leaf_reg": 3.0,
        "early_stopping_rounds": 30,
        "eval_metric": "AUC",
        "cat_features": ["city_grade"],   # raw categorical columns, no encoding needed
    },
)
cat.fit(train_X, train_y, val_X, val_y)

varimp = cat.get_feature_importance()
proba = cat.predict(test_X)
```

!!! note "The WOE pipeline usually doesn't need `cat_features`"

    The standard scorecard flow first WOE-encodes all features into numeric columns, so the model features no longer contain raw categorical columns,
    and `cat_features` can be omitted. You need it only when you feed raw categorical columns directly to CatBoost.

### Quick Training Functions

```python
from Modeling_Tool import lgbm_quick_train, xgbm_quick_train

model = lgbm_quick_train(train_X, train_y, val_X, val_y,
                         params={"n_estimators": 200})
```

### Incremental Learning (Warm-start)

Continue training on new data from an existing model, instead of starting from scratch. `GradientBoostingModel`
provides an interface that is **compatible with both lgb and xgb**: the old model's log-odds output is used as `init_score`
to continue learning on the new data, and at scoring time "old model margin + new model contribution" is fused into the final probability.

#### Differences between lgb / xgb

| Step | XGBoost | LightGBM |
|------|---------|----------|
| Get the raw margin (log-odds) | `predict(X, output_margin=True)` | `predict(X, raw_score=True)` |
| Pass an offset at training time | `fit(X, y, base_margin=...)` | `fit(X, y, init_score=...)` |
| Add the offset directly at prediction time | Natively supported | **Not supported** |

`GradientBoostingModel` hides these differences internally: externally it uses `init_score` uniformly, and the prediction fusion uniformly goes through
`sigmoid(base_margin + new-model raw score)` — the only approach that behaves the same for both frameworks (LightGBM
does not support injecting init_score at prediction time).

#### Three-Step Usage

```python
from Modeling_Tool import GradientBoostingModel

# 1) Get the base margin (log-odds) from the old model, as the starting point for incremental training
base_margin_train = init_model.get_base_margin(train_X)

# 2) Incremental training: pass the base margin as init_score (lgb / xgb both use init_score)
new_model = GradientBoostingModel("xgb", params)   # same for "lgb"
new_model.fit(train_X, train_y, val_X, val_y, init_score=base_margin_train)

# 3) Fused prediction: sigmoid(base_margin + new-model raw score)
base_margin_score = init_model.get_base_margin(score_X)
proba = new_model.predict_with_base_margin(score_X, base_margin_score, return_prob=True)
# return_prob=False returns the fused raw log-odds
```

!!! note "The offset applies only to the training set"

    Consistent with common production implementations, the `init_score` offset is injected only into the training set; the validation set gets no offset, so the early-stopping
    eval metric is evaluated in the "no-offset" space. If you need strict consistency, you can later pass through lgb's
    `eval_init_score` / xgb's `base_margin_eval_set`.

!!! tip "init_model should be a GradientBoostingModel"

    `get_base_margin` / `predict_with_base_margin` are instance
    methods of `GradientBoostingModel`, so the base model `init_model` should also be a `GradientBoostingModel` (not a bare
    `LGBMClassifier` / `XGBClassifier`).

## 3. Backward Variable Elimination — `BackwardVariableEliminator`

Removes variables step by step based on a **cumulative importance threshold**, commonly used for **lightweight variable screening**.

```python
from Modeling_Tool import BackwardVariableEliminator

eliminator = BackwardVariableEliminator(
    model_type="lgb",
    train_data=train_woe,
    validation_data=test_woe,
    oot_data=oot_woe,
    params={"n_estimators": 100, "learning_rate": 0.1},
    y="bad_flag",
    weight_col="sample_wgt",              # training-set weight column
    validation_weight_col="sample_wgt",   # validation-set weight column
    results_output_dir="./output/",   # constructor parameter, not a fit parameter
    modelsave_dir="./models/",
)

eliminator.fit(x=woe_features)
result = eliminator.analyze()
print(result)   # number of variables and performance at each backward-elimination round
```

The underlying `backward_lgbm` / `backward_xgbm` also accept `weight_col` and `validation_weight_col`,
and both training and the performance summary (`get_perf_summary`) are computed by weight.

### How It Works

1. Train one round of LGB with all features, and record each feature's gain importance
2. Remove variables whose **cumulative importance < threshold** (such as `< 0.001`)
3. Repeat 1–2 until the number of remaining variables reaches the lower limit or AUC stops improving

## 4. Model Persistence

```python
from Modeling_Tool import save_model, load_model

save_model(gbm._model.model, "./models/gbm_v1.pkl")

# Load
loaded = load_model("./models/gbm_v1.pkl")
```

## 5. Scoring Function

```python
from Modeling_Tool import scoring

scores = scoring(
    data=new_df,
    model=gbm._model.model,
    varlist=woe_features,
    scr_name="prob",
)
```

## Model Comparison in Practice

```python
from Modeling_Tool import (
    LRMaster, GradientBoostingModel, PerformanceEvaluator,
)

models = {
    "LR":   LRMaster({"C": 1.0}),
    "LGB":  GradientBoostingModel("lgb", {"n_estimators": 200}),
    "XGB":  GradientBoostingModel("xgb", {"n_estimators": 200}),
    "CAT":  GradientBoostingModel("cat", {"n_estimators": 200}),
}

results = {}
for name, model in models.items():
    if name == "LR":
        model.fit(train_woe, woe_features, "bad_flag", weight_col="sample_wgt")
    else:
        model.fit(train_woe[woe_features], train_woe["bad_flag"],
                  test_woe[woe_features],  test_woe["bad_flag"],
                  sample_weight=train_woe["sample_wgt"],
                  eval_sample_weight=test_woe["sample_wgt"])

    raw_model = model._model.model if name != "LR" else model.model
    evaluator = PerformanceEvaluator(
        tgt_name="bad_flag",
        model=raw_model,
        feature_cols=woe_features,
        weight_col="sample_wgt",
    )
    perf = evaluator.add_dataset("train", train_woe) \
                    .add_dataset("test",  test_woe).evaluate()
    results[name] = perf

# Compare KS / AUC
for name, perf in results.items():
    print(f"{name}: AUC={perf['AUC'].mean():.4f}  KS={perf['KS'].mean():.4f}")
```

## FAQ

??? question "LightGBM training raises a `categorical_feature` error"

    Make sure the categorical column has the `category` dtype in the DataFrame, or set it in params:

    ```python
    params["categorical_feature"] = ["city_grade"]
    ```

??? question "CatBoost parameter aliases: `n_estimators` / `max_depth`, or `iterations` / `depth`?"

    `GradientBoostingModel("cat", ...)` and `CatBoostModel` accept **unified parameter names**, so the three GBMs
    can share one configuration: `n_estimators` maps to CatBoost's native `iterations`, and `max_depth` maps to
    `depth`. You can also write CatBoost's native names directly (`iterations` / `depth`); either one works.
    Avoid passing an alias and its native name **together**, to prevent ambiguity; other parameters (`learning_rate`, `l2_leaf_reg`,
    `early_stopping_rounds`, `eval_metric`, etc.) are passed through under CatBoost's native names.

??? question "How does CatBoost handle categorical features?"

    CatBoost supports categorical features natively: specify raw categorical columns (column names or indices) with `cat_features` in params,
    and CatBoost encodes them internally with ordered target statistics, with **no** WOE / one-hot preprocessing needed:

    ```python
    cat = GradientBoostingModel("cat", {
        "n_estimators": 300,
        "cat_features": ["city_grade", "channel"],   # raw categorical columns
    })
    cat.fit(train_X, train_y, val_X, val_y)
    ```

    Note: the training set and the validation / scoring data must contain the same columns; if you are already on the WOE pipeline (all features numeric),
    `cat_features` is usually not needed.

??? question "Variable importance doesn't sum to 1"

    `get_feature_importance(importance_type='gain')` returns normalized relative values,
    and the `sum` should be 1.0; if it returns `split`, the values are weighted by split count.

??? question "After turning on standardization, the coefficients changed a lot / are interpreted differently"

    This is expected. With `standardize=True` the model is trained in the standardized space, and `get_variable_importance()`
    and `get_statsmodel_summary()` give standardized coefficients; if you need coefficients in the original units, turn standardization off
    (`standardize=False`, the default) and retrain.

??? question "How do I do hyperparameter search / cross-validation?"

    `LRMaster` has a built-in holdout-based grid search, `grid_search_params(...)` (see "Hyperparameter Grid Search" above).
    If you want k-fold cross-validation, or hyperparameter search for `GradientBoostingModel`, you can wrap it yourself with sklearn's
    `cross_val_score` (note that for GBM you get the underlying estimator with `model._model.model`, and for LR with `model.model`):

    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model._model.model, X, y, cv=5, scoring="roc_auc")
    ```

    For GBM hyperparameter search, see [GBM Hyperparameter Search](gbm_param_search.md).

??? question "When should I use sample weights? What is the difference between `weight_col` and `sample_weight`?"

    Typical scenarios: correcting sampling bias (such as giving original samples higher weight after oversampling), weighting by amount/balance,
    time-decay weighting, and so on. `weight_col` is resolved from a DataFrame column (recommended, consistent with the evaluation side);
    `sample_weight` takes a numpy array directly. The two cannot be passed together.
    The evaluation side uniformly uses `weight_col`; low-level plotting functions use the `sample_weight` key (see [Model Evaluation](eval.md)).

## LR p-value Backward Elimination (0.6.7+, G07)

```python
CreditModelPipelineConfig(
    train_models=["lr"],
    lr_elimination_mode="pvalue",     # None (default) disables it
    lr_elimination_params={
        "pvalue_threshold": 0.05, "min_features": 1,
        "max_iterations": 20, "tie_breaker": "pvalue",
    },
)
```

After the initial fit, it loops: take the largest coefficient p-value (scipy Fisher information, on exactly the same basis as the final sklearn LR),
and if it exceeds the threshold, drop that feature and refit, until all pass or `min_features`/`max_iterations` is reached.
The trajectory goes into `result.feature_selection_summary["lr_elimination"]` and is written to `lr_pvalue_elimination.csv`;
the feature list of `models["lr"]` is the reduced final one, and evaluation/explanation follow it automatically.
