# GBM Hyperparameter Search

`GradientBoostingModel.param_search(...)` provides **INS / OOS / OOT holdout hyperparameter search** for LightGBM / XGBoost, aligned with `LRMaster.grid_search_params(...)`.

It is not k-fold CV but the multi-time-window validation commonly used in risk modeling: each candidate parameter set is fitted on the training set, AUC is then computed on every dataset in `eval_sets`, and the best combination is chosen according to the specified objective.

## 1. Grid Search

With `engine="grid"`, `search_space` uses a plain grid: give each parameter a list of candidate values, and the Cartesian product is taken internally.

```python
from Modeling_Tool import GradientBoostingModel

base_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "early_stopping_rounds": 30,
    "eval_metric": "auc",
}

gbm = GradientBoostingModel("lgb", base_params)

results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    },
    engine="grid",
    objective="oot_gap_penalized",
    primary_set="oot",
    gap_ref_sets=["ins", "oos"],
    refit=True,
)

print(gbm.best_params_)
print(gbm.search_results_.head())
```

The returned `results` is sorted by `score` in descending order and contains:

- Parameter columns (such as `num_leaves`, `learning_rate`, `max_depth`)
- AUC for each eval set: `AUC_ins` / `AUC_oos` / `AUC_oot`
- `gap` (only under `oot_gap_penalized`)
- `score`

## 2. Optuna Search

With `engine="optuna"`, you need an extra install:

```bash
pip install supermodelingfactory[optuna]
```

`search_space` uses a compact search-space definition:

```python
gbm = GradientBoostingModel("xgb", {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "eval_metric": "auc",
})

results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={
        "max_depth": ("int", 2, 6),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_weight": ("categorical", [1, 5, 10]),
    },
    engine="optuna",
    n_trials=50,
    primary_set="oot",
    random_state=42,
    refit=True,
)
```

A dict form is also supported:

```python
search_space = {
    "max_depth": {"type": "int", "low": 2, "high": 6},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
}
```

## 3. Objective

Consistent with `LRMaster.grid_search_params(...)`, three selection objectives are supported:

| objective | Selection criterion |
|---|---|
| `'oot_gap_penalized'` (default) | `AUC[primary] - abs(mean(AUC[gap_refs]) - AUC[primary])`, which both rewards primary-set performance and penalizes the overfitting gap |
| `'max_primary'` | Directly maximize `AUC[primary]` |
| callable | Custom `f(metric_dict) -> float`, where `metric_dict` is `{set_name: AUC}` |

Custom objective example:

```python
def stable_oot(metrics):
    return metrics["oot"] - 0.5 * abs(metrics["ins"] - metrics["oot"])

results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={"max_depth": [3, 4, 5]},
    objective=stable_oot,
    primary_set="oot",
)
```

## 4. Validation Set

GBM training itself needs a validation set. `param_search` selects one in the following order by default:

1. `eval_sets["oos"]`, if present
2. `eval_sets["validation"]` or `eval_sets["valid"]`, if present
3. `primary_set`

You can also specify it explicitly:

```python
gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={"max_depth": [3, 4, 5]},
    validation_set="oos",
)
```

## 5. Sample Weights

`param_search` supports `weight_col` (the training set) and `eval_weight_col` (the weight column in each eval-set DataFrame).
Candidate models use the corresponding weights to compute weighted AUC during both training and holdout scoring:

```python
results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    },
    engine="grid",
    objective="oot_gap_penalized",
    primary_set="oot",
    gap_ref_sets=["ins", "oos"],
    weight_col="sample_wgt",        # training-set weight column (must be in data)
    eval_weight_col="sample_wgt",   # weight column for each eval set
    refit=True,
)

print(gbm.best_params_)
print(gbm.search_results_[["max_depth", "learning_rate", "AUC_oot", "score"]].head())
```

Symmetric with `LRMaster.grid_search_params`: `weight_col` controls candidate fitting, and `eval_weight_col` controls
scoring on each holdout. You can also pass `sample_weight` / `eval_sample_weight` arrays through `fit_kwargs`.

!!! note "The weight column must exist in every relevant DataFrame"

    `weight_col` must exist in `data`; `eval_weight_col` must exist in every
    DataFrame in `eval_sets`, otherwise the search raises a `KeyError` at startup.

## 6. Return Value and Side Effects

- **Returns**: a `pandas.DataFrame` sorted by `score` in descending order
- **Writes**: `gbm.best_params_` and `gbm.search_results_`
- **Updates**: merges the best parameters into `gbm.params`
- **Refit**: with `refit=True`, the current `gbm` is retrained on `data` with the best parameters

!!! note "Only AUC is supported for now"

    `metric` currently supports only `'auc'`. If you need KS / logloss / brier, you can extend the search results through a callable objective for now; more metrics will be built in in later versions.

!!! tip "Difference from the LR search"

    The LR search method is named `grid_search_params(...)`, while GBM uses `param_search(...)` because it supports both the grid and Optuna engines.
