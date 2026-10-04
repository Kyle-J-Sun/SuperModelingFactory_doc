# GBM Hyperparameter Search

`GradientBoostingModel.param_search(...)` tunes a LightGBM, XGBoost, or CatBoost model with a **holdout search** that
matches how scorecards are validated. Each candidate parameter set is fitted on the training sample, scored by AUC on every
sample in `eval_sets` (typically INS, OOS, and OOT), and ranked by an objective that can penalize the gap between them. It
is not k-fold cross-validation. `LRMaster.grid_search_params(...)` is the logistic-regression counterpart and uses the same
objectives.

Two engines are available: an exhaustive grid (the default) and Optuna's TPE sampler.

`param_search` is attached to `GradientBoostingModel` when `Modeling_Tool` is imported, so there is nothing to enable.

!!! warning "Build the final model yourself"

    In 0.8.2, `refit=True` retrains with the parameters the model was created with, not the best ones, and `best_params_`
    stores integer-valued parameters as floats (`15.0`), which LightGBM and XGBoost reject. The examples below therefore use
    `refit=False` and train the final model as shown in [Train the Final Model](#train-the-final-model).

## Example data

Every snippet on this page runs top to bottom in one Python session. The setup creates a synthetic portfolio and a
time-based INS / OOS / OOT split.

```python
import numpy as np
import pandas as pd

from Modeling_Tool import GradientBoostingModel

rng = np.random.default_rng(42)
n = 8000
months = [f"2025-{m:02d}" for m in range(1, 10)]

df = pd.DataFrame({
    "apply_month": rng.choice(months, n),
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
    "sample_wgt":  rng.uniform(0.5, 2.0, n),          # e.g. balance or correction weight
})
logit = -2.2 - 0.02 * (df["score_b"] - 600) + 0.5 * df["n_overdue"] - 0.8 * df["utilization"]
df["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "utilization", "n_overdue"]
ins = df[df["apply_month"] <= "2025-06"]       # fitting sample
oos = df[df["apply_month"] == "2025-07"]       # validation sample (early stopping)
oot = df[df["apply_month"] >= "2025-08"]       # out-of-time sample
eval_sets = {"ins": ins, "oos": oos, "oot": oot}

base_params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
    "early_stopping_rounds": 20,      # required for LightGBM
    "eval_metric": "auc",
    "verbose": -1,
    "n_jobs": 1,                      # threads per candidate fit
}
```

## Parameters

`param_search(data, varlist, tgt_name, eval_sets, search_space, engine="grid", objective="oot_gap_penalized", primary_set=None, gap_ref_sets=None, metric="auc", validation_set=None, n_trials=50, refit=True, verbose=True, fit_kwargs=None, random_state=None, weight_col=None, eval_weight_col=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `data` | required | Training `DataFrame`. Every candidate is fitted on it |
| `varlist` | required | Feature column names, present in `data` and in every eval set |
| `tgt_name` | required | Target column, present in `data` and in every eval set |
| `eval_sets` | required | Ordered `{name: DataFrame}` of the samples to score by AUC, for example `{"ins": ins, "oos": oos, "oot": oot}`. Must not be empty |
| `search_space` | required | `{parameter: candidates}` for `engine="grid"`; `{parameter: spec}` for `engine="optuna"` (see [Optuna Search](#2-optuna-search)). Names are the backend's own parameter names |
| `engine` | `"grid"` | `"grid"` or `"optuna"` |
| `objective` | `"oot_gap_penalized"` | `"oot_gap_penalized"`, `"max_primary"`, or a callable (see [Objective](#3-objective)) |
| `primary_set` | `None` | The eval set to maximize. `None` means the **last** key of `eval_sets` |
| `gap_ref_sets` | `None` | Eval sets whose mean AUC is the gap reference. `None` means every set except `primary_set` |
| `metric` | `"auc"` | Only `"auc"` is supported; anything else raises `ValueError` |
| `validation_set` | `None` | Eval set used for early stopping (see [Validation Set](#4-validation-set)) |
| `n_trials` | `50` | Number of Optuna trials; ignored by the grid engine |
| `refit` | `True` | Retrain `self` on `data` after the search. Read the warning above before relying on it |
| `verbose` | `True` | Print `param_search(grid): N combinations` before a grid search |
| `fit_kwargs` | `None` | Extra keyword arguments for `GradientBoostingModel.fit`, for example `init_score` |
| `random_state` | `None` | Seeds Optuna's TPE sampler so the same trials are proposed; ignored by the grid engine |
| `weight_col` | `None` | Training weight column in `data` (see [Sample Weights](#5-sample-weights)) |
| `eval_weight_col` | `None` | Weight column in every eval set, used for early stopping and for the weighted AUC |

## 1. Grid Search

With `engine="grid"`, give each parameter a list of candidate values. SMF fits the Cartesian product.

```python
gbm = GradientBoostingModel("lgb", base_params)

grid_space = {
    "num_leaves": [7, 15],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
}

results = gbm.param_search(
    data=ins,
    varlist=features,
    tgt_name="bad_flag",
    eval_sets=eval_sets,
    search_space=grid_space,
    engine="grid",
    objective="oot_gap_penalized",
    primary_set="oot",
    gap_ref_sets=["ins", "oos"],
    refit=False,
)

print(results.head())
print(gbm.best_params_)
print(gbm.search_results_.shape)
```

`results` has one row per candidate, sorted by `score` in descending order, with these columns:

- one column per searched parameter (`num_leaves`, `learning_rate`, `max_depth`)
- `AUC_<name>` for each eval set (`AUC_ins`, `AUC_oos`, `AUC_oot`), rounded to 5 decimals
- `gap`: the mean AUC of `gap_ref_sets` minus the AUC of `primary_set`; present only for the `"oot_gap_penalized"`
  objective with at least one reference set
- `score`: the value of the objective

## 2. Optuna Search

`engine="optuna"` needs the `optuna` extra:

```bash
pip install 'supermodelingfactory[optuna]'
```

Each `search_space` value is a tuple or a dictionary that describes how to sample the parameter:

| Parameter type | Tuple form | Dictionary form |
|---|---|---|
| Integer | `("int", low, high)` or `("int", low, high, step)` | `{"type": "int", "low": 2, "high": 6, "step": 1, "log": False}` |
| Float | `("float", low, high)` or `("float", low, high, "log")` | `{"type": "float", "low": 0.01, "high": 0.2, "log": True}`; `step` is optional |
| Categorical | `("categorical", [choice, ...])` | `{"type": "categorical", "choices": [choice, ...]}` |

!!! note "Tuple-form limits"

    In the tuple form, `("int", low, high, "log")` ignores `"log"`; write `("int", low, high, 1, "log")` or use the
    dictionary form for log-scaled integers. A float tuple cannot take a `step`; use the dictionary form.

```python
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)       # Optuna logs one line per trial otherwise

xgb_model = GradientBoostingModel("xgb", {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
    "early_stopping_rounds": 20,
    "n_jobs": 1,
})

optuna_space = {
    "max_depth": ("int", 2, 6),
    "learning_rate": ("float", 0.01, 0.2, "log"),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.6, 1.0),
    "min_child_weight": ("categorical", [1, 5, 10]),
}

optuna_results = xgb_model.param_search(
    data=ins,
    varlist=features,
    tgt_name="bad_flag",
    eval_sets=eval_sets,
    search_space=optuna_space,
    engine="optuna",
    n_trials=8,
    primary_set="oot",
    random_state=42,
    refit=False,
)
print(optuna_results.head())
```

The table has the same columns as the grid result plus a leading `trial_number`. The dictionary form of the same space:

```python
optuna_space_dict = {
    "max_depth": {"type": "int", "low": 2, "high": 6},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
}
```

## 3. Objective

Three selection objectives are supported, the same as in `LRMaster.grid_search_params(...)`:

| `objective` | Selection criterion |
|---|---|
| `"oot_gap_penalized"` (default) | `AUC[primary] - abs(mean(AUC[gap_ref_sets]) - AUC[primary])`: rewards the primary set and penalizes the gap to the reference sets |
| `"max_primary"` | `AUC[primary]` |
| callable | Your function `f(metric_dict) -> float`, where `metric_dict` maps each eval-set name to its AUC. Higher is better |

With `"max_primary"` or a callable there is no `gap` column. A custom objective:

```python
def stable_oot(metrics):
    return metrics["oot"] - 0.5 * abs(metrics["ins"] - metrics["oot"])


custom_results = gbm.param_search(
    data=ins,
    varlist=features,
    tgt_name="bad_flag",
    eval_sets=eval_sets,
    search_space={"max_depth": [3, 4, 5]},
    objective=stable_oot,
    primary_set="oot",
    refit=False,
)
print(custom_results[["max_depth", "AUC_ins", "AUC_oot", "score"]])
```

## 4. Validation Set

Every candidate is trained with early stopping, so it needs a validation sample. `param_search` picks the first match in
this order:

1. `validation_set`, if you pass it
2. `eval_sets["oos"]`
3. `eval_sets["validation"]` or `eval_sets["valid"]`
4. `primary_set`

```python
gbm.param_search(
    data=ins,
    varlist=features,
    tgt_name="bad_flag",
    eval_sets=eval_sets,
    search_space={"max_depth": [3, 4, 5]},
    validation_set="oos",
    refit=False,
)
```

!!! warning "Keep early stopping away from the primary set"

    If `eval_sets` has no `oos`, `validation`, or `valid` entry and you pass no `validation_set`, the primary set
    (for example OOT) also drives early stopping, so its AUC is optimistic. Keep a separate validation sample.

LightGBM requires `early_stopping_rounds` in the model parameters (`KeyError` otherwise). For XGBoost, early stopping
monitors logloss: SMF does not pass `eval_metric` to XGBoost.

## 5. Sample Weights

`weight_col` names the training weight column in `data`, and `eval_weight_col` names the weight column in **every** eval-set
`DataFrame`. Candidates are fitted with the training weights, early-stopped with the evaluation weights, and scored with
**weighted AUC** on each eval set:

```python
weighted_results = gbm.param_search(
    data=ins,
    varlist=features,
    tgt_name="bad_flag",
    eval_sets=eval_sets,
    search_space={
        "num_leaves": [7, 15],
        "learning_rate": [0.05, 0.1],
    },
    engine="grid",
    objective="oot_gap_penalized",
    primary_set="oot",
    gap_ref_sets=["ins", "oos"],
    weight_col="sample_wgt",          # training weight column (must be in data)
    eval_weight_col="sample_wgt",     # weight column in each eval set
    refit=False,
)

print(weighted_results[["num_leaves", "learning_rate", "AUC_oot", "score"]].head())
```

Like `LRMaster.grid_search_params`, `weight_col` controls candidate fitting and `eval_weight_col` controls scoring on each
holdout. Use the same weights in training and evaluation. `fit_kwargs={"sample_weight": array}` also weights the fit, but
it does not weight the AUC: only `eval_weight_col` does that, and `weight_col` takes precedence over a `sample_weight` in
`fit_kwargs`.

!!! note "The weight column must exist in every relevant DataFrame"

    `weight_col` must exist in `data` and `eval_weight_col` in every `DataFrame` of `eval_sets`. Otherwise the search
    raises a `KeyError` before it fits anything.

## 6. Return Value and Side Effects

- **Returns** a `pandas.DataFrame` sorted by `score` in descending order (also stored as `gbm.search_results_`).
- **Sets** `gbm.best_params_`: a dictionary with the searched parameters of the best row.
- **Merges** the best parameters into `gbm.params`.
- **Refits** (`refit=True`) `gbm` on `data`, validated on the validation set. With `refit=False`, `gbm` is left as it was; a
  model that was never fitted stays unfitted.

!!! warning "Known limitations in 0.8.2"

    - `refit=True` and any later `gbm.fit(...)` train with the parameters the instance was created with. The best values
      are stored in `gbm.params`, but the fitted model does not use them.
    - `best_params_` (and the merged `gbm.params`) hold integer-valued parameters as floats when every searched parameter
      is numeric, for example `{"num_leaves": 15.0, "max_depth": 3.0}`. LightGBM raises `Parameter num_leaves should be of
      type int, got "15.0"` and XGBoost raises `Invalid Parameter format for max_depth expect int but value='3.0'` when you
      train with them.

!!! note "Only AUC is supported"

    `metric` accepts only `"auc"`. To rank by something else (KS, log loss, Brier score), pass a callable `objective`; it
    still receives the AUC per eval set.

## Train the Final Model

Read the best row from the results table, which keeps the original types, and fit a new model with it:

```python
best_params = results.iloc[[0]][list(grid_space)].to_dict("records")[0]
print(best_params)                                  # e.g. {'num_leaves': 15, 'learning_rate': 0.05, 'max_depth': 3}

final = GradientBoostingModel("lgb", {**base_params, **best_params})
final.fit(ins[features], ins["bad_flag"], oos[features], oos["bad_flag"])
print(final.predict(oot[features])[:5])
```

`results.iloc[[0]]` is the best row as a one-row frame, and `to_dict("records")` keeps each column's type, so integer
parameters stay integers. The same pattern works for Optuna results: `optuna_results.iloc[[0]][list(optuna_space)]`.

## Troubleshooting

??? question "`KeyError: 'early_stopping_rounds'`"

    LightGBM models need `early_stopping_rounds` in their parameters, and `param_search` fits every candidate with early
    stopping. Add it to the parameters you pass to `GradientBoostingModel`.

??? question "`ImportError` about `optuna`"

    Install the extra: `pip install 'supermodelingfactory[optuna]'`.

??? question "`LightGBMError: Parameter num_leaves should be of type int, got \"15.0\"`"

    You trained with `gbm.best_params_` or `gbm.params`, which hold integer parameters as floats. Build the parameters from
    the results table as shown in [Train the Final Model](#train-the-final-model), or cast them with `int(...)`.

??? question "`AttributeError: 'NoneType' object has no attribute 'predict_proba'` after the search"

    With `refit=False` the model is not fitted. Train a final model as shown in
    [Train the Final Model](#train-the-final-model).

??? question "Optuna prints one line per trial"

    Call `optuna.logging.set_verbosity(optuna.logging.WARNING)` before the search.

??? question "How do I search an XGBoost or CatBoost model?"

    Create the model with `"xgb"` or `"cat"` and use that backend's parameter names in `search_space`, for example
    `"depth"` and `"l2_leaf_reg"` for CatBoost. The search itself is identical.
