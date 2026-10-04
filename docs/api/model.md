# Modeling_Tool.Model

Model training layer: logistic regression (`LRMaster`), gradient boosting with LightGBM, XGBoost, and CatBoost
(`GradientBoostingModel`), and backward variable elimination. The subpackage loads on first access of one of its names, and
the boosting libraries load when you build a model of that type.

## Logistic Regression: `LRM_Tool`

::: Modeling_Tool.Model.LRM_Tool

## Gradient Boosting Models: `GBM_Tool`

::: Modeling_Tool.Model.GBM_Tool

### `GradientBoostingModel.param_search`

`GBM_Search_Tool` attaches `param_search` to `GradientBoostingModel` when `Modeling_Tool.Model` is imported, so the
generated reference above does not list it. It runs a grid or Optuna search over holdout datasets, stores the winning
parameters in `best_params_` and the full table in `search_results_`, and refits the model when `refit=True`. The
signature, without `self`:

```text
param_search(data, varlist, tgt_name, eval_sets, search_space, engine='grid', objective='oot_gap_penalized',
             primary_set=None, gap_ref_sets=None, metric='auc', validation_set=None, n_trials=50, refit=True,
             verbose=True, fit_kwargs=None, random_state=None, weight_col=None, eval_weight_col=None)
```

See [GBM Hyperparameter Search](../guides/gbm_param_search.md) for the arguments and examples. `engine='optuna'` needs the
`optuna` extra.

## Backward Variable Elimination: `Backward_Tool`

::: Modeling_Tool.Model.Backward_Tool
