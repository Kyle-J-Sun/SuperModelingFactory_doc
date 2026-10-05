# Unreleased

Changes made after 0.8.2 that ship with the next release. The version number is not decided yet.

## 1. All Text Is English

Every docstring, comment, and user-visible string is English, and the four repositories contain no Chinese text. The code is unchanged: the translation touches comments, docstrings, and string contents only.

!!! warning "Code that matches on message text"
    Log, warning, and exception messages, Excel sheet names, and report headings changed language. Scripts that filter logs, use `pytest.raises(match=...)`, or read workbook sheets by name must use the new wording.

| Area | Now |
|---|---|
| Docstrings and comments | English. The [API Reference](../api/index.md) renders the English text |
| Log, warning, and exception messages | English. For example the binner logs `[MonotoneWOEBinner] Fitting 3 features ...`, and the `feature_screen` fallback warning reads `[feature_screen] stage 'psi' eliminated all 40 variables; keeping all of them (on_empty_stage='keep_all_warn')` |
| `MonotoneWOEBinner.export_woe_report` | The sheets are named `WOE Bin Details` and `WOE Bin Charts`, with English headers |
| `Report` templates | English headings, for example *Multi-Model Evaluation (Untuned)*, *Final Model Evaluation*, and *Feature Importance Evaluation* |
| Plot labels | English, for example the axes of `plot_var_reduction` |
| Pipeline GUI schema | `FieldMeta.label`, `description`, and `group`, the registry card text, and the `validate_pipeline_config` messages are English; field names, option values, and defaults are unchanged |

## 2. Fixes Found While Verifying the Documentation

The guides and the docstrings were checked against the installed package, and eleven defects surfaced. Each has a regression test in `test_doc_audit_regressions.py` that fails without the fix.

| Defect | Fix |
|---|---|
| Several `Report.Report_Tool` functions used `pd` and `os` without importing them, so they raised `NameError` when called | The imports are in place |
| The `SampleBalancer` methods `nearmiss`, `tomek`, and `enn` raised `TypeError` with imbalanced-learn 0.14 and later, which dropped `random_state` from those samplers | `random_state` is passed only to samplers that still accept it |
| `get_gains_table`, `get_gains_table_by_cust_metrics`, and `get_perf_summary` iterated over the unfiltered group list after dropping groups below `min_data_size`, so a small group made the next group appear twice and the last one disappear | They iterate over the filtered groups |
| `upload_score`, `h2o_apply_regex`, and `get_dtypes_file(ck_format=True)` raised `NameError`, and the all-missing log call in scoring made `logging` report a formatting error | The helpers import what they use, `ck_format=True` raises `NotImplementedError` with an explanation, and the log call has its placeholder |
| `calibrate` of `LightGBMModel`, `XGBoostModel`, `CatBoostModel`, and `GradientBoostingModel` defaulted to `cv='prefit'`, which scikit-learn 1.8 removed, so every call raised `InvalidParameterError` | The fitted estimator is wrapped in `FrozenEstimator` when it exists; older scikit-learn keeps `cv='prefit'` |
| `BackwardEliminationAnalyzer.get_perf_trend` returned `None` for every value, because it only read dictionary summaries while `BackwardVariableEliminator` stores DataFrames | It reads the stored DataFrames |
| `GradientBoostingModel.param_search` turned integer parameters into floats (`num_leaves=7.0`, which LightGBM rejects), and `refit=True` kept fitting with the pre-search parameters | `best_params_` keeps native types and the model uses the merged parameters |
| `TextEncryptor.decrypt` raised `ValueError` for any text with a non-ASCII character, because `encrypt` stored the plaintext length in characters and `decrypt` compared it with the length in bytes | The length prefix is the byte length. Ciphertexts of ASCII text are unchanged, so existing ones keep decrypting |
| `ExcelFormat.base_filepath` used `str.strip`, which removes characters and not a prefix (`out2/report.xlsx` gave `ut2/`) | It is the directory part of the path, including the trailing separator |
| `compute_overfitting_shift` told the caller that unequal column counts "are the the same" | The message says "are not the same" |
| `ModelExplainer.lime_explain_instance` and `lime_global_importance` raised `AttributeError: 'NoneType' object has no attribute 'copy'` when neither `X_train` nor `background_data` was available, because the intended `ValueError` was checked too late | The check runs first, so the message is `LIME requires X_train or background_data` |

## 3. Complete Docstrings and Parameter Tables

Every public class, function, and method of `Modeling_Tool`, `ExcelMaster`, and `Report` now documents exactly the parameters of its signature, in signature order, in the NumPy layout that the [API Reference](../api/index.md) renders. The Config dataclasses of the Pipelines list every field with its type, default, and behavior, and the parameter tables on the [Top-Level Pipelines](../pipeline_one_click.md) page have a row for every field (34 were missing). The pytest test `test_public_docstring_parameters.py` keeps the docstrings in step with the signatures.

## 4. Further Fixes

Eleven more defects came out of documenting the functions that the first pass had described as they behaved. Each has regression tests in `test_doc_audit_regressions.py` that fail without the fix.

| Defect | Fix |
|---|---|
| `ExcelMaster.add_worksheet(auto_fit=True)` raised `AttributeError`, because it called `worksheet.auto_fit()` and xlsxwriter only has `autofit()` | `close_workbook` fits the columns of those sheets with `autofit()` once the data is written. A column is only widened, never narrowed below the width the sheet already has |
| `write_duo_chart` without `y2_list` raised `TypeError`, because it built the second chart from `None` | Only the first chart is built and inserted. With `retChart=True` the result is `(chart1, None)`, and `write_combined_chart` accepts `chart2=None` |
| `multi_subset_wrapper({})` raised `UnboundLocalError`, because the final `return` used a loop variable of a loop that never ran | It returns an empty DataFrame that has only the column `subset_var_name` |
| `plot_boxplot` and `write_boxplot` multiplied the values by 100 whatever `y_percentage` said, so `y_percentage=False` only dropped the percent labels | `y_percentage=True` multiplies by 100 and formats the axis as percent; `False` draws the values as they are |
| `cross_risk_weighted_mean` kept the weight of a NaN row in the denominator while adding nothing to the numerator, so NaN values pulled the mean towards 0 | A NaN row is skipped. A cell whose values are all NaN is NaN |
| The weighted `get_gains_table` divided the regular bins by the weight of the non-special rows and the special rows by the weight of all rows, so `PROP` added up to more than 1 with `spec_values` | `PROP` is the share of the weight of all rows in every row, so it adds up to 1 |
| `CorrelationFilter(spec_values=...)` stored the special values but never passed them to the `VarExtractionInsights` that computes the IV and KS deciding each correlated group, so on the default path the argument was silently ignored; it worked only with `woe_binner` or `woe_engine="monotone"` | The special values reach the IV and KS calculation on the default path too, in `CorrelationFilter` and in `Feature_Insights.CorrelationFilter`. The default `spec_values=[]` gives the same result as before |
| `ModelExplainer.explain_owen` kept the first SHAP `PartitionExplainer` unless `rebuild=True` or `model_output` changed, so a new `prior_groups`, `coalition_structure`, or `background_data` updated `coalition_structure_` while the Owen values were still computed on the old partition tree (a switch between two groupings gave values that differed from a fresh explainer by up to 0.019 in my check) | The cached explainer is reused only while the coalition structure (features and linkage), the background values, and `model_output` are unchanged; any change rebuilds it. `rebuild=True` still forces a new one |
| `WOE_Master` gave a bin with only goods or only bads a WOE and IV of `-inf` or `+inf`, so a feature with such a bin made LR and XGBoost training fail (`ValueError` and `XGBoostError`) while LightGBM and CatBoost silently took the infinite value. The default `woe_engine="equal_freq"` is affected | Such a bin gets a finite WOE with the `eps` of `MonotoneWOEBinner`: `ln((bad_pct + eps) / (good_pct + eps))` with `eps=1e-06`. Every other WOE and IV value is unchanged, and `calc_woe`, `calc_iv`, and the Gains tables still return infinity for a zero share |
| `CreditModelPipelineConfig.random_state` never reached the LightGBM and XGBoost models, because their built-in parameters fixed `random_state` at 42; the Optuna candidates and the backward-elimination proxy were fixed at 42 too | The built-in LightGBM and XGBoost parameters carry no seed. `random_state` seeds every GBM model the pipeline builds (final models, Optuna candidates, backward proxy), and a `random_state` in `model_params[name]` still wins. The default seed 42 gives the same models as before |
| `CreditModelPipelineConfig.backward_model` ran the XGBoost elimination for every value except the exact lower-case `"lgb"`, so `"LGB"`, `"lr"`, and `"cat"` silently ran XGBoost, and the GUI schema offered `"lr"` and `"cat"` | `run()` accepts `"lgb"` and `"xgb"` in any case and raises `ValueError` for anything else while `backward_enabled` is on. The GUI schema offers only `"lgb"` and `"xgb"`, and `validate_pipeline_config` reports other values |

!!! warning "Results that change"
    A direct call of `plot_boxplot` or `write_boxplot` with the default `y_percentage=False` used to draw the values times 100 and now draws them as they are; the hyperparameter box plots of the reports pass `y_percentage=True` and do not change. `cross_risk_weighted_mean` returns a higher mean for every cell that has NaN values. `PROP` of the weighted `get_gains_table` is smaller in the regular bins when `spec_values` matches rows. A `backward_model` other than `"lgb"` or `"xgb"` now raises `ValueError` when `run()` starts, where it used to run XGBoost; `"LGB"` now runs LightGBM. A credit-model run with a `random_state` other than 42 now trains different LightGBM and XGBoost models (and Optuna candidates and backward proxies) than before; runs with the default seed are unchanged. A feature that has a bin with only goods or only bads now has a finite WOE and IV (it was infinite), so its IV can rank and screen differently; LR and XGBoost models that used to fail now train, LightGBM models on such features change slightly, and CatBoost models are unchanged. Tables with no such bin give identical results. `explain_owen` called again with another `prior_groups`, `coalition_structure`, or `background_data` now returns the Owen values of the new grouping or background; they used to come from the first grouping. Calls with unchanged inputs still reuse the cached explainer. A `CorrelationFilter` called with non-empty `spec_values` now computes its IV and KS with those values in rows of their own, so the IV of a variable whose signal sits in its special values drops and the survivor of a correlated group can change (in my test data the kept variable switched from `a` to `b`); calls with the default `spec_values=[]` are unchanged.

## 5. Known Issues

One issue found during the audit is not fixed yet: `iv_equal_freq` and `tie_breaker` have no effect. The guides and the API reference describe each one as it behaves today. The [FAQ](../faq.md#known-issues) lists them with their workarounds.
