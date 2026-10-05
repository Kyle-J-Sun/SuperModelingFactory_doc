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

Six more defects came out of documenting the functions that the first pass had described as they behaved. Each has regression tests in `test_doc_audit_regressions.py` that fail without the fix.

| Defect | Fix |
|---|---|
| `ExcelMaster.add_worksheet(auto_fit=True)` raised `AttributeError`, because it called `worksheet.auto_fit()` and xlsxwriter only has `autofit()` | `close_workbook` fits the columns of those sheets with `autofit()` once the data is written. A column is only widened, never narrowed below the width the sheet already has |
| `write_duo_chart` without `y2_list` raised `TypeError`, because it built the second chart from `None` | Only the first chart is built and inserted. With `retChart=True` the result is `(chart1, None)`, and `write_combined_chart` accepts `chart2=None` |
| `multi_subset_wrapper({})` raised `UnboundLocalError`, because the final `return` used a loop variable of a loop that never ran | It returns an empty DataFrame that has only the column `subset_var_name` |
| `plot_boxplot` and `write_boxplot` multiplied the values by 100 whatever `y_percentage` said, so `y_percentage=False` only dropped the percent labels | `y_percentage=True` multiplies by 100 and formats the axis as percent; `False` draws the values as they are |
| `cross_risk_weighted_mean` kept the weight of a NaN row in the denominator while adding nothing to the numerator, so NaN values pulled the mean towards 0 | A NaN row is skipped. A cell whose values are all NaN is NaN |
| The weighted `get_gains_table` divided the regular bins by the weight of the non-special rows and the special rows by the weight of all rows, so `PROP` added up to more than 1 with `spec_values` | `PROP` is the share of the weight of all rows in every row, so it adds up to 1 |

!!! warning "Results that change"
    A direct call of `plot_boxplot` or `write_boxplot` with the default `y_percentage=False` used to draw the values times 100 and now draws them as they are; the hyperparameter box plots of the reports pass `y_percentage=True` and do not change. `cross_risk_weighted_mean` returns a higher mean for every cell that has NaN values. `PROP` of the weighted `get_gains_table` is smaller in the regular bins when `spec_values` matches rows.

## 5. Known Issues

Five issues found during the audit are not fixed yet: `random_state` does not reach the LightGBM and XGBoost models of the credit-model pipeline, `backward_model` runs XGBoost for any value except `"lgb"`, `iv_equal_freq` and `tie_breaker` have no effect, `CorrelationFilter(spec_values=...)` is ignored by the default IV and KS, and `explain_owen` keeps a stale partition tree. The guides and the API reference describe each one as it behaves today. The [FAQ](../faq.md#known-issues) lists them with their workarounds.
