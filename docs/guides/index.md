# User Guides

Hands-on guides organized by modeling scenario. Each one covers **background → key APIs → complete code → FAQ**.

## Guide List

<div class="grid cards" markdown>

- :material-call-split: **[Sample Management](sample.md)**

    `SampleSplitter` / `StratifiedSampler` / `SampleBalancer` / `select_sample_seed`

- :material-database: **[ODPS Data Extraction](odps.md)**

    `ODPSRunner` / `parse_sql_file` / `pull_attributes_in_batch`

- :material-table-search: **[Data Consistency Comparison](proc_compare.md)**

    `ProcCompareEngine` / `proc_compare` / consistency checks for DataFrames and large CSV tables

- :material-application-brackets: **[Pipeline GUI Schema](pipeline_gui_schema.md)**

    `extract_pipeline_schema` / `config_to_yaml` / `validate_pipeline_config` / GUI form metadata

- :material-chart-bell-curve: **[WOE Encoding](woe.md)**

    `WOE_Master` / `MonotoneWOEBinner` / `is_monotonic`

- :material-vector-link: **[WOE Binning Engine](woe_binning_engine.md)**

    `as_woe_engine` / `binning_engine` / `woe_binner` / unified Master-Monotone protocol

- :material-filter-variant: **[Feature Screening](feature.md)**

    `PSICalculator` / `VarExtractionInsights` / `CorrelationFilter`

- :material-brain: **[Model Training](model.md)**

    `LRMaster` / `GradientBoostingModel` / `BackwardVariableEliminator`(with `weight_col`)

- :material-tune: **[GBM Hyperparameter Search](gbm_param_search.md)**

    `GradientBoostingModel.param_search` / grid search / Optuna / INS-OOS-OOT holdout(with weighted AUC)

- :material-archive-cog: **[Model Registry and Versioning](model_registry.md)**

    `save_model` / `load_model` / `load_model_metadata` / model metadata artifact

- :material-chart-line: **[Model Evaluation](eval.md)**

    `PerformanceEvaluator` / `GainsTableCalculator` / `EvaluationPipeline`(with weighted Gains / KS / AUC)

- :material-lightbulb-on: **[Model Explainability](explainability.md)**

    `ModelExplainer` (SHAP / Owen Value / PDP / ICE / ALE / LIME)

- :material-account-cancel: **[Reject Inference and Distribution Adaptation](reject_inference.md)**

    `RejectInferenceFactory` / `DistributionAdaptation`

- :material-shield-check: **[Online/Offline Consistency Check](uat.md)**

    `UATConsistencyChecker` / `UATConfig`

- :material-file-excel: **[Excel Report Generation](excel_report.md)**

    `ExcelMaster` / `Template` / `Report`

</div>

!!! tip "Version change notes"

    Release-by-release changes are collected under the [ChangeLog](../changelog/index.md) tab in the top-level navigation.

## Suggested Reading Order

!!! tip "Path for newcomers"

    1. Start with the [Quickstart](../quickstart.md) and run it end to end
    2. Then read [End-to-End Pipelines](../pipeline.md) to understand the overall flow
    3. If you use monotone binning, read the [WOE Binning Engine](woe_binning_engine.md) guide first
    4. Then consult the other guides as needed

!!! info "API lookup"

    To look up a specific function signature at any time, go to the [API Reference](../api/index.md), which is generated automatically from source docstrings.
