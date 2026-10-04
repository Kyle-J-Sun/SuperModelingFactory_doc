# User Guides

Task-oriented guides, one per topic. Each one explains the task and shows runnable examples of the key APIs. If you have not
run SMF before, start with the [Quickstart](../quickstart.md).

## Guide List

<div class="grid cards" markdown>

- :material-call-split: **[Sample Management](sample.md)**

    Split, stratify, and balance a modeling sample, and search for a good split seed.
    `SampleSplitter`, `StratifiedSampler`, `SampleBalancer`, `select_sample_seed`

- :material-database: **[ODPS Data Extraction](odps.md)**

    Run SQL on Alibaba Cloud MaxCompute, and download and upload tables.
    `ODPSRunner`, `parse_sql_file`, `pull_attributes_in_batch`

- :material-table-search: **[Data Consistency Comparison](proc_compare.md)**

    Compare two DataFrames or large CSV tables, in the spirit of SAS `proc compare`.
    `ProcCompareEngine`, `proc_compare`

- :material-application-brackets: **[Pipeline GUI Schema](pipeline_gui_schema.md)**

    Read pipeline config metadata to build a configuration form, and export or validate configs as YAML.
    `extract_pipeline_schema`, `config_to_yaml`, `validate_pipeline_config`

- :material-chart-bell-curve: **[WOE Encoding](woe.md)**

    Bin variables and encode them with weight of evidence.
    `WOE_Master`, `MonotoneWOEBinner`, `is_monotonic`

- :material-vector-link: **[WOE Binning Engine](woe_binning_engine.md)**

    Reuse one fitted binning engine for PSI, IV, correlation screening, and the final model.
    `as_woe_engine`, `binning_engine`, `woe_binner`

- :material-filter-variant: **[Feature Screening](feature.md)**

    Screen features by stability, information value, and correlation.
    `PSICalculator`, `VarExtractionInsights`, `CorrelationFilter`, `feature_screen`

- :material-brain: **[Model Training](model.md)**

    Train logistic regression and gradient-boosting models, optionally with sample weights, and eliminate variables.
    `LRMaster`, `GradientBoostingModel`, `BackwardVariableEliminator`

- :material-tune: **[GBM Hyperparameter Search](gbm_param_search.md)**

    Search LightGBM, XGBoost, or CatBoost parameters with a grid or Optuna, judged on INS, OOS, and OOT datasets.
    `GradientBoostingModel.param_search`

- :material-archive-cog: **[Model Registry and Versioning](model_registry.md)**

    Save a model with metadata (version, features, WOE mapping path, metrics) and load it back.
    `save_model`, `load_model`, `load_model_metadata`

- :material-chart-line: **[Model Evaluation](eval.md)**

    KS, AUC, Gains tables, score comparison, and slice-by-slice evaluation, weighted or unweighted.
    `PerformanceEvaluator`, `GainsTableCalculator`, `Model_Evaluation_Tool`, `EvaluationPipeline`

- :material-lightbulb-on: **[Model Explainability](explainability.md)**

    Explain a model with SHAP, Owen value, PDP, ICE, ALE, and LIME.
    `ModelExplainer`

- :material-account-cancel: **[Reject Inference and Distribution Adaptation](reject_inference.md)**

    Correct selection bias in a sample of approved applicants.
    `RejectInferenceFactory`, `DistributionAdaptation`

- :material-shield-check: **[Online/Offline Consistency Check](uat.md)**

    Check that online and offline scores and features agree before go-live.
    `UATConsistencyChecker`, `UATConfig`

- :material-file-excel: **[Excel Report Generation](excel_report.md)**

    Write formatted Excel workbooks from modeling results.
    `ExcelMaster`, `Template`, `Report`

</div>

Release-by-release changes are collected under the [ChangeLog](../changelog/index.md) tab.

## Suggested Reading Order

1. Run the [Quickstart](../quickstart.md) end to end.
2. Read [End-to-End Modeling Pipeline](../pipeline.md) for the full manual workflow, then the guides in workflow order:
   samples, WOE, feature screening, models, evaluation, explainability, and Excel reports.
3. If you want PSI, IV, correlation screening, and the final model to share one binning, read
   [WOE Encoding](woe.md) and then [WOE Binning Engine](woe_binning_engine.md).
4. To run the whole workflow in one call, see [Top-Level Pipelines](../pipeline_one_click.md).
5. To look up a signature, go to the [API Reference](../api/index.md).
