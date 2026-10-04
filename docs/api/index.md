# API Reference

The reference pages list the public classes and functions of each subpackage with their signatures, defaults, and
docstrings. They are generated from the SMF source by [mkdocstrings](https://mkdocstrings.github.io/python/), so they
describe the release the site was built from (0.8.2).

!!! note "Docstring language"

    Parameter names, order, and defaults come straight from the code. Many docstrings are written in Chinese, and some
    descriptions are outdated. The [User Guides](../guides/index.md) explain the behavior in English with runnable examples.

## Subpackages

| Import path | Reference page | Main contents |
|---|---|---|
| `Modeling_Tool.Core` | [Core](core.md) | Binning, model registry (`save_model`, `load_model`), sample-weight helpers, ODPS client, parallel apply engine, ProcCompare, encryption, JSON conversion, utilities |
| `Modeling_Tool.WOE` | [WOE](woe.md) | `WOE_Master`, `MonotoneWOEBinner`, WOE transforms, WOE plots, engine adapter |
| `Modeling_Tool.Feature` | [Feature](feature.md) | PSI, IV/KS insights, correlation filter, `feature_screen`, distribution analysis |
| `Modeling_Tool.Model` | [Model](model.md) | `LRMaster`, `GradientBoostingModel` (LightGBM, XGBoost, CatBoost), backward elimination |
| `Modeling_Tool.Eval` | [Eval](eval.md) | Gains tables, `PerformanceEvaluator`, score comparison, ROC/KS/PR/lift functions and plots |
| `Modeling_Tool.Sample` | [Sample](sample.md) | Splitting, sampling, balancing, reject inference, distribution adaptation |
| `Modeling_Tool.Explainability` | [Explainability](explainability.md) | `ModelExplainer` (SHAP, Owen value, PDP, ICE, ALE, LIME), coalition structure |
| `Modeling_Tool.UAT` | [UAT](uat.md) | `UATConsistencyChecker`, `UATConfig`, comparison helpers |
| `Modeling_Tool.Pipeline` | [Pipeline](pipeline.md) | The seven one-click pipelines with their config and result classes, config schema helpers |
| `ExcelMaster` | [ExcelMaster](excelmaster.md) | Cursor-based Excel writer, preset formats, report templates |
| `Report` | [Report](report.md) | `Report_Tool`: performance, WOE plot, and variable-importance sheets |

## Top-Level Names

`Modeling_Tool.__all__` lists 123 names, and you can import any of them directly:

```python
from Modeling_Tool import WOE_Master, GradientBoostingModel, PerformanceEvaluator
```

| Area | Names |
|---|---|
| Samples | `SampleSplitter`, `StratifiedSampler`, `SampleBalancer`, `select_sample_seed`, `RejectInferenceFactory`, `RejectInferrer`, `ParcelingInferrer`, `HardCutoffInferrer`, `FuzzyAugmentInferrer`, `SimpleAugmentInferrer`, `DistributionAdaptation` |
| WOE | `WOE_Master`, `MonotoneWOEBinner`, `WOEEngineAdapter`, `as_woe_engine`, `is_monotonic`, `woe_transform`, `woe_transformation`, `plot_woe`, `save_mapping_table`, `load_mapping_table`, `get_overall_woe_table`, `SMF_MISSING_BIN` |
| Feature screening | `PSICalculator`, `calculate_psi_within_dataset`, `VarExtractionInsights`, `CorrelationFilter`, `feature_screen`, `feature_screen_from_dataframe`, `FeatureScreenConfig`, `FeatureScreenResult`, `fit_screening_woe_engine`, `screen_config_from_mapping`, `weighted_feature_screen`, `WeightedScreenResult`, `DistributionShiftAnalyzer`, `DistributionPlotter`, `proc_means_odps` |
| Models | `LRMaster`, `GradientBoostingModel`, `LightGBMModel`, `XGBoostModel`, `CatBoostModel`, `lgbm_quick_train`, `xgbm_quick_train`, `catboost_quick_train`, `FeatureSelectionAnalyzer`, `BackwardVariableEliminator` |
| Evaluation | `PerformanceEvaluator`, `GainsTableCalculator`, `Model_Evaluation_Tool`, `EvaluationPipeline`, `cross_risk`, `get_gains_table_by_cust_metrics`, `calc_lift_apt`, `evaluate_performance`, `comparison_performance` |
| Explainability | `ModelExplainer`, `build_coalition_structure`, `CREDIT_PRIOR_GROUPS` |
| Binning and utilities | `Binning`, `super_binning`, `SlopeCalculator`, `DataFrameProcessor`, `FilePathManager`, `DateTimeUtils`, `WOEIVCalculator`, `TextEncryptor`, `get_feature_names`, `scoring` |
| Model files | `save_model`, `load_model`, `load_model_metadata` |
| ODPS | `ODPSRunner`, `ParallelODPSManager`, `ParallelODPSConfig`, `pull_attributes_in_batch` |
| Parallel apply | `parallel_apply`, `ParallelApplyEngine`, `ParallelApplyConfig`, `ParallelApplyResult` |
| Dataset comparison | `proc_compare`, `ProcCompareEngine`, `ProcCompareConfig`, `ProcCompareResult` |
| Pipelines | `CreditModelPipeline`, `FeatureValidationPipeline`, `RejectInferencePipeline`, `ScoreComparisonPipeline`, `ScoreConsistencyUATPipeline`, `SampleAnalysisPipeline`, `MockSamplePipeline`, each with a `<Name>Config` and a `<Name>Result` class |
| Pipeline schema | `FieldMeta`, `PipelineRegistryEntry`, `PIPELINE_REGISTRY`, `get_pipeline_registry`, `get_pipeline_registry_schema`, `get_config_field_meta`, `extract_config_schema`, `extract_pipeline_schema`, `extract_schema`, `config_to_dict`, `config_from_dict`, `config_to_yaml`, `config_from_yaml`, `validate_pipeline_config`, `generate_pipeline_code` |
| Package metadata | `__version__`, `__author__` |

!!! note "Lazy names"

    `import Modeling_Tool` stays light. The model, explainability, pipeline, and ODPS names load their subpackage the first
    time you access them, and the optional libraries behind them (LightGBM, XGBoost, CatBoost, `shap`, `lime`, `pyodps`)
    load only when you use them. Because `ODPSRunner` is in `__all__`, `from Modeling_Tool import *` and
    `from Modeling_Tool.Core import *` import `pyodps` and fail without the `odps` extra. Import the names you need
    explicitly.

### Names Outside `__all__`

These names can be imported from `Modeling_Tool` but are not part of `__all__`, so a star import skips them:

- `get_gains_table`, `get_perf_summary`, `calc_roc`, `calc_pr`, `calc_equid_dist`, `calc_equid_pct`, `calc_fixed_pct`
- `backward_lgbm`, `backward_xgbm`
- `FeatureScreeningArtifact`, `screen_result_to_summary`, `run_modeling_from_validation`
- `ParallelODPSPuller`, an alias of `ParallelODPSManager`

### Names That Need a Subpackage Import

Many public names are not exported at the top level. Import them from their subpackage:

```python
from Modeling_Tool.Feature import proc_means_by_grp
from Modeling_Tool.Core import calc_woe, calc_iv, parse_sql_file
from Modeling_Tool.Eval import plot_roc_curve, summarize_pct
from Modeling_Tool.Eval.evaluate_model import summarize_roc
from Modeling_Tool.UAT import UATConfig, UATConsistencyChecker
from Modeling_Tool.WOE import mapping_woe
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import single_model_perf
```

The pages above list each subpackage's names in the module that defines them.

## Docstring Style

SMF uses NumPy-style docstrings: a one-line summary followed by `Parameters`, `Returns`, and `Examples` sections.
mkdocstrings renders the sections as tables (`docstring_style: numpy` and `docstring_section_style: table` in
`mkdocs.yml`), shows members in source order, and hides names that start with `_`.

## Sample Weight API

The training and evaluation functions take sample weights as native arguments. The keyword depends on the function:
`weight_col` (a column name) for `LRMaster.fit`, `PerformanceEvaluator`, `GainsTableCalculator`, and the DataFrame-based
APIs; `sample_weight` (an array) for `GradientBoostingModel.fit` and the low-level `calc_roc` and `evaluate_performance`.
The [FAQ](../faq.md#which-keyword-passes-sample-weights-weight_col-or-sample_weight) lists the keyword of every function.
For metric semantics, see [Model Training: Sample Weights](../guides/model.md#sample-weights) and
[Model Evaluation: Sample-Weighted Evaluation](../guides/eval.md#sample-weighted-evaluation).
