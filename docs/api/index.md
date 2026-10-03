# API Reference

Signatures, parameters, and return values for the full public API, generated automatically from source docstrings through [mkdocstrings-python](https://mkdocstrings.github.io/) .

## Subpackage Index

| Subpackage | Path | Main contents |
|------|------|---------|
| **Modeling_Tool.Core** | [`api/core.md`](core.md) | Binning, ODPS, utilities, encryption, JSON, slope |
| **Modeling_Tool.WOE** | [`api/woe.md`](woe.md) | WOE master, transformers, plotters, monotone binner |
| **Modeling_Tool.Feature** | [`api/feature.md`](feature.md) | PSI, IV, correlation, distribution |
| **Modeling_Tool.Model** | [`api/model.md`](model.md) | LR, LightGBM, XGBoost, CatBoost, backward elimination (with sample weights) |
| **Modeling_Tool.Eval** | [`api/eval.md`](eval.md) | Gains table, ROC/KS, chained evaluation (with `weight_col` weighted metrics) |
| **Modeling_Tool.Sample** | [`api/sample.md`](sample.md) | Splitting, stratification, balancing, reject inference |
| **Modeling_Tool.Explainability** | [`api/explainability.md`](explainability.md) | SHAP model explanation (`ModelExplainer`) |
| **Modeling_Tool.UAT** | [`api/uat.md`](uat.md) | Online/offline consistency checks |
| **ExcelMaster** | [`api/excelmaster.md`](excelmaster.md) | General-purpose Excel reporting engine |
| **Report** | [`api/report.md`](report.md) | Risk-control report template functions |

## Top-Level Unified API

Commonly used APIs curated and exported by `Modeling_Tool/__init__.py` (prefer these):

```python
from Modeling_Tool import (
    # Core
    Binning, super_binning, ODPSRunner,
    SlopeCalculator, DataFrameProcessor, FilePathManager, DateTimeUtils,
    WOEIVCalculator, TextEncryptor,
    get_feature_names, pull_attributes_in_batch,
    save_model, load_model, scoring,

    # Model (lazy-loaded; lightgbm/xgboost are imported only on first access)
    GradientBoostingModel, LightGBMModel, XGBoostModel,
    lgbm_quick_train, xgbm_quick_train,
    LRMaster, FeatureSelectionAnalyzer, BackwardVariableEliminator,

    # Explainability (lazy-loaded; requires pip install supermodelingfactory[explain])
    ModelExplainer,

    # Eval
    cross_risk, GainsTableCalculator, PerformanceEvaluator,
    Model_Evaluation_Tool, EvaluationPipeline,
    get_gains_table_by_cust_metrics, calc_lift_apt,
    evaluate_performance, comparison_performance,

    # Sample
    DistributionAdaptation,
    RejectInferrer, RejectInferenceFactory,
    ParcelingInferrer, HardCutoffInferrer,
    FuzzyAugmentInferrer, SimpleAugmentInferrer,
    SampleSplitter, StratifiedSampler, SampleBalancer,
    select_sample_seed,

    # WOE
    WOE_Master, is_monotonic,
    woe_transform, woe_transformation, plot_woe,
    save_mapping_table, load_mapping_table, get_overall_woe_table,

    # Feature
    DistributionShiftAnalyzer, DistributionPlotter,
    VarExtractionInsights, CorrelationFilter,
    PSICalculator, calculate_psi_within_dataset,
)
```

## Reading Tips

!!! tip "How the API pages are structured"

    Each API page is organized by **class → method → function**:

    - Class headings list the inheritance hierarchy and constructor signature
    - Method headings give the signature, a parameter table, and the return value
    - Private members (starting with `_`) are hidden by default

## About the Docstring Style

This project uses **NumPy-style docstrings** (`Parameters / Returns / Examples`), which mkdocstrings renders as tables:

```python
def calc_woe(data, bad_pct, good_pct):
    """
    Compute WOE values.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing the bad rate and good rate.
    bad_pct : str
        Column name of the bad-sample share.
    good_pct : str
        Column name of the good-sample share.

    Returns
    -------
    pandas.Series
        WOE values.

    Examples
    --------
    >>> woe = calc_woe(stats, "BAD_PCT", "GOOD_PCT")
    """
```

When adding a docstring to a function, follow the same style for consistency.

## Sample Weight API

The weighting parameters for training and evaluation have been merged into the main repository as **native public APIs** (`LRMaster.fit(weight_col=...)`, `GradientBoostingModel.fit(sample_weight=...)`, `PerformanceEvaluator(weight_col=...)`, and so on). For practical usage and metric semantics, see [Model Training — Sample Weights](../guides/model.md#sample-weights) and [Model Evaluation — Sample-Weighted Evaluation](../guides/eval.md#sample-weighted-evaluation).
