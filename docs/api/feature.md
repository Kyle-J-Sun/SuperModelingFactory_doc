# Modeling_Tool.Feature

Feature analysis layer: PSI, IV/KS insights, correlation filtering, unified feature screening, and distribution analysis.

!!! note "`PSICalculator`, `VarExtractionInsights`, and `CorrelationFilter` are wrappers"

    The classes you import from `Modeling_Tool` or `Modeling_Tool.Feature` are defined in
    [`WOE_Engine_Feature_Patch`](#woe-engine-aware-classes-woe_engine_feature_patch). They wrap the implementations in
    `PSI_Tool` and `Feature_Insights` and add the arguments that reuse a fitted WOE engine: `binning_engine` for
    `PSICalculator`, and `woe_engine`, `woe_binner`, and `woe_engine_params` for `VarExtractionInsights` and
    `CorrelationFilter`. Every other argument and every method is the one documented for the implementation below.

## PSI Population Stability: `PSI_Tool`

::: Modeling_Tool.Feature.PSI_Tool

## Variable Insights and Correlation: `Feature_Insights`

::: Modeling_Tool.Feature.Feature_Insights

## WOE-Engine-Aware Classes: `WOE_Engine_Feature_Patch`

The signatures of the three exported classes. Their methods and arguments behave as described in `PSI_Tool` and
`Feature_Insights` above.

::: Modeling_Tool.Feature.WOE_Engine_Feature_Patch
    options:
      show_if_no_docstring: true
      members:
        - PSICalculator
        - VarExtractionInsights
        - CorrelationFilter

## Unified Feature Screening: `Feature_Screen`

::: Modeling_Tool.Feature.Feature_Screen

## Weighted Feature Screening: `Weighted_Screen`

::: Modeling_Tool.Feature.Weighted_Screen

## Distribution Analysis: `Distribution_Tool`

::: Modeling_Tool.Feature.Distribution_Tool

## ODPS-Side Distribution Analysis: `ODPS_Distribution_Tool`

::: Modeling_Tool.Feature.ODPS_Distribution_Tool
