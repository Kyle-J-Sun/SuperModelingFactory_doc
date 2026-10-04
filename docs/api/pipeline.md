# Modeling_Tool.Pipeline

One-click pipelines. Each pipeline is a class that takes a config dataclass, runs with `Pipeline(config).run(data)`
(`ScoreConsistencyUATPipeline` can also read SQL files, and `MockSamplePipeline` needs no data), and returns a result
dataclass. This page is the generated reference. The [Top-Level Pipelines](../pipeline_one_click.md) guide explains what each
pipeline does and how to configure it.

All names below are also importable from `Modeling_Tool`, for example
`from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig`.

## Credit Model Pipeline: `credit_model`

::: Modeling_Tool.Pipeline.credit_model

## Feature Validation Pipeline: `feature_validation`

::: Modeling_Tool.Pipeline.feature_validation

## Reject Inference Pipeline: `reject_inference`

::: Modeling_Tool.Pipeline.reject_inference

## Score Comparison Pipeline: `score_comparison`

::: Modeling_Tool.Pipeline.score_comparison

## Score Consistency UAT Pipeline: `score_consistency_uat`

::: Modeling_Tool.Pipeline.score_consistency_uat

## Sample Analysis Pipeline: `sample_analysis`

::: Modeling_Tool.Pipeline.sample_analysis

## Mock Sample Pipeline: `mock_sample`

::: Modeling_Tool.Pipeline.mock_sample

## Modeling from Validation Results: `orchestrator`

::: Modeling_Tool.Pipeline.orchestrator

## Screening Artifact: `screening_artifact`

::: Modeling_Tool.Pipeline.screening_artifact

## Config Schema and Registry: `field_meta`

Helpers for building a configuration GUI or loading configs from YAML. See [Pipeline GUI Schema](../guides/pipeline_gui_schema.md).

::: Modeling_Tool.Pipeline.field_meta
