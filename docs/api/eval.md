# Modeling_Tool.Eval

Model evaluation layer — Gains table, ROC/KS, chained evaluation pipeline. Passing `weight_col` or `sample_weight` takes the weighted path; omitting them keeps the historical unweighted behavior.

User guide: [Model Evaluation — Sample-Weighted Evaluation](../guides/eval.md#sample-weighted-evaluation).

## Weighted Evaluation Implementation — `weighted_eval_utils`

Internal weighted ROC / Gains / performance-summary implementation; the public API delegates to this module automatically when it detects a weight parameter.

::: Modeling_Tool.Eval.weighted_eval_utils

## Model Evaluation Master — `Model_Eval_Tool`

::: Modeling_Tool.Eval.Model_Eval_Tool

## Chained Evaluation Pipeline — `Evaluation_Tool`

::: Modeling_Tool.Eval.Evaluation_Tool

## Single/Multi-Model Plotting — `evaluate_model`

::: Modeling_Tool.Eval.evaluate_model
