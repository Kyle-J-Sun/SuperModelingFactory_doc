# Modeling_Tool.Eval

Model evaluation layer: Gains tables, KS/AUC summaries, challenger-versus-champion comparison, slice-by-slice evaluation,
and ROC/KDE/percentile/gain figures. Passing `weight_col` or `sample_weight` takes the weighted path; omitting them keeps the
historical unweighted behavior.

User guide: [Model Evaluation](../guides/eval.md), including [Sample-Weighted Evaluation](../guides/eval.md#sample-weighted-evaluation).

## Gains Tables and Performance Summaries: `Model_Eval_Tool`

::: Modeling_Tool.Eval.Model_Eval_Tool

## Score Comparison and Evaluation Pipeline: `Evaluation_Tool`

::: Modeling_Tool.Eval.Evaluation_Tool

## Curves, Summaries, and Figures: `evaluate_model`

::: Modeling_Tool.Eval.evaluate_model

## Weighted Implementations: `weighted_eval_utils`

Internal helpers behind the weighted path. The public functions above delegate to this module when they receive a weight
argument. They are not exported and their signatures can change without notice.

::: Modeling_Tool.Eval.weighted_eval_utils
