# Modeling_Tool.Eval

模型评估层 —— Gains 表、ROC/KS、链式评估流水线。传入 `weight_col` 或 `sample_weight` 时走加权路径；未传则与历史未加权行为一致。

用户指南：[模型评估 — 样本权重评估](../guides/eval.md#样本权重评估)。

## 加权评估实现 — `weighted_eval_utils`

内部加权 ROC / Gains / 性能汇总实现；公开 API 在检测到权重参数时自动委托此模块。

::: Modeling_Tool.Eval.weighted_eval_utils

## 模型评估主控 — `Model_Eval_Tool`

::: Modeling_Tool.Eval.Model_Eval_Tool

## 链式评估流水线 — `Evaluation_Tool`

::: Modeling_Tool.Eval.Evaluation_Tool

## 单/多模型绘图 — `evaluate_model`

::: Modeling_Tool.Eval.evaluate_model
