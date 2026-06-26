# 用户指南

按建模场景分册的实战指南，每一篇都包含 **背景 → 关键 API → 完整代码 → 常见问题**。

## 分册列表

<div class="grid cards" markdown>

- :material-call-split: **[样本管理](sample.md)**

    `SampleSplitter` / `StratifiedSampler` / `SampleBalancer` / `select_sample_seed`

- :material-database: **[ODPS 数据抽取](odps.md)**

    `ODPSRunner` / `parse_sql_file` / `pull_attributes_in_batch`

- :material-chart-bell-curve: **[WOE 编码](woe.md)**

    `WOE_Master` / `WOE_Monotone_Binner` / `is_monotonic`

- :material-filter-variant: **[特征筛选](feature.md)**

    `PSICalculator` / `VarExtractionInsights` / `CorrelationFilter`

- :material-brain: **[模型训练](model.md)**

    `LRMaster` / `GradientBoostingModel` / `BackwardVariableEliminator`

- :material-chart-line: **[模型评估](eval.md)**

    `PerformanceEvaluator` / `GainsTableCalculator` / `EvaluationPipeline`

- :material-lightbulb-on: **[模型解释](explainability.md)**

    `ModelExplainer`（SHAP 特征重要性 / summary / dependence / 单样本归因）

- :material-account-cancel: **[拒绝推断与分布适配](reject_inference.md)**

    `RejectInferenceFactory` / `DistributionAdaptation`

- :material-shield-check: **[线上线下一致性校验](uat.md)**

    `UATConsistencyChecker` / `UATConfig`

- :material-file-excel: **[Excel 报告生成](excel_report.md)**

    `ExcelMaster` / `Template` / `Report`

</div>

## 阅读顺序建议

!!! tip "新手路径"

    1. 先读 [快速上手](../quickstart.md) 跑通一遍
    2. 再读 [端到端流水线](../pipeline.md) 理解整体流程
    3. 然后按需查阅各分册

!!! info "API 速查"

    任何时刻想查具体函数签名，跳转到 [API 参考](../api/index.md) — 由源码 docstring 自动生成。
