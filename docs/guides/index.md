# 用户指南

按建模场景分册的实战指南，每一篇都包含 **背景 → 关键 API → 完整代码 → 常见问题**。

## 分册列表

<div class="grid cards" markdown>

- :material-call-split: **[样本管理](sample.md)**

    `SampleSplitter` / `StratifiedSampler` / `SampleBalancer` / `select_sample_seed`

- :material-database: **[ODPS 数据抽取](odps.md)**

    `ODPSRunner` / `parse_sql_file` / `pull_attributes_in_batch`

- :material-table-search: **[数据一致性对比](proc_compare.md)**

    `ProcCompareEngine` / `proc_compare` / DataFrame 与 CSV 大表一致性校验

- :material-application-brackets: **[Pipeline GUI Schema](pipeline_gui_schema.md)**

    `extract_pipeline_schema` / `config_to_yaml` / `validate_pipeline_config` / GUI 表单元数据

- :material-chart-bell-curve: **[WOE 编码](woe.md)**

    `WOE_Master` / `MonotoneWOEBinner` / `is_monotonic`

- :material-vector-link: **[WOE 分箱引擎](woe_binning_engine.md)**

    `as_woe_engine` / `binning_engine` / `woe_binner` / Master-Monotone 统一协议

- :material-filter-variant: **[特征筛选](feature.md)**

    `PSICalculator` / `VarExtractionInsights` / `CorrelationFilter`

- :material-brain: **[模型训练](model.md)**

    `LRMaster` / `GradientBoostingModel` / `BackwardVariableEliminator`（含 `weight_col`）

- :material-tune: **[GBM 超参搜索](gbm_param_search.md)**

    `GradientBoostingModel.param_search` / grid search / Optuna / INS-OOS-OOT holdout（含加权 AUC）

- :material-archive-cog: **[模型注册与版本管理](model_registry.md)**

    `save_model` / `load_model` / `load_model_metadata` / 模型 metadata artifact

- :material-chart-line: **[模型评估](eval.md)**

    `PerformanceEvaluator` / `GainsTableCalculator` / `EvaluationPipeline`（含加权 Gains / KS / AUC）

- :material-lightbulb-on: **[模型解释](explainability.md)**

    `ModelExplainer`（SHAP / Owen Value / PDP / ICE / ALE / LIME）

- :material-account-cancel: **[拒绝推断与分布适配](reject_inference.md)**

    `RejectInferenceFactory` / `DistributionAdaptation`

- :material-shield-check: **[线上线下一致性校验](uat.md)**

    `UATConsistencyChecker` / `UATConfig`

- :material-file-excel: **[Excel 报告生成](excel_report.md)**

    `ExcelMaster` / `Template` / `Report`

</div>

!!! tip "版本变更笔记"

    历史版本的迭代内容已集中到顶级导航的 [ChangeLog](../changelog/index.md) tab 下。

## 阅读顺序建议

!!! tip "新手路径"

    1. 先读 [快速上手](../quickstart.md) 跑通一遍
    2. 再读 [端到端流水线](../pipeline.md) 理解整体流程
    3. 如果使用单调分箱，优先读 [WOE 分箱引擎](woe_binning_engine.md)
    4. 然后按需查阅各分册

!!! info "API 速查"

    任何时刻想查具体函数签名，跳转到 [API 参考](../api/index.md) — 由源码 docstring 自动生成。
