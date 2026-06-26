# API 参考

完整公开 API 的签名、参数、返回值说明，由源码 docstring 通过 [mkdocstrings-python](https://mkdocstrings.github.io/) 自动生成。

## 子包索引

| 子包 | 路径 | 主要内容 |
|------|------|---------|
| **Modeling_Tool.Core** | [`api/core.md`](core.md) | 分箱、ODPS、工具、加密、JSON、斜率 |
| **Modeling_Tool.WOE** | [`api/woe.md`](woe.md) | WOE 主控、转换器、绘图器、单调分箱器 |
| **Modeling_Tool.Feature** | [`api/feature.md`](feature.md) | PSI、IV、相关性、分布 |
| **Modeling_Tool.Model** | [`api/model.md`](model.md) | LR、LightGBM、XGBoost、后向消元 |
| **Modeling_Tool.Eval** | [`api/eval.md`](eval.md) | Gains 表、ROC/KS、链式评估 |
| **Modeling_Tool.Sample** | [`api/sample.md`](sample.md) | 切分、分层、均衡、拒绝推断 |
| **Modeling_Tool.Explainability** | [`api/explainability.md`](explainability.md) | SHAP 模型解释（`ModelExplainer`） |
| **Modeling_Tool.UAT** | [`api/uat.md`](uat.md) | 线上线下一致性校验 |
| **ExcelMaster** | [`api/excelmaster.md`](excelmaster.md) | 通用 Excel 报告引擎 |
| **Report** | [`api/report.md`](report.md) | 风控报告模板函数 |

## 顶层统一 API

`Modeling_Tool/__init__.py` 精选导出的常用 API（建议优先使用）：

```python
from Modeling_Tool import (
    # Core
    Binning, super_binning, ODPSRunner,
    SlopeCalculator, DataFrameProcessor, FilePathManager, DateTimeUtils,
    WOEIVCalculator, TextEncryptor,
    get_feature_names, pull_attributes_in_batch,
    save_model, load_model, scoring,

    # Model（懒加载，首次访问时才导入 lightgbm/xgboost）
    GradientBoostingModel, LightGBMModel, XGBoostModel,
    lgbm_quick_train, xgbm_quick_train,
    LRMaster, FeatureSelectionAnalyzer, BackwardVariableEliminator,

    # Explainability（懒加载，需 pip install supermodelingfactory[explain]）
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

## 阅读建议

!!! tip "API 文档的章节结构"

    每个 API 页面按 **类 → 方法 → 函数** 层级组织：

    - 类标题列出继承层次、构造函数签名
    - 方法标题给出签名 + 参数表 + 返回值
    - 私有成员（以 `_` 开头）默认隐藏

## 关于 docstring 风格

本项目使用 **NumPy 风格 docstring**（`Parameters / Returns / Examples`），mkdocstrings 会渲染为表格形式：

```python
def calc_woe(data, bad_pct, good_pct):
    """
    计算 WOE 值。

    Parameters
    ----------
    data : pandas.DataFrame
        含坏样本率与好样本率的表。
    bad_pct : str
        坏样本占比列名。
    good_pct : str
        好样本占比列名。

    Returns
    -------
    pandas.Series
        WOE 值。

    Examples
    --------
    >>> woe = calc_woe(stats, "BAD_PCT", "GOOD_PCT")
    """
```

如需给函数补充 docstring，请遵循同一风格保持一致。
