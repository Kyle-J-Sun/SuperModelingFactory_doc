# 特征筛选

SuperModelingFactory 在 [`Feature`](../api/feature.md) 子包提供 **PSI / IV / 相关性 / 分布** 四类筛选工具。推荐顺序是：先稳定性，再解释力，最后去冗余。

```mermaid
flowchart LR
    A[原始特征集] --> B[PSI 稳定性]
    B --> C[IV / KS 信息量]
    C --> D[相关性去冗余]
    D --> E[最终特征集]
```

!!! important "分箱一致性"

    如果模型最终使用 `MonotoneWOEBinner`，请把同一个 binner 传给 `PSICalculator`、`VarExtractionInsights` 和 `CorrelationFilter`。否则筛选阶段可能重新分箱，指标会和最终 WOE 编码不一致。

    详情见 [WOE 分箱引擎](woe_binning_engine.md)。

## 1. PSI 群体稳定性指数

默认用法保持不变：

```python
from Modeling_Tool import PSICalculator

psi = PSICalculator(buckets=10, equal_freq=True, min_bin_prop=0.05)
psi_table = psi.calculate(train_df, oot_df, features)
stable_features = psi_table.loc[psi_table["psi"] < 0.1, "var"].tolist()
```

如果已有 WOE 分箱引擎，传入 `binning_engine`：

```python
psi = PSICalculator(buckets=10, binning_engine=binner)
psi_table = psi.calculate(train_df, oot_df, features)
```

### PSI 阈值

| PSI 区间 | 含义 |
|---------|------|
| `< 0.1` | 稳定，无需关注 |
| `0.1 - 0.25` | 轻微漂移，建议复盘 |
| `>= 0.25` | 显著漂移，需立即排查 |

## 2. IV / KS 信息量

默认路径：

```python
from Modeling_Tool import VarExtractionInsights

insights = VarExtractionInsights(
    data=train_df,
    dep="bad_flag",
    plot_path="./iv_plots/",
    nbins=10,
    equal_freq=True,
    tree_binning=True,
)
report = insights.get_var_analysis_report(train_df, features)
print(report[["var", "iv", "ks_in_gains", "lift_in_gains"]])
```

复用 Monotone 分箱：

```python
insights = VarExtractionInsights(
    data=train_df,
    dep="bad_flag",
    plot_path="./iv_plots/",
    woe_engine="monotone",
    woe_binner=binner,
)
report = insights.get_var_analysis_report(train_df, features)
```

### IV 阈值经验

| IV 区间 | 解释力 |
|---------|--------|
| `< 0.02` | 无预测力，剔除 |
| `0.02 - 0.1` | 弱 |
| `0.1 - 0.3` | 中 |
| `0.3 - 0.5` | 强 |
| `>= 0.5` | 异常强，警惕过拟合 / 信息泄露 |

## 3. 相关性去冗余

剔除两两相关性过高的变量，并保留 IV 或 KS 更高的变量。

```python
from Modeling_Tool import CorrelationFilter

keep_vars = CorrelationFilter(
    data=train_df,
    dep="bad_flag",
    corr_cutpoint=0.7,
    woe_engine="monotone",
    woe_binner=binner,
).remove_highly_correlated(features)
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `corr_cutpoint` | `0.8` | 相关系数阈值 |
| `base_metric` | `iv` | 高相关变量组内的保留指标，可选 `iv` 或 `ks` |
| `woe_engine` | `master` | 分箱引擎名称 |
| `woe_binner` | `None` | 已拟合的 `WOE_Master` 或 `MonotoneWOEBinner` |

## 4. 加权特征筛选（v0.3.8+）

`weighted_feature_screen` 把 PSI → IV → 相关性去冗余串成一条 API，并支持 `weight_col` 加权分位切点、加权 IV/PSI 与加权 Pearson 相关性（IV 仲裁）。

```python
from Modeling_Tool import weighted_feature_screen

result = weighted_feature_screen(
    data=df,                      # 需含 split_col 取值 ins/oos/oot
    feature_cols=features,
    target_col="badflag",
    split_col="sample_ind",
    weight_col="_weight",         # None 时走 legacy 无权工具，与旧 Pipeline 回归一致
    psi_compare_splits=["oos", "oot"],  # 独立 API 默认；Pipeline 默认仅 ["oos"]
)
selected = result.selected_features
iv_table = result.iv_table       # iv_weighted / n_bins / missing_rate
psi_table = result.psi_table     # psi_ins_oos / psi_ins_oot / psi_max
```

典型场景：Fuzzy Augment 双行样本带 `_weight` 时，用加权 IV 避免未加权计数对冲失真；05E 实验可对 RI 增广样本 + `weight_col` 做加权 top-N 筛选。

`CreditModelPipeline` 的 `_feature_selection` 已委托此 API；配置 `weight_col` 非空时自动走加权路径。

## 5. 分布偏移分析

```python
from Modeling_Tool import DistributionShiftAnalyzer

analyzer = DistributionShiftAnalyzer(dataset, grp_name="apply_month", benchmark_value="2025-01")
shift_table = analyzer.analyze(varlist=features, outlier_value=0.99)
print(shift_table)
```

## 完整筛选流水线

=== "WOE_Master（默认）"

    ```python
    from Modeling_Tool import WOE_Master, PSICalculator, VarExtractionInsights, CorrelationFilter

    woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
    woe.fit(nbins=10, equal_freq=True)

    psi = PSICalculator(binning_engine=woe).calculate(train_df, oot_df, features)
    features = psi.loc[psi["psi"] < 0.1, "var"].tolist()

    insights = VarExtractionInsights(train_df, "bad_flag", "./iv_plots/", woe_binner=woe)
    iv_report = insights.get_var_analysis_report(train_df, features)
    features = iv_report.loc[iv_report["iv"].between(0.02, 0.5), "var"].tolist()

    features = CorrelationFilter(train_df, "bad_flag", corr_cutpoint=0.7, woe_binner=woe) \
        .remove_highly_correlated(features)
    ```

=== "MonotoneWOEBinner（评分卡推荐）"

    ```python
    from Modeling_Tool import PSICalculator, VarExtractionInsights, CorrelationFilter
    from Modeling_Tool.WOE.WOE_Monotone_Binner import MonotoneWOEBinner

    binner = MonotoneWOEBinner(feature_cols=features, target_col="bad_flag")
    binner.fit(train_df, chi2_binning=True, chi2_p=0.95)

    psi = PSICalculator(binning_engine=binner).calculate(train_df, oot_df, features)
    features = psi.loc[psi["psi"] < 0.1, "var"].tolist()

    insights = VarExtractionInsights(
        train_df, "bad_flag", "./iv_plots/",
        woe_engine="monotone", woe_binner=binner,
    )
    iv_report = insights.get_var_analysis_report(train_df, features)
    features = iv_report.loc[iv_report["iv"].between(0.02, 0.5), "var"].tolist()

    features = CorrelationFilter(
        train_df, "bad_flag", corr_cutpoint=0.7,
        woe_engine="monotone", woe_binner=binner,
    ).remove_highly_correlated(features)
    ```

## 常见问题

??? question "PSI 和 IV 结果与 WOE 图不一致"

    通常是因为筛选阶段没有传入最终建模使用的 `binning_engine` / `woe_binner`。请复用训练期已拟合的 `WOE_Master` 或 `MonotoneWOEBinner`。

??? question "Monotone 分箱后，相关性过滤传 raw 数据还是 WOE 数据？"

    推荐传 raw 数据，并同时传入 `woe_binner`。这样相关性过滤仍基于原始变量相关性，变量保留决策则基于同一套 WOE 分箱的 IV/KS。
