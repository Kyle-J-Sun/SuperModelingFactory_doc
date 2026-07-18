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

psi = PSICalculator(
    buckets=10,
    equal_freq=True,
    min_bin_prop=0.05,
    feature_block_size=64,
)
psi_table = psi.calculate(train_df, oot_df, features)
stable_features = psi_table.loc[psi_table["psi"] < 0.1, "var"].tolist()
```

如果已有 WOE 分箱引擎，传入 `binning_engine`：

```python
psi = PSICalculator(buckets=10, binning_engine=binner)
psi_table = psi.calculate(train_df, oot_df, features)
```

### v0.5.1 PSI one-sided bucket policy

When a bucket appears on only one side of a PSI comparison, SMF now defaults to
Laplace smoothing instead of flooring the missing side to `1e-6`:

```python
psi = PSICalculator(psi_missing_bucket_policy="smooth_laplace")  # default
```

Use `psi_missing_bucket_policy="floor_1e6"` for legacy reports, or
`psi_missing_bucket_policy="exclude"` to remove one-sided buckets from the PSI sum.

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

From v0.5.1, expected per-variable failures are recorded in
`insights.failed_variables` and summarized with one warning instead of silently
disappearing from the report.

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

## 4. 统一特征筛选（v0.3.9+）

`feature_screen` 是 CM / FVP 共享的筛选内核，在 INS/OOS/OOT 分片上做 PSI → IV → 相关性剔除。`weighted_feature_screen` 与 `CreditModelPipeline._feature_selection` 均委托此 API。

```python
from Modeling_Tool import FeatureScreenConfig, feature_screen, fit_screening_woe_engine

splits = {"ins": ins_df, "oos": oos_df, "oot": oot_df}
binner = fit_screening_woe_engine(splits["ins"], features, "badflag", woe_engine="monotone")
config = FeatureScreenConfig(
    psi_compare_splits=["oos"],
    psi_use_woe_bins=True,
    iv_use_woe_bins=True,
    corr_use_woe_bins=True,
    corr_block_size=256,
)
result = feature_screen(splits, features, "badflag", config=config, prefit_woe_engine=binner)
selected = result.selected_features
```

`feature_selection` 字典可通过 `screen_config_from_mapping()` 转为 `FeatureScreenConfig`。`weight_col` 非空时默认走加权等频路径；设置 `*_use_woe_bins=True` 且提供（或自动 fit）WOE 引擎后，加权 PSI/IV 复用相同分箱边界。

`feature_block_size` 控制 PSI/WOE 适配器一次转换多少列，`corr_block_size` 控制加权相关性矩阵块大小。两者只用于限制超宽表峰值内存，不改变指标口径。

分组分布统计也支持列块：

```python
from Modeling_Tool import proc_means_by_grp

summary = proc_means_by_grp(
    data,
    features,
    groupby=["apply_month", "channel"],
    feature_block_size=128,
)
```

如果源数据位于 MaxCompute，不需要先把全量宽表拉到 pandas。`proc_means_odps()`
会在 ODPS 端按特征批次完成聚合，只下载最终统计结果：

```python
from Modeling_Tool import proc_means_odps

summary = proc_means_odps(
    input_table_name="mex_anls.feature_wide_table",
    select_cols=features,
    group=["apply_month", "channel"],
    batch_size=50,
    where_clause="dt >= '2026-01-01'",
)
```

返回字段与数值型 `proc_means_by_grp()` 对齐，包括 `N_ALL/N/MEAN/STD/MIN/分位数/MAX/MISSING_RATE`。
`batch_size` 表示一条聚合 SQL 中的特征数量，不会按行下载源数据。完整参数、分位数模式和 ODPS 写回规则见
[ODPS 数据抽取：`proc_means_odps`](odps.md#5-proc_means_odps-odps-端描述性统计)。

## 5. 加权特征筛选（v0.3.8+）

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

`CreditModelPipeline` 的 `_feature_selection` 已委托 `feature_screen`；配置 `weight_col` 非空时自动走加权路径。

## 6. 分布偏移分析

```python
from Modeling_Tool import DistributionShiftAnalyzer

analyzer = DistributionShiftAnalyzer(dataset, grp_name="apply_month", benchmark_value="2025-01")
shift_table = analyzer.analyze(varlist=features, outlier_value=0.99)
print(shift_table)
```

## 完整筛选流水线

=== "WOE_Master（默认）"

    ```python
    from Modeling_Tool import WOE_Master, PSICalculator, VarExtractionInsights, CorrelationFilter, SMF_MISSING_BIN

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

## 后置选择门（0.6.7+，G02–G06）

经典 缺失率 → PSI → IV → 相关性 之后，`feature_screen` 新增一组默认关闭的后置门，
按 **VIF → 组稳 → 多标签 → 截断** 顺序执行；证据帧与淘汰明细统一进
`result.stage_tables` / `result.dropped_detail`（列：var/stage/metric/value/threshold/reason）：

```python
FeatureValidationPipelineConfig(
    selection_enabled=True,
    selection_group_dims=["apply_month"],       # G03 证据分组维度
    selection_params={
        "iv_upper_threshold": 2.0,               # G02 IV 上限（疑似泄漏淘汰）
        "monthly_iv_min": 0.02,                  # G03 分组 IV 下限
        "monthly_iv_cv_max": 1.0,                #     分组 IV 变异系数上限
        "direction_consistency_min": 0.9,        #     方向一致组占比下限
        "insufficient_group_policy": "keep_warn",#     合格组<2：keep_warn/drop/raise
        "target_rules": "all",                   # G04 多标签联合门：all/any/min_pass_count
        "per_target_iv_range": {"y": (0.02, None)},
        "direction_reference_target": "y",
        "max_selected_features": 30,             # G05 硬截断（IV 排序，名字破平）
        "vif_enabled": True, "vif_threshold": 10.0,  # G06 需 pip install "SuperModelingFactory[stats]"
        "vif_use_woe_bins": True,                      # 0.7.0+：按 WOE 矩阵计算 VIF
    },
)
```

要点：

- G03/G04 的证据（分组/分标签 IV 与方向）由 FVP 以 **lazy 闭包**构建，只对 post-corr 幸存集计价；
  CMP 路径没有证据来源——配置了组稳/多标签阈值但无证据会在 `feature_screen` 入口报错。
- 方向的统一定义是 point-biserial 符号（`Screen_Gates.point_biserial_direction`）。
- G05 截断不足 `min_selected_features` 时只告警不回填，保住每个门的因果可审计性。
- 筛选自拟合的 monotone 引擎会挂到 `result.woe_engine` 并随 `FeatureScreeningArtifact`
  进入 CM 复用契约（G00）；`WOE_Master` 只附 woe_table + 警告（它持有训练帧，不宜序列化）。
- G06 默认 `vif_use_woe_bins=False`，只在 raw 数值列上计算 VIF：非数值幸存变量会被保留、发出 warning，并在 `selection_summary` / `stage_tables["vif"]` 中留下排除审计；数值列不足时该门以 `skipped_insufficient_numeric` 跳过。bool 与 pandas nullable 数值列只在 VIF 矩阵内转换为浮点。
- 将 `vif_use_woe_bins=True` 后，VIF 使用 INS 的 WOE 编码矩阵，分类变量也可参与共线性判断。声明了 `categorical_features` 时必须使用 `woe_engine="monotone"`；`equal_freq` 会在拟合前明确拒绝该组合。预拟合 WOE engine 必须覆盖全部幸存变量，且自定义 WOE 后缀会被自动识别。
