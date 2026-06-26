# 特征筛选

SuperModelingFactory 在 [`Feature`](../api/feature.md) 子包提供**PSI / IV / 相关性 / 分布**四类筛选工具。典型的筛选顺序：

```mermaid
flowchart LR
    A[原始特征集] --> B[PSI 稳定性]
    B --> C[IV 信息量]
    C --> D[相关性去冗余]
    D --> E[最终特征集]
```

## 1. PSI 群体稳定性指数 —— `PSICalculator`

衡量**实际分布**与**期望分布**的偏移，是风控模型监控的黄金指标。

```python
from Modeling_Tool import PSICalculator

psi = PSICalculator(
    buckets=10,
    equal_freq=True,
    min_bin_prop=0.05,
)

# 两组数据之间
psi_table = psi.calculate(
    expected_df=train_df,
    current_data=latest_df,
    varlist=features,
)
# calculate() 结果列为 var / psi
print(psi_table.sort_values("psi", ascending=False).head(10))

# 数据集内分组 PSI（各分组 vs 基准组）
from Modeling_Tool import calculate_psi_within_dataset
within_psi = calculate_psi_within_dataset(dataset, grp_name="apply_month", varlist=features)
```

### PSI 阈值

| PSI 区间 | 含义 |
|---------|------|
| `< 0.1` | 稳定，无需关注 |
| `0.1 – 0.25` | 轻微漂移，建议复盘 |
| `>= 0.25` | 显著漂移，需立即排查 |

### 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `buckets` | `10` | 分箱数 |
| `equal_freq` | `True` | 等频 vs 等距 |
| `min_bin_prop` | `0.05` | 每箱最小占比 |
| `content` | `1e-6` | 防除零小量 |

## 2. IV 信息量 —— `VarExtractionInsights`

IV（Information Value）衡量特征的**预测能力**，是变量筛选的核心指标。

```python
from Modeling_Tool import VarExtractionInsights

insights = VarExtractionInsights(
    data=train_df,
    dep="bad_flag",
    plot_path="./iv_plots/",   # 绘图输出目录
    nbins=10,
    equal_freq=True,
    tree_binning=True,
    seed=3407,
)

report = insights.get_var_analysis_report(train_df, features)
# 结果列（均为小写）：var / iv / ks_in_gains / lift_in_gains / n_bins ...
print(report[["var", "iv", "ks_in_gains", "lift_in_gains"]])
```

### IV 阈值经验

| IV 区间 | 解释力 |
|---------|--------|
| `< 0.02` | 无预测力，剔除 |
| `0.02 – 0.1` | 弱 |
| `0.1 – 0.3` | 中 |
| `0.3 – 0.5` | 强 |
| `>= 0.5` | 异常强，警惕过拟合 / 信息泄露 |

### 自动绘图

`VarExtractionInsights` 会在 `plot_path` 目录下输出每个特征的 IV 绘图（柱状图 + WOE 折线）。

## 3. 相关性去冗余 —— `CorrelationFilter`

剔除两两相关性过高（如 `> 0.7`）的特征，**保留 IV 较高**的。

```python
from Modeling_Tool import CorrelationFilter

corr_filter = CorrelationFilter(
    data=train_woe,
    dep="bad_flag",
    corr_cutpoint=0.7,
)
keep_vars = corr_filter.remove_highly_correlated(features)
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `corr_cutpoint` | `0.8` | 相关系数阈值（绝对值） |
| `dep` | - | 目标列，保留 IV 较高者 |

## 4. 分布偏移分析 —— `DistributionShiftAnalyzer`

```python
from Modeling_Tool import DistributionShiftAnalyzer

# 在同一数据框内按分组列比较：各分组相对基准组的分布偏移
analyzer = DistributionShiftAnalyzer(dataset, grp_name="apply_month", benchmark_value="2025-01")
shift_table = analyzer.analyze(varlist=features, outlier_value=0.99)
print(shift_table)   # 行=变量，列=各分组，值=超过基准组阈值的观测比例

# 单变量
single = analyzer.analyze_single_var("age", outlier_value=0.99)
```

## 5. 分布可视化 —— `DistributionPlotter`

```python
from Modeling_Tool import DistributionPlotter

plotter = DistributionPlotter(data=train_df, score="age")
plotter.plot_kdeplot(title="Age KDE")                 # 核密度图
plotter.plot_displot(title="Age 直方图", nbins=20)     # 带 KDE 的直方图
# 统一入口：plotter.plot(method="kdeplot", title="Age Distribution")
```

## 完整筛选流水线

```python
from Modeling_Tool import PSICalculator, VarExtractionInsights, CorrelationFilter

# 1) PSI 筛选（结果列 var / psi）
psi = PSICalculator(buckets=10).calculate(train_df, test_df, features)
features = psi.loc[psi["psi"] < 0.1, "var"].tolist()

# 2) IV 筛选（结果列 var / iv ...）
insights = VarExtractionInsights(train_df, "bad_flag", "./iv_plots/")
iv_report = insights.get_var_analysis_report(train_df, features)
features = iv_report.loc[iv_report["iv"].between(0.02, 0.5), "var"].tolist()

# 3) 相关性去冗余
features = CorrelationFilter(train_woe, "bad_flag", corr_cutpoint=0.7) \
              .remove_highly_correlated(features)

print(f"最终保留 {len(features)} 个特征: {features}")
```

## 常见问题

??? question "PSI 单变量都 < 0.1 但 KS/AUC 显著下降"

    可能是**多变量联合分布**漂移。需要用 `calculate_multivar_psi_two_sets()` 做联合 PSI。

??? question "高相关特征剔除后模型效果下降"

    调高 `corr_cutpoint`（如 `0.85`），或考虑主成分分析 / 业务合并。
