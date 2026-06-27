# WOE 编码

WOE（Weight of Evidence）把分类变量或分箱后的连续变量映射为线性可分的数值，是评分卡建模的标准做法。

SuperModelingFactory 在 [`WOE`](../api/woe.md) 子包提供 **主控类 + 单调分箱器 + 转换器 + 绘图器 + 分箱引擎适配器**。

## 1. 主控类 —— `WOE_Master`

```python
from Modeling_Tool import WOE_Master

woe = WOE_Master(
    train_data=train_df,
    varlist=features,
    dep="bad_flag",
    missing_ref_value=-999999,
)
woe.fit(nbins=10, equal_freq=True)

train_woe = woe.transform(train_df)
test_woe = woe.transform(test_df)
oot_woe = woe.transform(oot_df)
```

### 持久化映射表

```python
woe.save_mapping_table("./output/woe_mapping.csv")

from Modeling_Tool import load_mapping_table
varlist, woe_dict = load_mapping_table("./output/woe_mapping.csv")
```

## 2. 贪心单调分箱器 —— `MonotoneWOEBinner`

如果评分卡需要更强的单调约束，推荐使用 `MonotoneWOEBinner`。

```python
from Modeling_Tool.WOE.WOE_Monotone_Binner import MonotoneWOEBinner

binner = MonotoneWOEBinner(
    feature_cols=features,
    target_col="bad_flag",
    n_init_bins=20,
    min_bin_size=0.03,
    special_values=[-1, -100, -999999],
    cate_feats=["city_grade"],
)
binner.fit(train_df, chi2_binning=True, chi2_p=0.95)
binner.refine_cate(max_bins=5)

train_woe = binner.apply_woe(train_df)
bins = binner.get_final_bins()
edges = binner.get_bin_edges()
```

### 方法列表

| 方法 | 说明 |
|------|------|
| `fit(df, chi2_binning, chi2_p, n_jobs)` | 训练拟合 |
| `refine_cate(max_bins)` | 类别特征按坏率聚类合并 |
| `apply_woe(df)` | WOE 转换 |
| `get_final_bins()` | 导出分箱结果（含 WOE/IV） |
| `load_woe_bins(bins_dict)` | 加载已有分箱 |
| `get_bin_edges()` | 取分箱边界列表 |
| `export_woe_report(path)` | 导出 Excel 报告 |
| `plot_woe_graph(dir, group_name=)` | 输出 WOE 图 PNG |

## 3. 统一分箱引擎 —— `as_woe_engine`

`WOE_Master` 与 `MonotoneWOEBinner` 的内部产物格式不同。`as_woe_engine()` 会把它们转成统一接口，供 PSI、IV、相关性筛选复用。

```python
from Modeling_Tool import as_woe_engine

engine = as_woe_engine(binner)   # 也可以传 WOE_Master
woe_table = engine.get_woe_table(features)
train_woe = engine.transform(train_df, features)
```

更多说明见 [WOE 分箱引擎](woe_binning_engine.md)。

## 4. 与特征筛选联动

训练期拟合一次分箱器，后续筛选、监控、建模都复用同一对象：

```python
from Modeling_Tool import PSICalculator, VarExtractionInsights, CorrelationFilter

psi = PSICalculator(binning_engine=binner).calculate(train_df, oot_df, features)

iv_report = VarExtractionInsights(
    train_df, "bad_flag", "./iv_plots/",
    woe_engine="monotone", woe_binner=binner,
).get_var_analysis_report(train_df, features)

keep_vars = CorrelationFilter(
    train_df, "bad_flag", corr_cutpoint=0.7,
    woe_engine="monotone", woe_binner=binner,
).remove_highly_correlated(features)

train_woe = binner.apply_woe(train_df)
```

## 5. 单调性检查

```python
from Modeling_Tool import is_monotonic, get_overall_woe_table

for var in features:
    woe_table = get_overall_woe_table(woe, train_df, [var])
    mono, direction = is_monotonic(woe_table, "WOE", direction="auto")
    print(var, mono, direction)
```

## 6. 单独 WOE 转换

```python
from Modeling_Tool import woe_transform, woe_transformation

single_df, single_map = woe_transform(train_df, var="age", dep="bad_flag", nbins=10)
batch_result = woe_transformation(train_df, varlist=features, dep="bad_flag", nbins=10)
```

## 常见问题

??? question "什么时候选择 MonotoneWOEBinner？"

    当变量会进入评分卡、需要更强可解释性和单调约束时，优先使用 `MonotoneWOEBinner`。

??? question "为什么要在筛选阶段传入 binner？"

    因为 PSI / IV / KS 应该基于最终上线的同一套分箱计算，否则筛选指标和建模输入可能不一致。
