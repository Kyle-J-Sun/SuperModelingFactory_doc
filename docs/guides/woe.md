# WOE 编码

WOE（Weight of Evidence）把分类变量或分箱后的连续变量映射为线性可分的数值，是评分卡建模的标准做法。

SuperModelingFactory 在 [`WOE`](../api/woe.md) 子包提供 **主控类 + 单调分箱器 + 转换器 + 绘图器 + 分箱引擎适配器**。

## 1. 主控类 —— `WOE_Master`

```python
from Modeling_Tool import SMF_MISSING_BIN, WOE_Master

woe = WOE_Master(
    train_data=train_df,
    varlist=features,
    dep="bad_flag",
    missing_ref_value=SMF_MISSING_BIN,
)
woe.fit(nbins=10, equal_freq=True)

train_woe = woe.transform(train_df)
test_woe = woe.transform(test_df)
oot_woe = woe.transform(oot_df)
```

### 向量化执行与宽表性能

`WOE_Master` 的行级计算采用向量化路径：

- 分箱通过 `pandas.cut` 生成 categorical codes，再用 `numpy.take` 一次映射整列的 bin label。
- WOE 映射按整列 lookup，不使用逐行 `Series.apply()`。
- 多变量转换会先收集全部 `{feature}_woe` 数组，最后一次性 `concat` 回原表，避免宽表逐列插入引起 DataFrame fragmentation。
- WOE 和 IV 共用同一次对数计算；`WOEIVCalculator.calc_both()` 不会分别重复计算 WOE 与 IV。
- `WOE_Master.transform()` 会自动沿用实例的 `missing_ref_value`，训练和推理使用同一个缺失值分箱口径。

不同变量的分箱边界并不相同，因此 `fit()` / `transform()` 仍保留**变量级循环**；循环内部没有 Python 逐行回调。向量化的含义是按整列运算，而不是强迫所有变量共用一套 bins。

已有映射表也可以直接批量转换。独立调用 `mapping_woe()` 时，如训练使用了自定义缺失哨兵，应显式传入同一个值：

```python
from Modeling_Tool import mapping_woe

scored = mapping_woe(
    data=oot_df,
    varlist=features,
    woe_mapping_table=woe.get_mapping_table(),
    missing_ref_value=SMF_MISSING_BIN,
)
```

需要同时计算 WOE 与 IV 时，优先使用一次性接口：

```python
from Modeling_Tool import WOEIVCalculator

woe_values, iv_values = WOEIVCalculator(
    bin_summary,
    bad_pct_col="BAD_PCT_PER_BIN",
    good_pct_col="GOOD_PCT_PER_BIN",
).calc_both()
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
| `apply_woe(df, varlist=None)` | WOE 转换；`varlist` 可限制只转换指定变量，适合宽表分块 |
| `get_final_bins()` | 导出分箱结果（含 WOE/IV） |
| `load_woe_bins(bins_dict)` | 加载已有分箱 |
| `get_bin_edges()` | 取分箱边界列表 |
| `export_woe_report(path)` | 导出 Excel 报告 |
| `plot_woe_graph(dir, group_name=)` | 输出 WOE 图 PNG |

### Format-A 分箱往返（0.7.2）

`get_final_bins()` 返回的每个 DataFrame 都带有 `attrs["smf_woe_format_a"]`。其中的精确数值边界、稀疏箱号、类别成员和 `missing_woe` 只有在 metadata 摘要与行身份校验通过时才会被 `load_woe_bins()` 使用；直接传递或 pickle 往返可以保住这些信息，损坏或陈旧 metadata 会安全退回可见标签解析。

!!! warning "CSV/Excel 不是精确往返载体"
    CSV/Excel 会丢失 `DataFrame.attrs`。从这两类文件回载时，`bin_label`（默认 `.8g` 显示精度）是唯一事实来源，无法还原文本之外的精确切点、原始稀疏箱号、歧义类别成员或非默认 `missing_woe`。需要精确恢复时，请直接传递 DataFrame 或使用 pickle 等保留 attrs 的格式。

类别转换会先做精确匹配，再做受支持的 `str()` dtype 回退。只有两者都失败的值才进入 `_unseen_category_stats`；回退成功的值仍会计入 `_categorical_transform_stats[feature]["fallback_match_rows"]` 并触发既有 tripwire 告警。

!!! note "Clustered by-group 图片规格"
    当 `group_name` 非空且 `bar_mode="clustered"` 时，by-group WOE 图固定使用
    `figsize=(16, 6)` 和 `dpi=200`，以容纳并排柱、分组 WOE 曲线和右侧图例。
    其他模式仍使用调用方传入的 `figsize` 与 `dpi`。

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

## 分箱治理（0.6.7+，G08/G09/G17）

`MonotoneWOEBinner` 新增三组治理参数，默认全部关闭（`None`/`"auto"`）、行为与旧版逐字节一致：

```python
binner = MonotoneWOEBinner(
    feature_cols=feats, target_col="y",
    # G08 小箱治理：坏/好样本数下限 + 三态策略
    min_bad_count=50, min_good_count=50, small_bin_policy="merge",  # merge/warn/raise
    # G09 方向治理：固定方向 或 参考标签推导；冲突三态
    monotone_direction={"util_rate": "increasing"},   # 或 "increasing"/"decreasing"/"auto"
    reference_target="y",                              # 与 monotone_direction 二选一
    direction_conflict_policy="raise",                 # warn/raise/keep
    # 缺失箱语义：empirical_special / fixed_woe / fail
    missing_bin_strategy="fail",
    # G17 refine 治理：refine 结果箱数低于 min_n_bins 时 warn/enforce/raise
    refine_min_n_bins_policy="enforce",
)
binner.fit(train)
binner.refine_dtree(train, max_depth=3)   # 0.6.7+：可限制树深
binner.get_direction_summary()            # feat / direction / direction_basis / is_monotonic
```

要点：

- `small_bin_policy="merge"` 向 WOE 更接近的邻箱合并，合并轨迹记录在结果的 `merge_trace`；`raise` 抛 `BinningPolicyViolation`（穿透逐特征容错，不会被吞进日志）。0.7.2 起类别特征也会在初始 `fit()` 阶段执行 `merge/warn/raise`，且 `merge` 不会越过 `min_n_bins`；到达下限仍有违规箱时可从 `_small_bin_stats[feature]["remaining_violation"]` 审计。
- 方向在 `fit` 前解析：串行与并行 worker 使用同一份 `_expected_direction`，杜绝串并行漂移。
- 这些参数可经 `monotone_woe_params` 从 FVP / CMP / feature_screen 直通底层 binner。
- 0.7.1 起，`refine_min_n_bins_policy` 默认 `"warn"`；如需完全关闭该检查，请显式传 `None`。
