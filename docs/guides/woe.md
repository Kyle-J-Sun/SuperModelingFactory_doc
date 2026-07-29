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

## 低占比特殊值治理（SV Bin Governance，0.8.0）

0.6.7 的 `small_bin_policy` / `min_bin_size` 只治理**普通区间箱**；特殊值（special value，下称 SV）箱一直是**无条件**取经验 WOE 并计入总 IV。当某个 SV 占比极低（例如 `-1` 只占 0.05%）时，`ln(pct_bad/pct_good)` 由极少数样本估计，方差极大、IV 虚高、上线后 PSI 容易漂移。

0.8.0 为此引入**两组正交**的 SV 治理开关。四个参数在两个引擎上**同名同义、口径一致**：`MonotoneWOEBinner.__init__` 是构造器参数，`WOE_Master.fit()` / `update_woe()` 是方法参数。

| 参数 | 类型 / 默认 | 取值 | 语义 |
|------|-------------|------|------|
| `sv_min_bin_size` | `float = 0.0` | `[0.0, 1.0)` | SV 箱占**全量样本**的占比阈值；`0.0` = 关闭 |
| `sv_small_policy` | `str = "keep"` | `keep` / `neutral` / `merge_missing` | 占比低于阈值的 SV 箱如何兜底 |
| `sv_woe_smoothing` | `str = "none"` | `none` / `laplace` | 是否把 SV 箱 WOE 向全局 base rate 收缩 |
| `sv_smoothing_alpha` | `float = 0.0` | `>= 0.0` | 平滑强度 α；`0.0` = 关闭 |

!!! note "默认值严格等于旧行为"
    四个参数的默认值组合（`0.0` / `"keep"` / `"none"` / `0.0`）与 0.7.2 **逐位一致**，升级 0.8.0 不会改变任何既有产出。非法取值在 `__init__` / `fit()` 入口即抛 `ValueError`，风格与 G08 `small_bin_policy` 一致。

### 方式1 —— 低占比兜底（`sv_min_bin_size` + `sv_small_policy`）

占比按 `prop = n_bin / N_total` 计算（**分母不加 eps**），判定用**严格小于** `prop < sv_min_bin_size`；等于阈值**不**触发。

- `keep`（默认）：取经验 WOE，零行为变更。
- `neutral`：亚阈值 SV 箱 `woe = 0.0`、`iv = 0.0`。最稳，等价于"这个 SV 不提供任何证据"。
- `merge_missing`：亚阈值 SV 箱的 `bad` / `good` 计数**并入 `[Missing]` 箱**，`[Missing]` 箱随后按经验公式**重算** WOE；被合并行存表的 `woe` 会被**改写为 `[Missing]` 重算后的 WOE**，`iv` 置 `0` 避免重复计入总 IV。若该特征**没有** `[Missing]` 箱 → 降级为 `neutral` 并 `warnings.warn(UserWarning)`。

!!! tip "`merge_missing` 不需要改 transform"
    采用的是 **rewrite-stored-WOE** 方案：被合并 SV 行的存表 WOE 直接写成 `[Missing]` 的 WOE。因此 `apply_woe()` / `mapping_woe()` 照常按 `bin_label → WOE` 查表即可命中正确值，**两个引擎的 transform 路径都没有任何改动**，fit→transform 往返自动一致。

治理结果可从结果表的 `sv_policy_applied` 列审计，取值为
`keep` / `neutral` / `neutral(fallback)` / `merged_into_missing` / `merge_target`。

### 方式2 —— SV WOE 平滑（`sv_woe_smoothing="laplace"`）

平滑**只作用于 SV 箱**，普通区间箱完全不受影响。采用的是**坏率收缩（bad-rate shrinkage）**：先把箱内坏率向全局坏率收缩，再折回计数占比。

```
p = N_bad / (N_bad + N_good)          # 全局坏率
n_bin = n_bad + n_good                # 箱内样本量

r = (n_bad + alpha * p) / (n_bin + alpha)        # 坏率向 p 收缩，alpha 与 n_bin 竞争

pct_bad_smoothed  = n_bin * r       / N_bad      # 折回占比，分母仍是全局 N_bad / N_good
pct_good_smoothed = n_bin * (1 - r) / N_good

woe = ln(pct_bad_smoothed / pct_good_smoothed)
iv  = (pct_bad_smoothed - pct_good_smoothed) * woe
```

收敛性质（这也是选用该形式的原因）：

- `alpha = 0` → `r` 退化为经验坏率，**逐位还原**旧经验 WOE（回归护栏）。
- `alpha → ∞` → `r → p`，两个占比之比趋于全局比，`woe → 0`，且是**单调**收缩。
- α 与 `n_bin` 竞争 ⇒ **样本越少的箱收缩越强**，正是低占比 SV 需要的性质。

!!! danger "不要用"人口分母伪计数"形式"
    早期设计曾把伪计数放在人口分母上（`(n_bad + alpha*p) / (N_bad + alpha)`）。该式 `alpha → ∞` 时收敛到 `logit(p) = ln(p/(1-p)) ≠ 0`，而且**不单调**——加大平滑强度反而可能推高 |WOE|。两个引擎实现的都是上面的坏率收缩式，上方公式为唯一权威定义。

### 正交组合语义（执行顺序固定）

两个开关可独立启用，也可叠加。fit 阶段对每个 SV 箱按固定顺序决策：

```
1. prop = n_sv / N_total
2. if sv_small_policy != "keep" and sv_min_bin_size > 0 and prop < sv_min_bin_size:
       走方式1 兜底（neutral / merge_missing），【不再】走方式2
   else:
       若 sv_woe_smoothing == "laplace" 且 alpha > 0 → 走方式2 平滑
       否则 → 经验 WOE（旧行为）
```

即**方式1 优先**：一个亚阈值箱**永远不会被平滑**，它已经被兜底掉了；方式2 只作用于"占比达标、保留经验 WOE"的 SV 箱。

`merge_missing` + `laplace` 同时开启时：合并目标 `[Missing]` 箱按**经验**公式重算（**不**平滑），因为合并后的桶应当反映真实的 post-merge 坏率；其他占比达标的 SV 箱仍照常平滑。

### `[Missing]` 箱也是受治理的 SV 箱

`[Missing]`（NaN）箱在**两个引擎**里都是一个正常的受治理 SV 箱：占比达标就照常平滑，亚阈值就照常 `neutral` 置零。它唯一的特殊之处是在 `merge_missing` 下**只能当合并目标、不能当合并来源**。

这与 `missing_bin_strategy`（`empirical_special` / `fixed_woe` / `fail`）**正交**：后者只治理 NaN 缺失箱的语义，前者治理**所有** SV 箱（含 `-1` 这类非 NaN 哨兵值）。

### 用法示例

```python
binner = MonotoneWOEBinner(
    feature_cols=feats, target_col="y",
    special_values=[-1, -999999, float("nan")],
    # 占比 < 1% 的 SV 箱直接中性化
    sv_min_bin_size=0.01,
    sv_small_policy="neutral",       # keep / neutral / merge_missing
    # 占比达标的 SV 箱做坏率收缩
    sv_woe_smoothing="laplace",      # none / laplace
    sv_smoothing_alpha=50.0,
)
binner.fit(train)
binner.get_final_bins()["risk_score"]   # sv_policy_applied 列记录每个 SV 箱的实际处置
```

`WOE_Master` 侧同名同义，只是落在 `fit()` 上：

```python
woe = WOE_Master(train_data=train, varlist=feats, dep="y")
woe.fit(
    nbins=10, equal_freq=True, spec_values=[-1, -999999],
    sv_min_bin_size=0.01, sv_small_policy="neutral",
    sv_woe_smoothing="laplace", sv_smoothing_alpha=50.0,
)
```

### Pipeline 层暴露

`CreditModelPipelineConfig` 与 `FeatureValidationPipelineConfig` 的**两个**透传字典都已带上四个 `sv_*` 默认键：`woe_params`（喂 `equal_freq` / `WOE_Master`）和 `monotone_woe_params`（喂 `MonotoneWOEBinner`）。

```python
from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig

cfg = CreditModelPipelineConfig(
    target_col="y",
    woe_engine="monotone",
    monotone_woe_params={
        "n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2,
        "special_values": [-999999],
        "sv_min_bin_size": 0.01,
        "sv_small_policy": "neutral",
    },
)
```

FVP 的 `config_snapshot` 会原样 dump 这两个字典，`sv_*` 自动进快照，便于复现"某个模型用了什么 SV 治理口径"。

!!! warning "维护者注意：FVP 的白名单必须同步"
    `FeatureValidationPipeline` 的 monotone 分支**不是**直接 `**monotone_woe_params`，而是先过一层
    `_MONOTONE_INIT_KEYS` 白名单：

    ```python
    init_params = {k: v for k, v in params.items() if k in self._MONOTONE_INIT_KEYS}
    ```

    **不在白名单里的 key 会被静默丢弃，不报错、不告警**——参数看起来传进去了，实际治理完全没生效。
    四个 `sv_*` 已加入 `_MONOTONE_INIT_KEYS`（同样也加进了
    `Feature_Screen._MONOTONE_INIT_KEYS`，否则筛选期复用的 WOE 拟合会漏掉治理，导致筛选 IV 与建模 WOE 口径分裂）。
    今后给 `MonotoneWOEBinner.__init__` 新增任何参数，都必须同步这两份白名单。

    `sv_*` 是**构造器**参数，**不要**加进 `_MONOTONE_FIT_KEYS`（`{chi2_binning, chi2_p, chi2_init_size, n_jobs}`），
    否则会被当成 `fit()` kwarg 传下去而抛 `TypeError`。CM 侧的 monotone fit-only `pop` 名单同理，
    保持不含 `sv_*` 才能让它们正确进入构造器。
