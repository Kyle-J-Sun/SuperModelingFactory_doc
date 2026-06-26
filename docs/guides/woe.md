# WOE 编码

WOE（Weight of Evidence）把分类变量或分箱后的连续变量映射为线性可分的数值，是评分卡建模的标准做法。

SuperModelingFactory 在 [`WOE`](../api/woe.md) 子包提供**主控类 + 单调分箱器 + 转换器 + 绘图器**四类工具。

## 1. 主控类 —— `WOE_Master`

最简单的入口：

```python
from Modeling_Tool import WOE_Master

woe = WOE_Master(
    train_data=train_df,
    varlist=features,
    dep="bad_flag",
    missing_ref_value=-999999,
)

woe.fit(
    nbins=10,
    equal_freq=True,         # True=等频, False=等距
)

train_woe = woe.transform(train_df)
test_woe  = woe.transform(test_df)
oot_woe   = woe.transform(oot_df)
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `nbins` | `10` | 目标分箱数 |
| `equal_freq` | `True` | 等频 vs 等距分箱 |
| `tree_binning_seed` | `None` | 决策树预分箱随机种子（`fit` 参数） |
| `chi2_config` | `None` | 卡方分箱配置 `(init_bins, p)` 元组（`fit` 参数） |
| `missing_ref_value` | `-999999` | 缺失值填充 |
| `spec_values` | `[]` | 单独成箱的特殊值 |

### 持久化映射表

```python
from Modeling_Tool import WOE_Master, load_mapping_table

# 导出 WOE 映射表（实例方法）
woe.save_mapping_table("./output/woe_mapping.csv")

# 模块函数加载，返回 (varlist, woe_dict)
varlist, woe_dict = load_mapping_table("./output/woe_mapping.csv")

# 部署：载入新的 WOE_Master 实例后 transform 新数据
woe2 = WOE_Master(train_data=new_data, varlist=varlist, dep="bad_flag")
woe2.load_mapping_table("./output/woe_mapping.csv")
new_woe = woe2.transform(new_data)
```

## 2. 单调性检查 —— `is_monotonic`

LR 评分卡通常要求 WOE 单调。

```python
from Modeling_Tool import is_monotonic, get_overall_woe_table

for var in features:
    woe_table = get_overall_woe_table(woe, train_df, [var])
    mono, direction = is_monotonic(woe_table, "WOE", direction="auto")
    print(f"{var:20s}  单调={mono}  方向={direction}")
    # direction: 1=递增, -1=递减, 0=非单调
```

## 3. 贪心单调分箱器 —— `MonotoneWOEBinner`

如果 `WOE_Master` 产出非单调 WOE，使用 `MonotoneWOEBinner` 自动合并到单调。

```python
from Modeling_Tool.WOE.WOE_Monotone_Binner import MonotoneWOEBinner

binner = MonotoneWOEBinner(
    feature_cols=features,
    target_col="bad_flag",
    n_init_bins=20,
    min_bin_size=0.03,
    special_values=[-1, -100, -999999],
    cate_feats=["city_grade"],       # 类别特征直接算 WOE/IV
)

# 训练：贪心合并 + 卡方初始化
binner.fit(train_df, chi2_binning=True, chi2_p=0.95)

# 类别特征按坏率聚类
binner.refine_cate(max_bins=5)

# WOE 转换
train_woe = binner.apply_woe(train_df)

# 直接加载已有分箱
binner2 = MonotoneWOEBinner(feature_cols=features, target_col="bad_flag")
binner2.load_woe_bins(binner.get_final_bins())

# 一键导出 Excel 报告
binner.export_woe_report("./output/woe_report.xlsx")

# 绘图输出
binner.plot_woe_graph("./output/woe_charts/")
```

### 方法列表

| 方法 | 说明 |
|------|------|
| `fit(df, chi2_binning, chi2_p, n_jobs)` | 训练拟合（贪心单调合并） |
| `refine_cate(max_bins)` | 类别特征按坏率聚类合并 |
| `apply_woe(df)` | WOE 转换 |
| `get_final_bins()` | 导出分箱结果（含 WOE） |
| `load_woe_bins(bins_dict)` | 加载已有分箱 |
| `get_bin_edges()` | 仅取分箱边界列表（含 ±inf） |
| `export_woe_report(path)` | 导出 Excel 报告（含图表 Sheet） |
| `plot_woe_graph(dir, group_name=, _df_for_group=)` | 输出 WOE 图 PNG |
| `iv_summary` | 属性：逐特征 IV 汇总表 |

## 4. 单独 WOE 转换 —— `woe_transform`

不通过主控类，对**单变量**做 WOE 转换。

```python
from Modeling_Tool import woe_transform

new_df, woe_map = woe_transform(
    train_df=train_df,
    var="age",
    dep="bad_flag",
    nbins=10,
    equal_freq=True,
    fillna=-999999,
)
```

## 5. 批量 WOE 转换 —— `woe_transformation`

```python
from Modeling_Tool import woe_transformation

result = woe_transformation(
    train_df=train_df,
    varlist=features,
    dep="bad_flag",
    nbins=10,
)
```

## 6. WOE 绘图 —— `plot_woe`

```python
from Modeling_Tool import plot_woe

# 在 matplotlib 中展示 / 保存单变量 WOE 图
plot_woe(
    woe_df=woe_table,
    var_rename="年龄",
    to_show=False,
    save_dir="./output/woe_plot/",
    fig_name="age.png",
)
```

## 常见问题

??? question "训练集 WOE 单调但测试集 WOE 不单调"

    正常。`WOE_Master.fit()` 只保证**训练集**单调；
    测试集 / OOT 由于分布偏移可能轻微抖动。建议检查 PSI 是否过大。

??? question "类别特征如何分箱"

    两种方式：

    1. `MonotoneWOEBinner(..., cate_feats=["city_grade"])` 后调用 `.refine_cate(max_bins=5)` 按坏率聚类合并
    2. 直接使用 `WOE_Master`（会按目标编码自动分箱）

??? question "`MonotoneWOEBinner.export_woe_report` 报 `ExcelMaster` 找不到"

    需要 `PYTHONPATH` 包含 SuperModelingFactory 根目录：

    ```bash
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    ```
