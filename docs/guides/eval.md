# 模型评估

SuperModelingFactory 在 [`Eval`](../api/eval.md) 子包提供**Gains 表 / 性能汇总 / ROC-KS-PR 图 / 链式评估流水线**等工具。

## 样本权重评估

传入 `weight_col`（或底层函数的 `sample_weight`）后，Gains 表、性能汇总、ROC/KS/Lift
等指标均按权重聚合。**不传权重时与历史行为完全一致**（向后兼容）。

### N 与 N_RAW

Gains 表与分箱聚合输出中，样本量有两套计数：

| 列 | 含义 | 典型用途 |
|----|------|----------|
| `N` | 该分箱内**权重之和**（有效样本量） | 加权坏样本率、Lift、累计捕获率 |
| `N_RAW` | 该分箱内**行数**（原始记录数） | 审计、核对原始户数/笔数 |

举例：一行代表一个账户、权重为贷款余额时，`N_RAW` 是账户数，`N` 是余额总量。
加权场景下，坏样本率 = `sum(y * w) / sum(w)`，Lift = 该箱加权坏率 / 总体加权坏率。

### 加权指标语义

| 指标 | 加权行为 |
|------|----------|
| `AUC` | 透传 sklearn `roc_auc_score(..., sample_weight=...)` |
| `KS` | 在加权 ROC 曲线上取最大 TPR−FPR |
| `Gini` | `2 * AUC - 1`（基于加权 AUC） |
| `Top10%_TargetRate` | Top 10% 分数段内的**加权**坏样本占比 |
| `Top10%_Lift` | Top 10% 分数段的**加权**提升倍数 |
| Gains 坏样本率 / WOE / IV | 按 `N`（权重和）聚合 |

## 1. 多数据集评估 —— `PerformanceEvaluator`

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
    weight_col="sample_wgt",   # 可选：各 add_dataset 的 DataFrame 中须有该列
)

perf = (
    evaluator.add_dataset("train", train_woe)
              .add_dataset("test",  test_woe)
              .add_dataset("oot",   oot_woe)
              .evaluate()
)

print(perf[["index", "KS", "AUC", "Top10%_TargetRate"]])
```

### 输出指标

| 指标 | 含义 |
|------|------|
| `KS` | Kolmogorov-Smirnov（最大 TPR-FPR） |
| `AUC` | ROC 曲线下面积 |
| `Gini` | `2*AUC - 1` |
| `Top10%_TargetRate` | Top 10% 分数段内的坏样本占比 |
| `Top10%_Lift` | Top 10% 分数段的提升倍数 |
| `AvgScore` | 平均分数 |

### 多 y 标签对比

`tgt_name` 可传入**多个 y 标签的 list/tuple**，对每个标签分别评估：

- **表格**：新增 `tgt_name` 列，各标签结果**纵向拼接**为一张表；
- **图片**：每个标签**各出一张图**，`to_show=True` 时循环显示；`fig_save_path` 自动按标签加后缀（`perf.png` → `perf_<label>.png`）。

```python
perf = (
    PerformanceEvaluator(
        tgt_name=["bad_dpd7", "bad_dpd30"],   # 多个 y 标签
        model=gbm._model.model,
        feature_cols=woe_features,
        weight_col="sample_wgt",
    )
    .add_dataset("train", train_woe)
    .add_dataset("oot",   oot_woe)
    .evaluate(to_show=True)                    # 每个标签各出一张图
)

# 输出含 tgt_name 列, 各标签纵向拼接
print(perf[["tgt_name", "index", "KS", "AUC"]])
```

> 传单个 `str`（如 `tgt_name="bad_flag"`）时行为不变，输出不含 `tgt_name` 列（向后兼容）。

## 2. Gains 表 —— `GainsTableCalculator`

按分数分箱的收益表，是业务方最熟悉的报表形式。

```python
from Modeling_Tool import GainsTableCalculator

gains = GainsTableCalculator(
    data=test_woe,
    score="prob",       # 分数列
    dep="bad_flag",
    weight_col="sample_wgt",     # 可选：加权 Gains
    weighted_binning=True,       # True=按累计权重等频分箱；False=按行数（默认）
    nbins=10,
)
gains_table = gains.calculate()
print(gains_table)   # 含 N（权重和）与 N_RAW（行数）
```

`get_gains_table` / `get_gains_table_by_cust_metrics` 同样接受 `weight_col`；
加权时坏/好样本计数、坏样本率、Lift、KS、WOE、IV 均按权重计算。

## 3. 自定义指标 —— `get_gains_table_by_cust_metrics`

```python
from Modeling_Tool import get_gains_table_by_cust_metrics

gains = get_gains_table_by_cust_metrics(
    data=test_woe,
    score="prob",
    dep="bad_flag",
    weight_col="sample_wgt",
    nbins=10,
    eval_metrics=["income", "n_overdue"],   # 待聚合的列
    metric_agg_func="mean",
)
```

## 4. 交叉风险矩阵 —— `cross_risk`

按**两个分数分箱**做联合风险评估（典型场景：新模型 vs 老模型对比）。

```python
from Modeling_Tool import cross_risk

risk_matrix = cross_risk(
    data=test_woe,
    score_list=["score_old", "score_new"],
    dep="bad_flag",
    weight_col="sample_wgt",
    nbins=5,
)
print(risk_matrix)
```

## 5. 链式评估流水线 —— `EvaluationPipeline`

按条件分组/子集后执行自定义函数：

```python
from Modeling_Tool import EvaluationPipeline, Model_Evaluation_Tool

m_eval = Model_Evaluation_Tool(
    data=test_woe,
    dep="bad_flag",
    comp_scrlist=["prob"],
    weight_col="sample_wgt",
)

pipeline = (
    EvaluationPipeline(m_eval)
    .group_by("apply_month", min_size=100)        # 按月份分组
    .subset_by({"city_grade": ["A", "B"]}, name="top_city")  # 筛选头部城市
)

def per_group_metrics(current_data):
    return current_data.groupby("apply_month").agg(
        bad_rate=("bad_flag", "mean"),
        avg_score=("prob", "mean"),
        n=("bad_flag", "size"),
    )

result = pipeline.apply(per_group_metrics)
```

## 6. ROC / KS / PR / KDE 图 —— `evaluate_model.py`

直接绘制图片到本地：

```python
from Modeling_Tool import evaluate_performance, comparison_performance

# 单模型：datasets = {数据集名: {'y_true':, 'y_score':, 'sample_weight': (可选)}}
evaluate_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score": test_woe["prob"],
            "sample_weight": test_woe["sample_wgt"],   # 可选
        }
    },
    to_show=False,
    save_path="./output/perf/",
)

# 多模型对比：每个数据集用 y_score_dict 放多个模型的分
comparison_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score_dict": {"lgb": test_woe["prob_lgb"], "lr": test_woe["prob_lr"]},
            "sample_weight": test_woe["sample_wgt"],
        },
        "oot": {
            "y_true": oot_woe["bad_flag"],
            "y_score_dict": {"lgb": oot_woe["prob_lgb"], "lr": oot_woe["prob_lr"]},
            "sample_weight": oot_woe["sample_wgt"],
        },
    },
    to_show=False,
    save_path="./output/perf_compare/",
)
```

`calc_pr` / `calc_roc` 也接受 `sample_weight`，透传至 sklearn。
分箱聚合（`calc_equid_dist` / `calc_equid_pct` / `calc_fixed_pct`）在提供权重时输出
`n`（权重和）与 `n_raw`（行数）两列。

生成的图包括：

- ROC 曲线（含 AUC）
- KS 曲线
- PR 曲线
- 分数 KDE 分布
- 累计分布图
- Gain / Lift 图

## 7. Lift 表 —— `calc_lift_apt`

```python
from Modeling_Tool import calc_lift_apt

lift_table = calc_lift_apt(
    y_true=test_woe["bad_flag"],
    y_score=test_woe["prob"],
    sample_weight=test_woe["sample_wgt"],   # 可选
    start=1.5, stop=3.0, step=0.5,   # lift 阈值区间与步长
)
print(lift_table)
```

## 完整评估流水线

```python
from Modeling_Tool import (
    PerformanceEvaluator, GainsTableCalculator,
    get_gains_table_by_cust_metrics, evaluate_performance,
)

# 1) 多数据集汇总
perf = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
    weight_col="sample_wgt",
).add_dataset("train", train_woe) \
 .add_dataset("test",  test_woe).evaluate()

# 2) Gains 表 + 自定义指标
gains = get_gains_table_by_cust_metrics(
    test_woe, score="prob", dep="bad_flag", nbins=10,
    weight_col="sample_wgt",
    eval_metrics=["income"], metric_agg_func="mean",
)

# 3) 输出图
evaluate_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score": test_woe["prob"],
            "sample_weight": test_woe["sample_wgt"],
        }
    },
    to_show=False,
    save_path="./output/perf/",
)
```

## 常见问题

??? question "AUC 与 KS 不一致（AUC 高 KS 低）"

    通常意味着**分数集中在中段**。检查 Top 10% 分数段覆盖了多少坏样本。

??? question "OOT AUC 远低于测试集"

    PSI > 0.25 表示分布显著漂移，需做：

    1. 检查入模变量 PSI
    2. 必要时重新训练（用更新窗口的样本）

??? question "Gains 表里 N 与 N_RAW 差很多，该看哪个？"

  业务指标（坏样本率、Lift、捕获率）基于 `N`（权重和）。
  `N_RAW` 仅反映原始行数，用于核对「这一箱有多少户/笔」。
  若一行一户且权重均为 1，则 `N == N_RAW`。

??? question "训练用了权重，评估忘记传会怎样？"

  评估默认等权（每行权重 1），与加权训练模型的目标分布不一致，
  可能导致 AUC/KS/Lift 与训练期认知偏差。请在 `PerformanceEvaluator`、
  `GainsTableCalculator` 等处统一传入相同的 `weight_col`。
