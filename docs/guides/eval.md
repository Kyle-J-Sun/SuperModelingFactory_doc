# 模型评估

SuperModelingFactory 在 [`Eval`](../api/eval.md) 子包提供**Gains 表 / 性能汇总 / ROC-KS-PR 图 / 链式评估流水线**等工具。

## 1. 多数据集评估 —— `PerformanceEvaluator`

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
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
    nbins=10,
)
gains_table = gains.calculate()
print(gains_table)
```

## 3. 自定义指标 —— `get_gains_table_by_cust_metrics`

```python
from Modeling_Tool import get_gains_table_by_cust_metrics

gains = get_gains_table_by_cust_metrics(
    data=test_woe,
    score="prob",
    dep="bad_flag",
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

# 单模型：datasets = {数据集名: {'y_true':, 'y_score':}}
evaluate_performance(
    datasets={"test": {"y_true": test_woe["bad_flag"], "y_score": test_woe["prob"]}},
    to_show=False,
    save_path="./output/perf/",
)

# 多模型对比：每个数据集用 y_score_dict 放多个模型的分
comparison_performance(
    datasets={
        "test": {
            "y_true": test_woe["bad_flag"],
            "y_score_dict": {"lgb": test_woe["prob_lgb"], "lr": test_woe["prob_lr"]},
        },
        "oot": {
            "y_true": oot_woe["bad_flag"],
            "y_score_dict": {"lgb": oot_woe["prob_lgb"], "lr": oot_woe["prob_lr"]},
        },
    },
    to_show=False,
    save_path="./output/perf_compare/",
)
```

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
).add_dataset("train", train_woe) \
 .add_dataset("test",  test_woe).evaluate()

# 2) Gains 表 + 自定义指标
gains = get_gains_table_by_cust_metrics(
    test_woe, score="prob", dep="bad_flag", nbins=10,
    eval_metrics=["income"], metric_agg_func="mean",
)

# 3) 输出图
evaluate_performance(
    datasets={"test": {"y_true": test_woe["bad_flag"], "y_score": test_woe["prob"]}},
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
