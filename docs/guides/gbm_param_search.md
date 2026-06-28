# GBM 超参搜索

`GradientBoostingModel.param_search(...)` 为 LightGBM / XGBoost 提供与 `LRMaster.grid_search_params(...)` 对齐的 **INS / OOS / OOT holdout 超参搜索**。

它不是 k-fold CV，而是针对风控建模常用的多时间窗验证：每个候选参数在训练集拟合，然后在 `eval_sets` 中的每个数据集上计算 AUC，并按指定目标函数选择最优组合。

## 1. Grid Search

`engine="grid"` 时，`search_space` 使用普通网格：每个参数给一个候选值列表，内部做笛卡尔积。

```python
from Modeling_Tool import GradientBoostingModel

base_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "early_stopping_rounds": 30,
    "eval_metric": "auc",
}

gbm = GradientBoostingModel("lgb", base_params)

results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    },
    engine="grid",
    objective="oot_gap_penalized",
    primary_set="oot",
    gap_ref_sets=["ins", "oos"],
    refit=True,
)

print(gbm.best_params_)
print(gbm.search_results_.head())
```

返回的 `results` 按 `score` 降序排列，包含：

- 参数列（如 `num_leaves`、`learning_rate`、`max_depth`）
- 每个 eval set 的 AUC：`AUC_ins` / `AUC_oos` / `AUC_oot`
- `gap`（仅 `oot_gap_penalized` 下）
- `score`

## 2. Optuna Search

`engine="optuna"` 时，需要额外安装：

```bash
pip install supermodelingfactory[optuna]
```

`search_space` 使用紧凑搜索空间定义：

```python
gbm = GradientBoostingModel("xgb", {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "eval_metric": "auc",
})

results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={
        "max_depth": ("int", 2, 6),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_weight": ("categorical", [1, 5, 10]),
    },
    engine="optuna",
    n_trials=50,
    primary_set="oot",
    random_state=42,
    refit=True,
)
```

也支持 dict 形式：

```python
search_space = {
    "max_depth": {"type": "int", "low": 2, "high": 6},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
}
```

## 3. Objective

与 `LRMaster.grid_search_params(...)` 一致，支持三种选择目标：

| objective | 选择标准 |
|---|---|
| `'oot_gap_penalized'`（默认） | `AUC[primary] - abs(mean(AUC[gap_refs]) - AUC[primary])`，同时鼓励主集表现和惩罚过拟合 gap |
| `'max_primary'` | 直接最大化 `AUC[primary]` |
| callable | 自定义 `f(metric_dict) -> float`，其中 `metric_dict` 为 `{集名: AUC}` |

自定义 objective 示例：

```python
def stable_oot(metrics):
    return metrics["oot"] - 0.5 * abs(metrics["ins"] - metrics["oot"])

results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={"max_depth": [3, 4, 5]},
    objective=stable_oot,
    primary_set="oot",
)
```

## 4. Validation Set

GBM 训练本身需要 validation set。`param_search` 默认按以下顺序选择：

1. `eval_sets["oos"]`，如果存在
2. `eval_sets["validation"]` 或 `eval_sets["valid"]`，如果存在
3. `primary_set`

也可以显式指定：

```python
gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={"max_depth": [3, 4, 5]},
    validation_set="oos",
)
```

## 5. 样本权重

`param_search` 支持 `weight_col`（训练集）和 `eval_weight_col`（各 eval set DataFrame 中的权重列）。
候选模型在训练与 holdout 评分时均使用对应权重计算加权 AUC：

```python
results = gbm.param_search(
    data=ins_woe,
    varlist=woe_features,
    tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    search_space={
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    },
    engine="grid",
    objective="oot_gap_penalized",
    primary_set="oot",
    gap_ref_sets=["ins", "oos"],
    weight_col="sample_wgt",        # 训练集权重列（须在 data 中）
    eval_weight_col="sample_wgt",   # 各 eval set 的权重列
    refit=True,
)

print(gbm.best_params_)
print(gbm.search_results_[["max_depth", "learning_rate", "AUC_oot", "score"]].head())
```

与 `LRMaster.grid_search_params` 对称：`weight_col` 控制候选拟合，`eval_weight_col` 控制
各 holdout 上的评分。也可通过 `fit_kwargs` 额外透传 `sample_weight` / `eval_sample_weight` 数组。

!!! note "权重列须存在于所有相关 DataFrame"

    `weight_col` 必须在 `data` 中存在；`eval_weight_col` 必须在 `eval_sets` 的每个
    DataFrame 中存在，否则搜索启动时会报 `KeyError`。

## 6. 返回值与副作用

- **返回**：按 `score` 降序的 `pandas.DataFrame`
- **写入**：`gbm.best_params_` 和 `gbm.search_results_`
- **更新**：把最优参数合并进 `gbm.params`
- **重训**：`refit=True` 时，用最优参数在 `data` 上重训当前 `gbm`

!!! note "目前仅支持 AUC"

    `metric` 目前只支持 `'auc'`。如果需要 KS / logloss / brier，可先通过 callable objective 在搜索结果基础上扩展，后续版本再内置更多指标。

!!! tip "与 LR 搜索的区别"

    LR 搜索方法名为 `grid_search_params(...)`，GBM 使用 `param_search(...)`，因为它同时支持 grid 和 Optuna 两种 engine。
