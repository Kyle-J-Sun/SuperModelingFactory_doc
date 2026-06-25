# 模型训练

SuperModelingFactory 在 [`Model`](../api/model.md) 子包封装了**逻辑回归（评分卡首选）、LightGBM、XGBoost** 三大类模型，并提供**后向变量消元**辅助工具。

## 1. 逻辑回归 —— `LRMaster`

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
lr.fit(train_woe[woe_features], train_woe["bad_flag"])

# 完整摘要：系数、p-value、VIF、AIC/BIC
summary = lr.get_model_summary()
print(summary)
```

### 关键参数（透传 sklearn）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `C` | `1.0` | 正则化强度倒数，越小越强正则 |
| `penalty` | `"l2"` | `l1` / `l2` / `elasticnet` |
| `solver` | `"lbfgs"` | 优化算法 |
| `max_iter` | `100` | 最大迭代次数 |

### 变量重要性

```python
# LR 系数（绝对值排序）
varimp = lr.get_variable_importance()
print(varimp[["variable", "coefficient", "p_value"]])
```

### 逐步变量选择

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0})
selected = lr.stepwise_selection(
    ins_df=train_woe, oos_df=test_woe, oot_df=oot_woe,
    candidate_varlist=woe_features, dep="bad_flag",
    method="backward",
)
print(f"逐步选择保留 {len(selected)} 个变量: {selected}")
```

### 超参网格搜索

`grid_search_params` 在 **holdout** 口径上对超参做网格搜索（非 k 折 CV）：每个候选在 `data` 上训练，在 `eval_sets` 各集算 AUC，按 `objective` 选最优。默认目标 `oot_gap_penalized` = `AUC[primary] − |mean(AUC[gap_refs]) − AUC[primary]|`，即最大化主集（如 OOT）AUC 同时惩罚过拟合落差。

```python
import numpy as np
from Modeling_Tool import LRMaster

tuner = LRMaster(params={"C": 1.0, "solver": "lbfgs", "max_iter": 1000})
search_df = tuner.grid_search_params(
    data=ins_fit_woe, varlist=woe_features, tgt_name="bad_flag",
    eval_sets={"ins": ins_woe, "oos": oos_woe, "oot": oot_woe},
    param_grid={"C": np.logspace(-3, 2, 31)},     # 支持多参, 做笛卡尔积
    objective="oot_gap_penalized",                 # 或 "max_primary" / callable(auc_dict)->float
    primary_set="oot", gap_ref_sets=["ins", "oos"],
    refit=False,                                   # True 则用最优参在 data 上重训 self
)
print(tuner.best_params_)        # e.g. {'C': 0.21544}
# search_df 列 = 参数名 + AUC_<集名> + gap + score, 按 score 降序
```

- 返回 `search_df`（按 `score` 降序）；并设 `tuner.best_params_` / `tuner.search_results_`，把最优参合并进 `tuner.params`。
- `primary_set` 默认 = `eval_sets` 最后一个；`gap_ref_sets` 默认 = 除 primary 外其余集。
- `refit=True`（默认）时用最优参在 `data` 上重训 `self.model`。

## 2. 梯度提升模型 —— `GradientBoostingModel`

LightGBM / XGBoost 的统一接口。

```python
from Modeling_Tool import GradientBoostingModel

gbm = GradientBoostingModel(
    model_type="lgb",       # 'lgb' 或 'xgb'
    params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
        "eval_metric": "auc",
    },
)
gbm.fit(
    train_X, train_y,
    val_X,   val_y,
)

# 变量重要性
varimp = gbm.get_feature_importance()
print(varimp.head(15))

# 预测概率
proba = gbm.predict_proba(test_X)

# 校准（可选）
gbm.fit_calibrated_model(val_X, val_y, method="isotonic")
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `model_type` | `"lgb"` | `"lgb"` 或 `"xgb"` |
| `n_estimators` | `100` | 树数量 |
| `learning_rate` | `0.1` | 学习率 |
| `max_depth` | `-1` | 最大深度（-1=不限） |
| `early_stopping_rounds` | `None` | 早停轮数 |
| `eval_metric` | `"auc"` | 评估指标 |

### 单独使用 LightGBMModel / XGBoostModel

```python
from Modeling_Tool import LightGBMModel, XGBoostModel

lgb = LightGBMModel(params={"n_estimators": 200, "learning_rate": 0.05})
lgb.fit(train_X, train_y, val_X, val_y)

xgb = XGBoostModel(params={"n_estimators": 200, "max_depth": 6})
xgb.fit(train_X, train_y, val_X, val_y)
```

### 快速训练函数

```python
from Modeling_Tool import lgbm_quick_train, xgbm_quick_train

model = lgbm_quick_train(train_X, train_y, val_X, val_y,
                         params={"n_estimators": 200})
```

## 3. 后向变量消元 —— `BackwardVariableEliminator`

基于**累计重要性阈值**逐步剔除变量，常用于**轻量级变量筛选**。

```python
from Modeling_Tool import BackwardVariableEliminator

eliminator = BackwardVariableEliminator(
    model_type="lgb",
    train_data=train_woe,
    validation_data=test_woe,
    oot_data=oot_woe,
    params={"n_estimators": 100, "learning_rate": 0.1},
    y="bad_flag",
    results_output_dir="./output/",   # 构造器参数，非 fit 参数
    modelsave_dir="./models/",
)

eliminator.fit(x=woe_features)

result = eliminator.analyze()
print(result[["round", "n_features", "auc", "ks"]])
```

### 工作原理

1. 用全部特征训练一轮 LGB，记录每个特征的 gain 重要性
2. 剔除**累计重要性 < 阈值**的变量（如 `< 0.001`）
3. 重复 1–2 直至剩余变量数达下限或 AUC 不再提升

## 4. 模型持久化

```python
from Modeling_Tool import save_model, load_model

save_model(gbm._model.model, "./models/gbm_v1.pkl")

# 加载
loaded = load_model("./models/gbm_v1.pkl")
```

## 5. 评分函数

```python
from Modeling_Tool import scoring

scores = scoring(
    data=new_df,
    model=gbm._model.model,
    varlist=woe_features,
    scr_name="prob",
)
```

## 模型对比实践

```python
from Modeling_Tool import (
    LRMaster, GradientBoostingModel, PerformanceEvaluator,
)

models = {
    "LR":   LRMaster({"C": 1.0}),
    "LGB":  GradientBoostingModel("lgb", {"n_estimators": 200}),
    "XGB":  GradientBoostingModel("xgb", {"n_estimators": 200}),
}

results = {}
for name, model in models.items():
    if name == "LR":
        model.fit(train_woe[woe_features], train_woe["bad_flag"])
    else:
        model.fit(train_woe[woe_features], train_woe["bad_flag"],
                  test_woe[woe_features],  test_woe["bad_flag"])

    evaluator = PerformanceEvaluator(
        tgt_name="bad_flag",
        model=model._model.model,
        feature_cols=woe_features,
    )
    perf = evaluator.add_dataset("train", train_woe) \
                    .add_dataset("test",  test_woe).evaluate()
    results[name] = perf

# 对比 KS / AUC
for name, perf in results.items():
    print(f"{name}: AUC={perf['AUC'].mean():.4f}  KS={perf['KS'].mean():.4f}")
```

## 常见问题

??? question "LightGBM 训练报 `categorical_feature` 错误"

    确保类别列在 DataFrame 中是 `category` dtype，或在 params 中设置：

    ```python
    params["categorical_feature"] = ["city_grade"]
    ```

??? question "变量重要性总和不为 1"

    `get_feature_importance(importance_type='gain')` 返回归一化的相对值，
    `sum` 应为 1.0；若返回 `split`，则按分裂次数加权。

??? question "如何做超参搜索 / 交叉验证"

    `LRMaster.grid_search_params(...)` 提供 **holdout 口径**的超参网格搜索（见上「超参网格搜索」）。
    若需 **k 折交叉验证**，`LRMaster` / `GradientBoostingModel` 未内置，请用 sklearn 的
    `cross_val_score` 自行包装：

    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model._model.model, X, y, cv=5, scoring="roc_auc")
    ```
