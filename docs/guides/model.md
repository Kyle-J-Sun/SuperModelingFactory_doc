# 模型训练

SuperModelingFactory 在 [`Model`](../api/model.md) 子包封装了**逻辑回归（评分卡首选）、LightGBM、XGBoost** 三大类模型，并提供**后向变量消元**辅助工具。

## 1. 逻辑回归 —— `LRMaster`

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
# fit 接收 (data, varlist, tgt_name)，而非 (X, y)
lr.fit(train_woe, woe_features, "bad_flag")

# 统计摘要：系数、标准误、z、p-value、置信区间
summary = lr.get_statsmodel_summary()
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
# LR 系数（按绝对值排序）；列为 varlist / coef / importance
varimp = lr.get_variable_importance()
print(varimp[["varlist", "coef", "importance"]])
```

### 逐步变量选择

`stepwise_selection(data, varlist, tgt_name, ...)` 基于 AIC/BIC 做前向 / 后向 / 双向选择：

```python
from Modeling_Tool import LRMaster

lr = LRMaster(params={"C": 1.0})
selected = lr.stepwise_selection(
    train_woe, woe_features, "bad_flag",
    criterion="aic",        # 'aic' 或 'bic'
    direction="both",       # 'forward' / 'backward' / 'both'
)
print(f"逐步选择保留 {len(selected)} 个变量: {selected}")
```

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

# 预测（返回正类概率，一维）
proba = gbm.predict(test_X)

# 校准（可选）
gbm.calibrate(val_X, val_y, method="isotonic")
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

### 增量学习（Warm-start）

在已有模型基础上、用新数据继续训练，而非从头开始。`GradientBoostingModel`
提供一套**同时兼容 lgb 和 xgb** 的接口：用旧模型的 log-odds 输出作为 `init_score`
在新数据上继续学习，打分时再把「旧模型 margin + 新模型贡献」融合成最终概率。

#### lgb / xgb 的差异

| 环节 | XGBoost | LightGBM |
|------|---------|----------|
| 取原始 margin（log-odds） | `predict(X, output_margin=True)` | `predict(X, raw_score=True)` |
| 训练时传偏移 | `fit(X, y, base_margin=...)` | `fit(X, y, init_score=...)` |
| 预测时直接加偏移 | 原生支持 | **不支持** |

`GradientBoostingModel` 把这些差异封装在内部：对外统一用 `init_score`，预测融合统一走
`sigmoid(base_margin + 新模型 raw score)`——这是唯一对两种框架行为一致的做法（LightGBM
在预测期并不支持注入 init_score）。

#### 三步用法

```python
from Modeling_Tool import GradientBoostingModel

# 1) 用旧模型取 base margin（log-odds），作为增量训练的起点
base_margin_train = init_model.get_base_margin(train_X)

# 2) 增量训练：把 base margin 当作 init_score 传入（lgb / xgb 统一用 init_score）
new_model = GradientBoostingModel("xgb", params)   # "lgb" 同理
new_model.fit(train_X, train_y, val_X, val_y, init_score=base_margin_train)

# 3) 融合预测：sigmoid(base_margin + 新模型 raw score)
base_margin_score = init_model.get_base_margin(score_X)
proba = new_model.predict_with_base_margin(score_X, base_margin_score, return_prob=True)
# return_prob=False 则返回融合后的原始 log-odds
```

!!! note "偏移只作用于训练集"

    与常见生产实现一致，`init_score` 偏移只注入训练集；验证集未加偏移，因此早停的
    eval 指标是在「未加偏移」的空间上评估的。如需严格一致，可后续透传 lgb 的
    `eval_init_score` / xgb 的 `base_margin_eval_set`。

!!! tip "init_model 应是 GradientBoostingModel"

    `get_base_margin` / `predict_with_base_margin` 是 `GradientBoostingModel` 的实例
    方法，因此基准模型 `init_model` 也应是 `GradientBoostingModel`（而非裸
    `LGBMClassifier` / `XGBClassifier`）。

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
print(result)   # 后向消元每轮的变量数与性能
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
        model.fit(train_woe, woe_features, "bad_flag")
    else:
        model.fit(train_woe[woe_features], train_woe["bad_flag"],
                  test_woe[woe_features],  test_woe["bad_flag"])

    raw_model = model._model.model if name != "LR" else model.model
    evaluator = PerformanceEvaluator(
        tgt_name="bad_flag",
        model=raw_model,
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

    `LRMaster` / `GradientBoostingModel` 未内置超参搜索 / 交叉验证接口，
    请用 sklearn 的 `cross_val_score` 自行包装（注意 GBM 取底层估计器用
    `model._model.model`，LR 用 `model.model`）：

    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model._model.model, X, y, cv=5, scoring="roc_auc")
    ```
