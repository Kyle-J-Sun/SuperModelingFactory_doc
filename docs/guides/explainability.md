# 模型解释

SuperModelingFactory 在 [`explainability`](../api/explainability.md) 子包提供了基于 **SHAP** 的统一解释器 `ModelExplainer`。它能直接消费本工具链训练出的模型，自动选择合适的 SHAP 算法，输出**全局特征重要性**、**summary / dependence 图**以及**单样本归因**。

!!! note "安装可选依赖"
    `shap` 是可选依赖，需单独安装：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

    `shap` 采用懒加载：`from Modeling_Tool import ModelExplainer` 不会拉起 shap，只有调用 `explain()` / 绘图时才导入；未安装时会报出带安装提示的 `ImportError`。

## 自动选择解释器

| 模型 | 底层估计器 | SHAP 算法 | 是否需 `background_data` |
|------|-----------|-----------|--------------------------|
| `GradientBoostingModel('lgb'/'xgb')` | LGBMClassifier / XGBClassifier | `TreeExplainer` | 否 |
| `LRMaster` | sklearn `LogisticRegression` | `LinearExplainer` | **是** |
| 其他（如校准模型） | 任意 `predict_proba` 估计器 | model-agnostic `Explainer` | **是** |

## 1. 解释梯度提升模型（LGB / XGB）

```python
from Modeling_Tool import GradientBoostingModel, ModelExplainer

gbm = GradientBoostingModel("lgb", {
    "n_estimators": 200, "learning_rate": 0.05,
    "max_depth": 4, "early_stopping_rounds": 30, "eval_metric": "auc",
})
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],
    test_woe[woe_features],  test_woe["bad_flag"],
)

# 树模型无需 background_data
exp = ModelExplainer(gbm)
exp.explain(test_woe[woe_features])

# 全局特征重要性（mean |SHAP|，降序）
print(exp.feature_importance().head(10))
```

## 2. 解释逻辑回归（LRMaster）

线性模型需要传入 `background_data`（一份代表性的训练特征样本，作为 SHAP 的参照分布）：

```python
from Modeling_Tool import LRMaster, ModelExplainer

lr = LRMaster(params={"C": 1.0, "max_iter": 1000})
lr.fit(train_woe, woe_features, "bad_flag")

exp = ModelExplainer(lr, background_data=train_woe[woe_features])
exp.explain(test_woe[woe_features])
print(exp.feature_importance().head(10))
```

## 3. 全局特征重要性

```python
fi = exp.feature_importance(test_woe[woe_features])
#   feature        mean_abs_shap
#   feat_3_woe     0.42
#   feat_0_woe     0.31
#   ...

# 归一化为占比
fi_pct = exp.feature_importance(normalize=True)
print(fi_pct[["feature", "importance_pct"]])
```

!!! tip "X 可省略"
    调用过 `exp.explain(X)` 后，`feature_importance()` / `summary_plot()` / `dependence_plot()` 省略 `X` 时会复用缓存的 SHAP 值；传入新 `X` 则重新计算。

## 4. 可视化

```python
# summary（蜂群 / 条状）
exp.summary_plot(test_woe[woe_features], show=False, save_path="shap_summary.png")
exp.summary_plot(plot_type="bar", show=False, save_path="shap_bar.png")

# dependence（单特征依赖）
exp.dependence_plot("feat_3_woe", show=False, save_path="shap_dep.png")
```

所有绘图方法返回 `matplotlib.figure.Figure`，支持 `show=False`（headless）与 `save_path` 落盘。

## 5. 单样本归因

```python
contrib = exp.explain_instance(test_woe[woe_features].iloc[[0]])
print(contrib)
#   feature        value     shap_value
#   feat_3_woe     0.81      0.27
#   feat_0_woe    -0.40     -0.19
#   ...
print("base value:", contrib.attrs["base_value"])
```

返回的 DataFrame 按 |SHAP| 降序，SHAP 基值存于 `contrib.attrs['base_value']`。

## 6. 直接传入原生估计器

除了 SMF 包装类，`ModelExplainer` 也接受裸的已拟合估计器：

```python
from lightgbm import LGBMClassifier
from Modeling_Tool import ModelExplainer

clf = LGBMClassifier(n_estimators=200).fit(train_X, train_y)
exp = ModelExplainer(clf, model_type="lgb")          # 显式指定也可
print(exp.feature_importance(test_X).head())
```

## 常见问题

??? question "`ImportError: ModelExplainer requires the optional shap dependency`"

    未安装 shap。执行：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

??? question "`ValueError: Linear models require background_data`"

    线性 / model-agnostic 解释器需要参照分布。传入一份训练特征样本：

    ```python
    ModelExplainer(lr, background_data=train_woe[woe_features])
    ```

??? question "如何解释校准后的模型？"

    校准模型（`CalibratedClassifierCV`）不是树 / 线性结构，会走 **model-agnostic** 路径，同样需要 `background_data`：

    ```python
    ModelExplainer(calibrated_clf, background_data=train_X)
    ```

    该路径基于 `predict_proba` 计算，速度比 TreeExplainer 慢，建议 `background_data` 取样不要过大。
