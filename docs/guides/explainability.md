# 模型解释

SuperModelingFactory 在 [`Explainability`](../api/explainability.md) 子包提供统一解释器 `ModelExplainer`。它不仅支持 SHAP 归因，也支持 PDP、ICE、ALE 和 LIME，用同一个入口解释 SMF 模型或原生 sklearn / LightGBM / XGBoost 模型。

!!! note "安装可选依赖"

    SHAP 与 LIME 是可选依赖，需单独安装：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

    `ModelExplainer` 采用懒加载：`from Modeling_Tool import ModelExplainer` 不会立即导入 `shap` 或 `lime`，只有调用对应方法时才加载依赖。

## 方法对比

| 方法 | 类型 | 适合回答的问题 | 依赖 |
|------|------|----------------|------|
| SHAP | 全局 + 局部归因 | 哪些变量贡献最大？单样本为什么高风险？ | `shap` |
| PDP | 全局平均效应 | 某变量上升时，平均预测风险怎么变？ | 内置 |
| ICE | 个体响应曲线 | 不同样本对同一变量的响应是否一致？ | 内置 |
| ALE | 累计局部效应 | 特征相关性较强时，如何更稳健地看边际影响？ | 内置 |
| LIME | 局部代理模型 | 单样本附近，局部线性解释是什么？ | `lime` |

## 1. 创建解释器

```python
from Modeling_Tool import GradientBoostingModel, ModelExplainer

gbm = GradientBoostingModel("lgb", {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "early_stopping_rounds": 30,
    "eval_metric": "auc",
})
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],
    test_woe[woe_features], test_woe["bad_flag"],
)

exp = ModelExplainer(
    gbm,
    background_data=train_woe[woe_features],  # SHAP linear/generic 与 LIME 可复用
)
```

树模型做 SHAP 时可以不传 `background_data`；但 LIME、线性 SHAP 或 model-agnostic SHAP 需要代表性训练样本作为参照分布。

## 2. SHAP 归因

```python
exp.explain(test_woe[woe_features])

# 全局特征重要性：mean |SHAP|
fi = exp.feature_importance(normalize=True)
print(fi[["feature", "mean_abs_shap", "importance_pct"]].head(10))

# 可视化
exp.summary_plot(show=False, save_path="shap_summary.png")
exp.summary_plot(plot_type="bar", show=False, save_path="shap_bar.png")
exp.dependence_plot("age_woe", show=False, save_path="shap_dep_age.png")

# 单样本归因
contrib = exp.explain_instance(test_woe[woe_features].iloc[[0]])
print(contrib)
print("base value:", contrib.attrs["base_value"])
```

## 3. PDP：平均边际影响

PDP（Partial Dependence Plot）固定某个特征到一组网格值，观察模型平均预测如何变化。

```python
pdp = exp.partial_dependence(
    X=test_woe[woe_features],
    feature="age_woe",
    grid_resolution=50,
    sample_size=2000,
    random_state=42,
)
print(pdp.head())
# feature / grid_value / average_prediction

exp.pdp_plot(
    test_woe[woe_features],
    feature="age_woe",
    show=False,
    save_path="pdp_age.png",
)
```

## 4. ICE：个体响应曲线

ICE（Individual Conditional Expectation）保留每个样本自己的曲线，适合观察人群异质性。

```python
ice = exp.ice(
    X=test_woe[woe_features],
    feature="age_woe",
    grid_resolution=50,
    sample_size=200,
    random_state=42,
)
print(ice.head())
# feature / sample_index / grid_value / prediction

exp.ice_plot(
    test_woe[woe_features],
    feature="age_woe",
    centered=True,
    show=False,
    save_path="ice_age.png",
)
```

`centered=True` 会把每条 ICE 曲线平移到同一起点，更容易看斜率差异。

## 5. ALE：累计局部效应

ALE（Accumulated Local Effects）按特征分位数切箱，只在每个箱内做局部扰动。相比 PDP，它在强相关特征场景下通常更稳健。

```python
ale = exp.ale(
    X=test_woe[woe_features],
    feature="income_woe",
    bins=20,
)
print(ale.head())
# feature / bin_left / bin_right / bin_center / ale_value / n

exp.ale_plot(
    test_woe[woe_features],
    feature="income_woe",
    bins=20,
    show=False,
    save_path="ale_income.png",
)
```

当前 ALE 支持数值型单特征。对于类别变量，建议先使用 WOE 编码后的数值列解释。

## 6. LIME：局部代理解释

LIME 为单样本附近拟合局部代理模型，输出局部线性权重。

```python
lime_one = exp.lime_explain_instance(
    x_row=test_woe[woe_features].iloc[0],
    X_train=train_woe[woe_features],
    num_features=10,
    num_samples=5000,
    random_state=42,
)
print(lime_one)
# feature / feature_rule / weight / abs_weight
print("local score:", lime_one.attrs["score"])
```

也可以对一批样本聚合 LIME 权重，得到近似全局重要性：

```python
lime_global = exp.lime_global_importance(
    X=test_woe[woe_features].sample(100, random_state=42),
    X_train=train_woe[woe_features],
    num_features=10,
    num_samples=2000,
)
print(lime_global)
# feature / mean_abs_lime_weight / frequency
```

!!! warning "性能建议"

    PDP / ICE / ALE / LIME 都会反复调用模型预测。生产数据较大时，建议使用 `sample_size` 抽样解释，不要直接全量运行。

## 7. 直接传入原生估计器

```python
from sklearn.linear_model import LogisticRegression
from Modeling_Tool import ModelExplainer

clf = LogisticRegression(max_iter=1000).fit(train_X, train_y)
exp = ModelExplainer(clf, background_data=train_X)

print(exp.partial_dependence(test_X, feature="age").head())
print(exp.ale(test_X, feature="income").head())
```

## 常见问题

??? question "`PDD` 是不是 `PDP`？"

    是的，通常叫 `PDP`，全称 Partial Dependence Plot。

??? question "PDP 和 ALE 怎么选？"

    如果特征之间相关性不强，PDP 直观易懂；如果特征相关性明显，ALE 通常更稳健，因为它只在局部区间内扰动特征。

??? question "LIME 是全局解释吗？"

    LIME 本质是局部解释。`lime_global_importance()` 是对多个局部解释的聚合，只能作为近似全局重要性参考。

??? question "为什么 LIME / SHAP 报缺依赖？"

    需要安装 explain extra：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```
