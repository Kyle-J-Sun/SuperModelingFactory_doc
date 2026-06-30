# 模型解释

SuperModelingFactory 在 [`Explainability`](../api/explainability.md) 子包提供统一解释器 `ModelExplainer`。它支持 SHAP、Owen Value、PDP、ICE、ALE 和 LIME，用同一个入口解释 SMF 模型或原生 sklearn / LightGBM / XGBoost 模型。

!!! note "安装可选依赖"

    SHAP、Owen Value 与 LIME 需要可选解释依赖：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

    `ModelExplainer` 采用懒加载：`from Modeling_Tool import ModelExplainer` 不会立即导入 `shap` 或 `lime`，只有调用对应方法时才加载依赖。

## 方法对比

| 方法 | 类型 | 适合回答的问题 | 依赖 |
|------|------|----------------|------|
| SHAP | 全局 + 局部归因 | 哪些变量贡献最大？单样本为什么高风险？ | `shap` |
| Owen Value | 分组归因 | 逾期、多头、负债等业务模块整体贡献是多少？ | `shap` |
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
    background_data=train_woe[woe_features].sample(1000, random_state=42),
)
```

## 2. SHAP 归因

```python
exp.explain(test_woe[woe_features])

fi = exp.feature_importance(normalize=True)
print(fi[["feature", "mean_abs_shap", "importance_pct"]].head(10))

exp.summary_plot(show=False, save_path="shap_summary.png")
exp.summary_plot(plot_type="bar", show=False, save_path="shap_bar.png")
exp.dependence_plot("age_woe", show=False, save_path="shap_dep_age.png")

contrib = exp.explain_instance(test_woe[woe_features].iloc[[0]])
print(contrib)
print("base value:", contrib.attrs["base_value"])
```

## 3. Owen Value：分组归因

普通 SHAP 把每个特征视为独立玩家。强相关变量会分摊同一个风险信号，例如多个逾期变量的单特征贡献都显得“不够突出”。Owen Value 先把特征放进 coalition，再在 coalition 内部分摊贡献，因此更适合输出业务模块级 reason code。

### 3.1 构建 Coalition Structure

```python
from Modeling_Tool import build_coalition_structure

prior_groups = {
    "delinquency": ["max_dpd_12m_woe", "dpd_cnt_6m_woe", "ever_dpd30_woe"],
    "multi_lending": ["inquiries_3m_woe", "inquiries_6m_woe", "active_loans_woe"],
    "affordability": ["monthly_income_woe", "debt_to_income_woe", "monthly_obligation_woe"],
}

cs = build_coalition_structure(
    train_woe[woe_features],
    prior_groups=prior_groups,
    threshold=0.35,
    method="complete",
    corr_method="spearman",
)

print(cs["summary"][["n_features", "mean_abs_corr", "max_abs_corr"]])
```

融合逻辑是：业务先验优先，其余特征使用 Spearman 相关距离的自动聚类兜底。`prior_groups` 可以包含当前数据中不存在的变量名，工具会自动忽略；但同一个有效特征不能同时出现在多个先验组里。

#### 使用 MIC 捕捉非线性关联

默认 `corr_method="spearman"` 基于秩相关，适合单调关系。若变量之间存在明显非线性关联（例如 `x` 与 `x²`），可改用 **Maximal Information Coefficient (MIC)**：

```python
cs = build_coalition_structure(
    train_woe[woe_features],
    threshold=0.35,
    corr_method="MIC",
)
```

!!! note "MIC 可选依赖"

    MIC 需要额外安装 `minepy`（GPLv3，含 C 扩展）：

    ```bash
    pip install 'supermodelingfactory[mic]'
    ```

    - `corr_method="MIC"` 大小写不敏感（`"mic"` 亦可）。
    - `threshold` 仍作用于 `1 - MIC` 距离：MIC 越高，特征越容易分到同一组。
    - `summary` 中的 `mean_abs_corr` / `max_abs_corr` 在 MIC 模式下表示组内平均/最大 MIC。
    - MIC 需对所有特征两两计算，复杂度约为特征对数量级，通常比 Spearman 慢得多。
    - `minepy` 目前在 Python 3.11+ 上存在已知构建问题；若安装失败，请使用 Python 3.10 环境，或回退 `corr_method="spearman"`。

### 3.2 通过 ModelExplainer 计算 Owen Value

```python
owen_exp = exp.explain_owen(
    X=test_woe[woe_features],
    coalition_structure=cs,
    model_output="log_odds",
    max_evals=500,
)

# 组级全局重要性
owen_group = exp.owen_group_importance(normalize=True)
print(owen_group[["group", "n_features", "mean_abs_owen", "importance_pct"]])

# 单样本模块级 reason code
local_reason = exp.owen_explain_instance(test_woe[woe_features].iloc[0])
print(local_reason[["group", "owen_value", "abs_owen_value", "features"]])
```

`model_output="log_odds"` 适合信贷 reason code，因为组贡献可以直接解释为对 log-odds 的加减影响；如果希望解释正类概率，使用默认的 `model_output="probability"`。

### 3.3 一步式调用

如果不需要单独检查分组，也可以直接在 `explain_owen()` 里传入先验分组：

```python
exp.explain_owen(
    test_woe[woe_features],
    prior_groups=prior_groups,
    threshold=0.35,
    model_output="log_odds",
)
print(exp.owen_group_importance().head())
```

## 4. PDP：平均边际影响

```python
pdp = exp.partial_dependence(
    X=test_woe[woe_features],
    feature="age_woe",
    grid_resolution=50,
    sample_size=2000,
    random_state=42,
)
print(pdp.head())

exp.pdp_plot(test_woe[woe_features], feature="age_woe", show=False, save_path="pdp_age.png")
```

## 5. ICE：个体响应曲线

```python
ice = exp.ice(
    X=test_woe[woe_features],
    feature="age_woe",
    grid_resolution=50,
    sample_size=200,
    random_state=42,
)
print(ice.head())

exp.ice_plot(test_woe[woe_features], feature="age_woe", centered=True, show=False, save_path="ice_age.png")
```

## 6. ALE：累计局部效应

```python
ale = exp.ale(X=test_woe[woe_features], feature="income_woe", bins=20)
print(ale.head())

exp.ale_plot(test_woe[woe_features], feature="income_woe", bins=20, show=False, save_path="ale_income.png")
```

当前 ALE 支持数值型单特征。对于类别变量，建议先使用 WOE 编码后的数值列解释。

## 7. LIME：局部代理解释

```python
lime_one = exp.lime_explain_instance(
    x_row=test_woe[woe_features].iloc[0],
    X_train=train_woe[woe_features],
    num_features=10,
    num_samples=5000,
    random_state=42,
)
print(lime_one)

lime_global = exp.lime_global_importance(
    X=test_woe[woe_features].sample(100, random_state=42),
    X_train=train_woe[woe_features],
    num_features=10,
    num_samples=2000,
)
print(lime_global)
```

!!! warning "性能建议"

    PDP / ICE / ALE / LIME / Owen Value 都会反复调用模型预测。生产数据较大时，建议使用 `sample_size`、`background_data.sample(...)` 或 `max_evals` 控制解释成本。

## 常见问题

??? question "Owen Value 和 SHAP 有什么关系？"

    Owen Value 是带 coalition structure 的 Shapley 分配。实现上使用 `shap.PartitionExplainer`，先尊重特征分组，再在组内分摊贡献。

??? question "什么时候用 Owen Value？"

    当变量高度相关，或者业务需要模块级 reason code 时使用。例如逾期、多头、负债能力、设备欺诈等同源变量，单特征 SHAP 容易归因稀释。

??? question "threshold=0.35 是什么意思？"

    使用距离 `1 - abs(association)` 聚类。默认 `corr_method="spearman"` 时，`0.35` 约等价于 `|corr| > 0.65` 的特征更容易先聚到同一组。若使用 `corr_method="MIC"`，同一阈值作用于 `1 - MIC`。强相关场景可用 `0.20~0.35`，更宽松可用 `0.50`。

??? question "什么时候用 corr_method='MIC'？"

    当变量之间存在明显非线性关联、Spearman 难以把它们聚到同一组时。MIC 更慢，且需要单独安装 `pip install 'supermodelingfactory[mic]'`；Python 3.11+ 若 `minepy` 安装失败，请使用 Python 3.10 或继续用 `spearman`。

??? question "PDP 和 ALE 怎么选？"

    如果特征之间相关性不强，PDP 直观易懂；如果特征相关性明显，ALE 通常更稳健，因为它只在局部区间内扰动特征。

??? question "LIME 是全局解释吗？"

    LIME 本质是局部解释。`lime_global_importance()` 是对多个局部解释的聚合，只能作为近似全局重要性参考。

??? question "为什么 LIME / SHAP / Owen 报缺依赖？"

    需要安装 explain extra：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

??? question "XGBoost + SHAP 报 `could not convert string to float: '[5E-1]'` 怎么办？"

    这是 XGBoost 3.1+ 与旧版 SHAP 的已知兼容问题。XGBoost 会把单目标模型的 `base_score` 序列化为单元素数组字符串，例如 `"[5E-1]"`，而 SHAP 0.49 及更早版本仍按标量解析。

    SuperModelingFactory 会在 `ModelExplainer` 构建 XGBoost `TreeExplainer` 时自动启用兼容层：

    - 在 SHAP 的 UBJSON 解码阶段把单元素 `base_score` 归一为标量。
    - 为 `XGBTreeModelLoader` 增加 XGBoost 专属 fallback，并在失败时重试一次。
    - 仅作用于 `model_type in {"xgb", "xgboost"}` 的 TreeSHAP 路径，不影响 LightGBM、LR、Owen、PDP/ICE/ALE/LIME。

    推荐顺序：

    1. 安装最新 SuperModelingFactory 主分支或最新 release。
    2. Python 3.11+ 环境可升级到 `shap>=0.50.0` 获得上游原生支持。
    3. 仅当多分类/多目标模型仍失败时，再考虑临时固定 `xgboost<3.1` 作为最后兜底。

    注意：SMF 不会把多元素 `base_score` 向量静默压成第一个元素，避免多目标 attribution 被错误解释。
