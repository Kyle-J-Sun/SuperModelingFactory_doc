# 拒绝推断与分布适配

建模样本通常只来自**审批通过**的客户，会引入**选择偏差**。SuperModelingFactory 提供**拒绝推断**与**样本加权**两类工具来修正。

## 1. 拒绝推断 —— `RejectInferenceFactory`

### 工厂方法

```python
from Modeling_Tool import RejectInferenceFactory

inferrer = RejectInferenceFactory.create(
    method="parceling",
    target_col="bad_flag",
    score_col="prob",
)
df_combined = inferrer.infer(
    df_approved=approved_df,
    df_rejected=rejected_df,
    score_col="prob",
)
```

### 支持的推断方法

| 方法 | 适用场景 | 思路 |
|------|---------|------|
| `hard_cutoff` | 老模型评分阈值明确 | 拒绝样本中分数 > 阈值的标记为坏 |
| `fuzzy` | 阈值附近过渡 | 按分数分段给予不同的坏样本概率 |
| `parceling` | 分箱映射 | 把拒绝样本按分数分箱，按箱坏率赋标签 |
| `simple_augment` | 快速近似 | 拒绝样本按整体坏样本率赋标签 |

### 各类使用示例

=== "Parceling（推荐）"

    ```python
    from Modeling_Tool import ParcelingInferrer

    inferrer = ParcelingInferrer(target_col="bad_flag", score_col="prob", n_parcels=10)
    df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
    ```

=== "Hard Cutoff"

    ```python
    from Modeling_Tool import HardCutoffInferrer

    inferrer = HardCutoffInferrer(
        target_col="bad_flag",
        score_col="prob",
        cutoff=0.5,
    )
    df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
    ```

=== "Fuzzy Augment"

    ```python
    from Modeling_Tool import FuzzyAugmentInferrer

    inferrer = FuzzyAugmentInferrer(
        target_col="bad_flag",
        score_col="prob",
        weight_factor=0.8,
    )
    df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
    ```

=== "Simple Augment"

    ```python
    from Modeling_Tool import SimpleAugmentInferrer

    inferrer = SimpleAugmentInferrer(target_col="bad_flag", score_col="prob")
    df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
    ```

### 自定义推断器

继承 `RejectInferrer` 抽象类（`infer(self, df_approved, df_rejected, score_col=None)`）：

```python
import pandas as pd
from Modeling_Tool import RejectInferrer

class MyInferrer(RejectInferrer):
    def infer(self, df_approved, df_rejected, score_col=None):
        # 自定义逻辑
        score_col = score_col or self.score_col
        df_rejected = df_rejected.copy()
        df_rejected[self.target_col] = 0   # 默认全部好
        df_rejected.loc[df_rejected[score_col] > 0.3, self.target_col] = 1
        return pd.concat([df_approved, df_rejected], ignore_index=True)
```

## 2. 分布适配 —— `DistributionAdaptation`

用于**训练集与 OOT 分布不一致**的场景。

### 密度比估计

```python
from Modeling_Tool import DistributionAdaptation

adapter = DistributionAdaptation(method="density_ratio")
sample_weights = adapter.estimate_density_ratio(
    X_train=train_df[features],
    X_oot=oot_df[features],
)

# 在训练时使用 sample_weight
import lightgbm as lgb
dtrain = lgb.Dataset(train_X, train_y, weight=sample_weights)
```

### 协变量偏移修正

```python
adapter = DistributionAdaptation(method="covariate_shift")
weights = adapter.covariate_shift_weighting(train_X, oot_X)
```

!!! tip "fit / get_weights 接口"

    也可用统一的 `fit(X_train, X_oot)` + `get_weights()`：

    ```python
    adapter = DistributionAdaptation(method="density_ratio")
    adapter.fit(X_train=train_df[features], X_oot=oot_df[features])
    sample_weights = adapter.get_weights()
    ```

## 3. 完整流水线

```python
from Modeling_Tool import RejectInferenceFactory, DistributionAdaptation

# 1) 拒绝推断（先扩样本）
inferrer = RejectInferenceFactory.create("parceling", target_col="bad_flag", score_col="prob")
df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")

# 2) 分布适配（再加权）
adapter = DistributionAdaptation(method="density_ratio")
adapter.fit(X_train=df_combined[features], X_oot=oot_df[features])
sample_weights = adapter.get_weights()

# sample_weights 可作为下游训练器的样本权重（当前 SMF 无内置加权训练接口）。
```

## 常见问题

??? question "哪种拒绝推断方法最准确"

    学术上没有定论。**Parceling** 在多数实证研究中表现稳定，建议优先尝试；
    业务上一般配合 AB 测试验证。

??? question "推断后的样本要不要做 SMOTE"

    通常**不需要**。拒绝推断已经增加了少数类样本，再做 SMOTE 可能放大噪声。

??? question "OOT 样本量极小（< 1000）时如何评估加权效果"

    1. 把 OOT 拆成多段时间窗口分别评估
    2. 用 PSI 而非 AUC 做主要指标
