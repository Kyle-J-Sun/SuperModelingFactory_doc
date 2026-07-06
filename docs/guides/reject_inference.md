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

## 4. Pipeline 层：样本权重自动传递

从 **v0.3.16** 起，`RejectInferencePipeline` 内部会把每个 RI 方法生成的 `_weight`
列自动作为 `sample_weight` 传给下游 `GradientBoostingModel.fit`，无需用户显式配置：

| RI 方法 | 是否生成 `_weight` | 训练时是否加权 |
|---|---|---|
| `simple_augment` | ❌（每条拒绝样本被复制成 1 条 good，权重视为 1.0） | ❌ |
| `hard_cutoff` | ❌ | ❌ |
| `fuzzy_augment` | ✅（每条拒绝样本被复制成 good + bad 两行，权重分别为 `1-p(bad)` 和 `p(bad)`） | ✅ |
| `parceling` | ✅（按 score 分段随机赋标签，权重按段内 bad 率） | ✅ |

!!! warning "历史 bug（v0.3.15 及以前）"

    在 v0.3.15 及更早版本中，`RejectInferencePipeline` 将 `_weight` 以 `wgt=`
    传给 `GradientBoostingModel.fit`。由于 fit 的正确参数名是 `sample_weight`，
    `wgt` 会被 `**kwargs` 静默吞掉，**样本权重从未真正生效**。
    v0.3.16 已修复，`fuzzy_augment` / `parceling` 的加权训练现在按预期工作。

    如果你在 v0.3.15 或更早版本训练过 RI 后模型，建议在 v0.3.16 重新训练一次，
    对比 KS/AUC 差异确认权重是否显著影响模型。

## 5. Pipeline 层：输入 & 配置校验（v0.3.17）

v0.3.17 收紧了 `RejectInferencePipeline` 的 fail-fast 校验，把两类此前会在深层
代码里报错或静默数据泄漏的情况都抬到 `run()` 起始的 `_validate_input` /
`_validate_ri_approved_config` 阶段。

### 5.1 隐式 prescore 需要 `target_col`

即使 `train_prescore=False`，只要输入 DataFrame 中缺少 `score_col`（默认
`"prescore_prob"`），Pipeline 仍会走一次隐式 prescore 训练（`_run` 内部逻辑：
`if cfg.train_prescore or cfg.score_col not in data.columns: _fit_prescore(...)`）。

历史版本上，`_validate_input` 只在 `train_prescore=True` 时才检查 `target_col`
是否存在，所以上述隐式路径下缺 target 会在 `_fit_prescore` 里报一个位置奇怪的
`KeyError`。v0.3.17 起，只要 Pipeline 判断出**任何一条 prescore 训练路径会被触发**，
`_validate_input` 就在最外层显式 `raise KeyError`，报错信息包含 `train_prescore`
和 `score_col` 的实际状态，便于定位。

对应的合法组合：

| `train_prescore` | `score_col` 是否在数据中 | `target_col` 是否必需 |
|---|---|---|
| `True` | 有或无 | ✅ 必需 |
| `False` | ✅ 已提供 | ❌ 可选（Pipeline 直接使用外部分数） |
| `False` | ❌ 缺失（隐式训练） | ✅ 必需（v0.3.17 起显式要求） |

### 5.2 `oot_frac + ri_validation_frac` 必须 `< 1.0`

`RejectInferencePipeline._sample_validation_ids` 会在按 `oot_frac` 抽出 OOT 之后，
从剩余样本里再按 `ri_validation_frac` 抽验证集。如果两者之和 ≥ 1，剩余池会为空，
历史版本会静默 fallback 到 `pool = approved.copy()`，验证集因此会与 OOT 有交集，
**造成 OOT 行泄漏进验证指标**。

v0.3.17 起，`_validate_ri_approved_config` 会在 `run()` 起始直接抛
`ValueError`：

```python
cfg = RejectInferencePipelineConfig(oot_frac=0.5, ri_validation_frac=0.5)
RejectInferencePipeline(cfg).run(df)
# ValueError: oot_frac (0.5) + ri_validation_frac (0.5) = 1.000 must be < 1.0
# to leave a non-empty training pool disjoint from OOT and validation
```

同时新增了单独的边界约束：`oot_frac` 必须在 `[0, 1)` 内（`0` 允许，表示禁用
OOT 切分）。

## 常见问题

??? question "哪种拒绝推断方法最准确"

    学术上没有定论。**Parceling** 在多数实证研究中表现稳定，建议优先尝试；
    业务上一般配合 AB 测试验证。

??? question "推断后的样本要不要做 SMOTE"

    通常**不需要**。拒绝推断已经增加了少数类样本，再做 SMOTE 可能放大噪声。

??? question "OOT 样本量极小（< 1000）时如何评估加权效果"

    1. 把 OOT 拆成多段时间窗口分别评估
    2. 用 PSI 而非 AUC 做主要指标
