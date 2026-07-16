# 样本管理

样本管理是建模的第一步。SuperModelingFactory 在 [`Sample`](../api/sample.md) 子包中提供**切分 / 分层 / 均衡 / 最优种子搜索**四类工具。

## 1. 样本切分 —— `SampleSplitter`

```python
from Modeling_Tool import SampleSplitter

splitter = SampleSplitter(
    test_size=0.3,
    random_state=42,
    stratify=True,      # 按目标分层
)
train_df, test_df = splitter.split_df(data, target="bad_flag")
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `test_size` | `0.25` | 验证集占比（0–1） |
| `random_state` | `None` | 随机种子 |
| `stratify` | `True` | 是否按目标列分层 |

### 返回值

`(train_df, test_df)` —— pandas DataFrame 元组。

## 2. 分层采样 —— `StratifiedSampler`

保持坏样本率不变的随机采样，常用于**训练样本过大**时的下采样。

```python
from Modeling_Tool import StratifiedSampler

sampler = StratifiedSampler(random_state=42)
sample_df = sampler.sample(train_df, target="bad_flag", n_samples=5000)
```

## 3. 样本均衡 —— `StratifiedSampler.balance`

处理**正负样本极不平衡**的问题（如坏样本率 < 1%）。

```python
from Modeling_Tool import StratifiedSampler

# method：'undersample' / 'oversample' / 'smote'
sampler = StratifiedSampler(random_state=42)
balanced_df = sampler.balance(train_df, target="bad_flag", method="smote")
```

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `undersample` | 随机下采样多数类 | 样本充足，训练耗时敏感 |
| `oversample` | 随机上采样少数类 | 样本不足 |
| `smote` | SMOTE 合成少数类 | 样本极少，需保留分布信息（需 `imbalanced-learn`） |

!!! note "`SampleBalancer`（imblearn 风格欠采样器）"

    若需要 `random` / `nearmiss` / `tomek` / `enn` 等欠采样器，并返回 `(X, y)` 元组：

    ```python
    from Modeling_Tool import SampleBalancer
    X_res, y_res = SampleBalancer(method="nearmiss", random_state=42).fit_resample(X, y)
    ```

## 4. 最优种子搜索 —— `select_sample_seed`

固定训练/验证/OOT 划分后，搜索使 **OOT AUC 最大化** 的随机种子。注意 `model` 传入的是 **GradientBoostingModel 包装类**（而非底层估计器）：

```python
from Modeling_Tool import select_sample_seed, GradientBoostingModel

gbm = GradientBoostingModel("lgb", {"n_estimators": 100, "learning_rate": 0.1})

best_seed = select_sample_seed(
    master_df=df,
    oot_split_col="sample_ind",   # 1=INS, 2=OOT
    model=gbm,                    # 传包装类
    tgt_name="bad_flag",
    seed_range=(3000, 3050),      # 搜索范围
    ins_prop=0.7,
)
print(f"最优种子: {best_seed}")
```

!!! tip "何时需要搜索"

    - 训练集极小（< 1 万）时，结果对随机种子敏感
    - 希望最大化 OOT 表现而非训练集表现

## 5. 拒绝推断 —— `RejectInferenceFactory`

建模数据只来自**审批通过**的样本时使用。

```python
from Modeling_Tool import RejectInferenceFactory

inferrer = RejectInferenceFactory.create("parceling", target_col="bad_flag", score_col="prob")
df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
```

支持的推断方法详见 [拒绝推断与分布适配](reject_inference.md)。

## 常见问题

??? question "切分后训练集坏样本率与全量不一致"

    检查是否设置了 `stratify=True`。否则 `train_test_split` 会做纯随机切分，
    小样本下坏样本率会有 ±1% 的波动。

??? question "`StratifiedSampler.balance(method='smote')` 报 `ModuleNotFoundError`"

    安装可选依赖：

    ```bash
    pip install imbalanced-learn>=0.10.0
    ```

## 行级切分物化（0.6.7+，G01）

样本分析的推荐切分现在可以直接物化成行级归属，避免"统计口径与实际建模切分对不上"：

```python
SampleAnalysisPipelineConfig(
    materialize_split=True, id_col="loan_id",
    oot_cutoff="2025-04",          # 可选：覆盖推荐窗口，OOT = oot_time_dim >= cutoff
    split_col_name="sample_split", # 输出列名，可直接回灌 CMP/FVP 的 split_col
    persist_split_map=True,        # 落盘 row_level_split.csv + split_artifact.json
)
```

- 物化按推荐 (窗口, 比例, 种子) 经同一个 `SampleSplitter` 重放，保证与统计阶段完全同索引。
- 响亮断言：`id_col` 成熟行内唯一（重复即报计数）、三段两两互斥、覆盖全部成熟行。
- `split_artifact` 携带每段的 sha256 ID 哈希（ins/oos/oot/full），两次运行可直接比对哈希验证一致性。
