# Sample Management

Sample management is the first step of modeling. SuperModelingFactory provides four kinds of tools in the [`Sample`](../api/sample.md) subpackage: **splitting / stratification / balancing / optimal seed search**.

## 1. Sample Splitting — `SampleSplitter`

```python
from Modeling_Tool import SampleSplitter

splitter = SampleSplitter(
    test_size=0.3,
    random_state=42,
    stratify=True,      # stratify by the target
)
train_df, test_df = splitter.split_df(data, target="bad_flag")
```

### Key Parameters

| Parameter | Default | Description |
|------|-------|------|
| `test_size` | `0.25` | Share of the validation set (0–1) |
| `random_state` | `None` | Random seed |
| `stratify` | `True` | Whether to stratify by the target column |

### Return Value

`(train_df, test_df)` — a tuple of pandas DataFrames.

## 2. Stratified Sampling — `StratifiedSampler`

Random sampling that preserves the bad rate, commonly used to downsample when the **training sample is too large**.

```python
from Modeling_Tool import StratifiedSampler

sampler = StratifiedSampler(random_state=42)
sample_df = sampler.sample(train_df, target="bad_flag", n_samples=5000)
```

## 3. Sample Balancing — `StratifiedSampler.balance`

Handles **severely imbalanced positive/negative samples** (for example, a bad rate below 1%).

```python
from Modeling_Tool import StratifiedSampler

# method: 'undersample' / 'oversample' / 'smote'
sampler = StratifiedSampler(random_state=42)
balanced_df = sampler.balance(train_df, target="bad_flag", method="smote")
```

| Mode | Description | When to use |
|------|------|---------|
| `undersample` | Randomly undersample the majority class | Plenty of samples, training time is a concern |
| `oversample` | Randomly oversample the minority class | Not enough samples |
| `smote` | SMOTE synthesizes minority-class samples | Very few samples, distribution information must be preserved (requires `imbalanced-learn`) |

!!! note "`SampleBalancer` (imblearn-style undersampler)"

    If you need undersamplers such as `random` / `nearmiss` / `tomek` / `enn` that return an `(X, y)` tuple:

    ```python
    from Modeling_Tool import SampleBalancer
    X_res, y_res = SampleBalancer(method="nearmiss", random_state=42).fit_resample(X, y)
    ```

## 4. Optimal Seed Search — `select_sample_seed`

With the train/validation/OOT split fixed, search for the random seed that **maximizes OOT AUC**. Note that `model` takes the **GradientBoostingModel wrapper class** (not the underlying estimator):

```python
from Modeling_Tool import select_sample_seed, GradientBoostingModel

gbm = GradientBoostingModel("lgb", {"n_estimators": 100, "learning_rate": 0.1})

best_seed = select_sample_seed(
    master_df=df,
    oot_split_col="sample_ind",   # 1=INS, 2=OOT
    model=gbm,                    # pass the wrapper class
    tgt_name="bad_flag",
    seed_range=(3000, 3050),      # search range
    ins_prop=0.7,
)
print(f"Best seed: {best_seed}")
```

!!! tip "When to search"

    - With a very small training set (< 10k), results are sensitive to the random seed
    - You want to maximize OOT performance rather than training-set performance

## 5. Reject Inference — `RejectInferenceFactory`

Use this when the modeling data comes only from **approved** samples.

```python
from Modeling_Tool import RejectInferenceFactory

inferrer = RejectInferenceFactory.create("parceling", target_col="bad_flag", score_col="prob")
df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
```

For the supported inference methods, see [Reject Inference and Distribution Adaptation](reject_inference.md).

## FAQ

??? question "The training-set bad rate after splitting differs from the full-sample bad rate"

    Check whether `stratify=True` is set. Otherwise `train_test_split` does a purely random split,
    and with small samples the bad rate can fluctuate by ±1%.

??? question "`StratifiedSampler.balance(method='smote')` raises `ModuleNotFoundError`"

    Install the optional dependency:

    ```bash
    pip install imbalanced-learn>=0.10.0
    ```

## Row-Level Split Materialization (0.6.7+, G01)

The recommended split from sample analysis can now be materialized directly into row-level assignments, avoiding a mismatch between the statistics basis and the actual modeling split:

```python
SampleAnalysisPipelineConfig(
    materialize_split=True, id_col="loan_id",
    oot_cutoff="2025-04",          # optional: override the recommended window, OOT = oot_time_dim >= cutoff
    split_col_name="sample_split", # output column name, can be fed straight back into the CMP/FVP split_col
    persist_split_map=True,        # write row_level_split.csv + split_artifact.json to disk
)
```

- Materialization replays the recommended (window, ratio, seed) through the same `SampleSplitter`, guaranteeing identical indices to the statistics stage.
- Loud assertions: `id_col` must be unique among matured rows (duplicates are reported with counts), the three segments must be pairwise disjoint, and they must cover all matured rows.
- `split_artifact` carries a sha256 ID hash for each segment (ins/oos/oot/full), so two runs can be compared directly by hash to verify consistency.
