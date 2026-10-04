# Sample Management

The [`Modeling_Tool.Sample`](../api/sample.md) subpackage covers the first modeling step: turning raw data into samples you
can model on. It splits a frame into training and test samples, draws and rebalances samples, shows how much a split depends
on its random seed, and adds rejected applications to an approved-only sample.

| I want to... | Use | Import |
|---|---|---|
| Split into train and test samples with matching bad rates | `SampleSplitter` | `from Modeling_Tool import SampleSplitter` |
| Draw a random subsample | `StratifiedSampler.sample` | `from Modeling_Tool import StratifiedSampler` |
| Rebalance the classes (random under/over-sampling, SMOTE) | `StratifiedSampler.balance` | `from Modeling_Tool import StratifiedSampler` |
| Undersample with NearMiss, Tomek links, or ENN | `SampleBalancer` | `from Modeling_Tool import SampleBalancer` |
| See how split results change with the random seed | `select_sample_seed` | `from Modeling_Tool import select_sample_seed` |
| Add rejected applications to an approved-only sample | `RejectInferenceFactory` | `from Modeling_Tool import RejectInferenceFactory` |
| Label every row INS, OOS, or OOT from a sample analysis | `SampleAnalysisPipeline` | `from Modeling_Tool import SampleAnalysisPipeline, SampleAnalysisPipelineConfig` |

SMOTE and the `nearmiss`, `tomek`, and `enn` balancers need the optional `imbalanced-learn` package:

```bash
pip install 'supermodelingfactory[imblearn]'
```

## Example data

Every snippet on this page runs top to bottom in one Python session. The setup below creates a synthetic loan sample with
a rare bad outcome (about 5%), a month, a channel, and a unique loan id.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 6000
months = [f"2025-{m:02d}" for m in range(1, 10)]

data = pd.DataFrame({
    "loan_id":     [f"L{i:06d}" for i in range(n)],
    "apply_month": rng.choice(months, n),
    "channel":     rng.choice(["online", "branch", "partner"], n),
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
})
data["apply_time"] = pd.to_datetime(data["apply_month"] + "-15")

logit = -3.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"] - 0.8 * data["utilization"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "utilization", "n_overdue"]
print(f"rows={len(data)}  bad rate={data['bad_flag'].mean():.3f}")
```

## 1. Sample Splitting: `SampleSplitter`

`SampleSplitter` splits rows at random into a training and a test sample. With `stratify=True` (the default) both samples
keep the bad rate of the input. It never looks at dates, so to keep an out-of-time (OOT) sample, cut the frame by time first
and split only the earlier part.

```python
from Modeling_Tool import SampleSplitter

dev = data[data["apply_month"] < "2025-08"]        # development window
oot = data[data["apply_month"] >= "2025-08"]       # out-of-time window, never split

splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
train_df, test_df = splitter.split_df(dev, target="bad_flag")
print(len(train_df), len(test_df), round(train_df["bad_flag"].mean(), 4), round(test_df["bad_flag"].mean(), 4))
```

| Parameter | Default | Meaning |
|---|---|---|
| `test_size` | `0.3` | Share of the rows that go to the test sample (0 to 1) |
| `random_state` | `None` | Random seed. Set it to get the same split on every run |
| `stratify` | `True` | Keep the target's bad rate equal in both samples |

`split_df(df, target, exclude_cols=None, test_size=None)` returns `(train_df, test_df)`. Both are copies that keep every
column and the original index labels, in shuffled row order. `test_size` overrides the constructor value for this call.
`exclude_cols` is accepted for backward compatibility and has no effect, because all columns are always kept.

Two lower-level methods work on arrays and indices instead of frames:

| Method | Returns |
|---|---|
| `split(X, y, test_size=None, stratify=None)` | `(X_train, X_test, y_train, y_test)`, as `sklearn.model_selection.train_test_split` does |
| `split_indices(index, y, test_size=None, stratify=None)` | `(train_index, test_index)` as NumPy arrays. The last result is also kept in `train_index_` and `test_index_` |

## 2. Random Subsampling: `StratifiedSampler.sample`

Use it to downsample a **training sample that is too large**. Pass either `n_samples` (number of rows) or `sample_frac`
(fraction of rows). If both are given, `sample_frac` wins. With neither, a copy of the frame is returned.

```python
from Modeling_Tool import StratifiedSampler

sampler = StratifiedSampler(random_state=42)
sample_df = sampler.sample(train_df, target="bad_flag", n_samples=1000)
subset_df = sampler.sample(train_df, target="bad_flag", sample_frac=0.2)
print(len(sample_df), len(subset_df))
```

!!! note "Not stratified"

    Despite the class name, `sample` is a plain `DataFrame.sample` (without replacement). It keeps the bad rate only in
    expectation and ignores `target`. To control the class ratio exactly, use `balance` below.

## 3. Class Balancing

Balancing changes the share of bad rows in a sample, which helps when bads are very rare. Balance **only the training
sample**. Keep the validation, test, and OOT samples at their natural bad rate, because probabilities from a model trained
on a balanced sample are overstated.

### `StratifiedSampler.balance`

`StratifiedSampler(target_rate=None, random_state=None)` holds the settings, and
`balance(df, target, method="undersample")` returns the balanced frame. The input frame is not modified, and the input bad
rate is stored in `original_rate_`.

```python
numeric_df = train_df[features + ["bad_flag"]]          # SMOTE needs numeric columns without NaN

for method in ["undersample", "oversample", "smote"]:
    balanced = StratifiedSampler(random_state=42).balance(numeric_df, target="bad_flag", method=method)
    print(f"{method:12s} rows={len(balanced):5d}  bad rate={balanced['bad_flag'].mean():.3f}")

# Aim for a 20% bad rate instead of 50% (honored by undersample and oversample only)
to_20pct = StratifiedSampler(target_rate=0.20, random_state=42).balance(numeric_df, "bad_flag", method="undersample")
print(len(to_20pct), round(to_20pct["bad_flag"].mean(), 3))
```

| `method` | What it does | `target_rate` |
|---|---|---|
| `"undersample"` (default) | Keeps every bad row and randomly drops good rows | Honored: keeps `bads * (1 - rate) / rate` good rows, at most all of them |
| `"oversample"` | Draws bad rows with replacement until their count matches the good rows. Duplicated rows keep their index label | Honored: draws `goods * rate / (1 - rate)` bad rows |
| `"smote"` | Synthesizes new bad rows with SMOTE | **Ignored**: the result is always 50/50 |

Without `target_rate`, the result is balanced 50/50.

!!! warning "Behavior to know"

    - Set `target_rate` **above** the current bad rate. Below it, `undersample` returns the frame unchanged and
      `oversample` resamples the bad class down.
    - `smote` raises `ValueError` if any non-target column is non-numeric or contains NaN. It returns a new frame with a
      fresh `RangeIndex`, float feature columns, and the target as the last column. With fewer than 6 bad rows it silently
      falls back to random oversampling.

### `SampleBalancer`

`SampleBalancer(method="random", target_ratio=None, random_state=None)` returns an `(X, y)` pair instead of a frame, like
an `imbalanced-learn` sampler: `fit_resample(X, y)` gives `(X_resampled, y_resampled)`. Pandas input comes back as a
DataFrame and a Series, and NumPy input as arrays.

```python
from Modeling_Tool import SampleBalancer

X, y = numeric_df[features], numeric_df["bad_flag"]
for method in ["random", "nearmiss", "tomek", "enn"]:
    X_res, y_res = SampleBalancer(method=method, random_state=42).fit_resample(X, y)
    print(f"{method:9s} rows={len(X_res):5d}  bad rate={y_res.mean():.3f}")
```

| `method` | What it does | Needs `imbalanced-learn` |
|---|---|---|
| `"random"` (default) | Randomly drops good rows until there are as many goods as bads (50/50) | No |
| `"nearmiss"` | NearMiss (version 1) undersampling of the good class (50/50) | Yes |
| `"tomek"` | Removes good rows that form Tomek links. The classes stay imbalanced | Yes |
| `"enn"` | Removes good rows that their nearest neighbors disagree with (Edited Nearest Neighbours). The classes stay imbalanced | Yes |

The `nearmiss`, `tomek`, and `enn` methods need numeric features without NaN. `random` keeps the original index labels, and
the other three return a new `RangeIndex`.

!!! warning "`target_ratio` does not work in 0.8.2"

    Only `method="random"` reads `target_ratio`, and it does not produce the documented ratio. On a sample with 5% bads,
    `target_ratio=0.5` returns every row unchanged, and `target_ratio=0.2` raises a `ValueError` ("Cannot take a larger sample
    than population"). Leave it unset, or use `StratifiedSampler(target_rate=...)` to reach a target bad rate.

!!! note "Global random state"

    `method="random"` calls `numpy.random.seed(random_state)`, so it resets NumPy's global random generator. With
    `random_state=None` it reseeds from the operating system.

## 4. Seed Sensitivity: `select_sample_seed`

`select_sample_seed(master_df, oot_split_col, model, tgt_name, seed_range=(3000, 3050), ins_prop=0.7)` re-splits the
development rows once per seed and scores each split with a model you have already fitted. It does not choose a seed for
you. It returns a **DataFrame** with one row per seed and dataset, and you pick the seed from it.

| Argument | Meaning |
|---|---|
| `master_df` | All rows. It must contain the target, the split flag, and the model's input columns (or the score column) |
| `oot_split_col` | Flag column: `1` marks development rows, which are re-split by seed, and `2` marks OOT rows. Other values are ignored |
| `model` | A fitted model with `predict_proba` (`GradientBoostingModel` or a scikit-learn estimator fitted on a DataFrame), or the **name of an existing score column** |
| `tgt_name` | Target column |
| `seed_range` | `(start, stop)`: the seeds `start` to `stop - 1` |
| `ins_prop` | Share of the development rows that go to `train` (INS). The rest go to `validation` (OOS) |

For each seed the development rows are split with a stratified `SampleSplitter`, and the result is the
[`PerformanceEvaluator`](eval.md) table for `train`, `validation`, and `oot` (the `index` column), plus a `seed` column.
A `tqdm` progress bar is shown while it runs. If no row has flag `2`, the `oot` rows are left out.

```python
from Modeling_Tool import GradientBoostingModel, select_sample_seed

gbm = GradientBoostingModel(
    "lgb",
    params={"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
            "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1},
)
gbm.fit(train_df[features], train_df["bad_flag"], test_df[features], test_df["bad_flag"])

# 1 = development rows (re-split for every seed), 2 = out-of-time rows
master_df = pd.concat([dev.assign(sample_ind=1), oot.assign(sample_ind=2)])

seed_perf = select_sample_seed(
    master_df=master_df,
    oot_split_col="sample_ind",
    model=gbm,
    tgt_name="bad_flag",
    seed_range=(3000, 3010),      # seeds 3000 .. 3009
    ins_prop=0.7,
)

# Pick the seed whose INS and OOS samples look most alike
ks = seed_perf.pivot(index="seed", columns="index", values="KS")
ks["gap"] = (ks["train"] - ks["validation"]).abs()
print(ks.sort_values("gap").head(3))
best_seed = int(ks["gap"].idxmin())
print("best seed:", best_seed)
```

!!! warning "What the table can and cannot tell you"

    - The model is **not refitted** per seed, and the OOT rows never change, so the `oot` metrics are identical for every
      seed. A seed cannot improve OOT performance. Rank seeds on `train` versus `validation`, as above.
    - The model has usually seen some of the development rows already, so both samples can be optimistic. Treat the table as
      a measure of how sensitive the metrics are to the split, then refit on the split you choose.
    - The smaller the development sample, the more the metrics move from seed to seed.

## 5. Reject Inference: `RejectInferenceFactory`

Use it when the modeling data comes only from **approved** applications, so the outcome of rejected ones is unknown.
`RejectInferenceFactory.create(method, **kwargs)` builds an inferrer, and `infer(approved_df, rejected_df, score_col)`
returns the approved and rejected rows stacked in one frame.

```python
from sklearn.linear_model import LogisticRegression
from Modeling_Tool import RejectInferenceFactory

# Approved applications have a label. Rejected ones do not
approved = (data["score_b"] + np.random.default_rng(1).normal(0, 30, n)) > 590
approved_df = data[approved].copy()
rejected_df = data[~approved].drop(columns="bad_flag")

# Score both groups with a model trained on the approved sample
scorer = LogisticRegression(max_iter=1000).fit(approved_df[features], approved_df["bad_flag"])
approved_df["prob"] = scorer.predict_proba(approved_df[features])[:, 1]
rejected_df["prob"] = scorer.predict_proba(rejected_df[features])[:, 1]

inferrer = RejectInferenceFactory.create(
    "parceling", target_col="bad_flag", score_col="prob", score_direction="high_bad", random_state=42,
)
df_combined = inferrer.infer(approved_df, rejected_df, score_col="prob")
print(len(approved_df), len(rejected_df), len(df_combined), df_combined["bad_flag"].isna().sum())
```

`rejected_df` must contain the score column but no label. With `"parceling"`, each rejected row gets a label drawn at random
from the bad rate of the approved rows in the same score band. `prob` is a probability of bad, so the example sets
`score_direction="high_bad"`. The default, `"high_good"`, is for credit scores where a higher value means lower risk. See
[Reject Inference and Distribution Adaptation](reject_inference.md) for the other methods.

## 6. Row-Level INS / OOS / OOT Labels: `SampleAnalysisPipeline`

`SampleAnalysisPipeline` evaluates many (OOT window, INS/OOS ratio, seed) combinations and recommends one per target. With
`materialize_split=True` (0.6.7+) it also **replays the recommended split into row-level labels**, so the modeling split
matches the statistics it was chosen from. Pass the labels to a modeling pipeline through its `split_col` (the values are
`ins`, `oos`, and `oot`).

!!! warning "Defaults tied to a legacy dataset"

    `target_cols`, `time_dims`, `population_dims`, `profile_cols`, and `approved_col` default to column names from the
    library author's data (for example `y_flag_dpd7_in_mob1` and `strategy_version`). On your own data, override them all, or
    `run()` raises `KeyError: Missing required columns`. The default search is large (4 targets x 4 windows x 3 ratios x
    20 seeds = 960 splits). Set `dry_run=True` to only estimate the number of splits.

```python
from Modeling_Tool import SampleAnalysisPipeline, SampleAnalysisPipelineConfig

config = SampleAnalysisPipelineConfig(
    target_cols=["bad_flag"],
    time_col="apply_time",
    time_dims=["apply_month"],
    population_dims=["channel"],
    profile_cols=["age", "income"],
    oot_time_dim="apply_month",
    oot_windows=[1, 2],                 # trailing periods to try as OOT
    ins_oos_ratios=[0.7, 0.8],
    random_seeds=range(3000, 3005),
    min_sample_size=200,
    approved_col=None,
    write_outputs=False,                # skip the summary CSV files
    write_excel=False,                  # skip the Excel report
    # Row-level labels
    id_col="loan_id",
    materialize_split=True,
    oot_cutoff="2025-08",               # optional: OOT = apply_month >= "2025-08"
    split_col_name="sample_split",
    persist_split_map=True,             # write row_level_split.csv and split_artifact.json
    output_dir="output/sample_analysis",
)
result = SampleAnalysisPipeline(config).run(data)

print(result.split_recommendation[["target_col", "oot_window_periods", "ins_ratio", "seed"]])
print(result.row_level_split.head(3))
print(result.row_level_split["sample_split"].value_counts().to_dict())
print(result.output_paths)

# Attach the labels to the modeling frame (pass sample_split as split_col to a pipeline)
modeling_df = data.merge(result.row_level_split[["loan_id", "sample_split"]], on="loan_id")
print(modeling_df["sample_split"].value_counts().to_dict())
```

| Result attribute | Content |
|---|---|
| `split_recommendation` | One row per target: the (`oot_window_periods`, `ins_ratio`, `seed`) with the smallest bad-rate gap between any two of INS, OOS, and OOT. Candidates need at least `min_sample_size` rows in each part (if none qualifies, all are considered). Ties go to the larger OOT, then to the ratio closest to 75/25 |
| `row_level_split` | `None` unless `materialize_split=True`. Otherwise a long frame with the id column, a column named `target_col` that holds the target's name, and the `split_col_name` column (`ins`, `oos`, or `oot`). Rows without a label for a target are left out |
| `split_artifact` | Audit record per target: seed, ratio, OOT basis, row counts, and a SHA-256 hash of the sorted ids of each part (`ins`, `oos`, `oot`, `full`). Two runs with equal hashes produced the same membership |
| `output_paths` | Paths of the files written |

- `oot_cutoff` fixes the OOT part as `oot_time_dim >= oot_cutoff` instead of the recommended trailing window. The INS/OOS
  pool is then rebuilt from the new boundary, so the rows can differ from the ones scored in the statistics step. Without
  `oot_cutoff`, the replay reproduces those rows exactly.
- `run()` fails loudly if `id_col` has duplicates among the labeled rows (`ValueError`), or if the three parts overlap or do
  not cover every labeled row (`AssertionError`).
- `persist_split_map=True` writes `row_level_split.csv` and `split_artifact.json` to `output_dir`, even when
  `write_outputs` and `write_excel` are off.

## FAQ

??? question "The bad rate of the training sample differs from the full-sample bad rate"

    Check that `stratify=True`. With `stratify=False` the split is purely random, so the bad rates of the two samples vary
    with the draw, and more so on small samples or with a rare bad outcome.

??? question "`balance(method='smote')` or `SampleBalancer(method='nearmiss')` raises `ImportError: imbalanced-learn required`"

    Install the optional dependency:

    ```bash
    pip install 'supermodelingfactory[imblearn]'
    ```

??? question "`ValueError: could not convert string to float` from SMOTE, NearMiss, Tomek, or ENN"

    These methods need numeric columns without NaN. Pass only the numeric feature columns, for example
    `train_df[features + ["bad_flag"]]`, and impute or encode the rest first.

??? question "Every seed gives the same OOT result in `select_sample_seed`"

    That is expected. The model is fixed and the OOT rows do not depend on the seed. Compare `train` with `validation`
    instead. See [Seed Sensitivity](#4-seed-sensitivity-select_sample_seed).
