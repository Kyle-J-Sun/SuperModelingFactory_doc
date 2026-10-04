# Online/Offline Consistency Check (UAT)

Before and after a model goes live, you need proof that the **online** system and the **offline** pipeline produce the same
score and the same model inputs for the same applications. `UATConsistencyChecker` automates that check. It runs two SQL
files, one for the offline results and one for the online results, merges them on `flow_id`, and compares the main model
score, the sub-model scores, every shared feature, and the time fields. It returns a summary and writes an Excel report.

!!! info "The checker is driven by SQL, not by files"

    `UATConsistencyChecker` does not read CSV files or DataFrames. It reads the two SQL files and executes them with the
    runner you pass in. In production the runner is `ODPSRunner()`; any object with a `run_sql(sql, n_process=...)` method
    that returns a DataFrame works. If you already hold both result sets as DataFrames, use the
    [DataFrame route](#8-compare-two-dataframes-you-already-have) instead.

`UATConfig` and `UATConsistencyChecker` are not exported at the top level. Import them from `Modeling_Tool.UAT`:

```python
from Modeling_Tool.UAT import UATConfig, UATConsistencyChecker
```

## 1. Preparation: SQL Files and a Runner

Put two SQL files in the folder `sql_dir`. One pulls the **offline** result and the other the **online** result. Each file
holds a single statement.

```
sql/
├── pull_offline.sql   # offline scoring results: flow_id, main score, features, ...
└── pull_online.sql    # online scoring results: the same columns
```

Both results must follow these rules:

- **Key.** A column named `flow_id`, unique within each result. The checker reports duplicates (`dup_offline`, `dup_online`)
  but does not remove them, and duplicated keys multiply in the merge.
- **Same column names.** The main score and every feature must have the same name online and offline.
- **Numeric features.** Only numeric columns are compared (see [How values are compared](#how-values-are-compared)).
- **A `launch_time` column** in at least one of the two results. When the main scores differ, `check_main_score()` copies
  `launch_time` into its mismatch table and raises `KeyError: "['launch_time'] not in index"` if the column is missing.

The runner pulls the data. In production, create it with `ODPSRunner()`, which reads `ALIBABA_CLOUD_ACCESS_KEY_ID` and
`ALIBABA_CLOUD_ACCESS_KEY_SECRET` from the environment (see [ODPS Data Extraction](odps.md)), and pass it as
`UATConsistencyChecker(config, ODPSRunner())`.

The rest of this page uses a synthetic example and a stand-in runner, so that every snippet runs without MaxCompute. The
stand-in returns an in-memory table for each SQL text. The offline and online frames differ in planted ways: four
applications exist only offline and two only online; three main scores, three `income` values, and two `sub_score` values
differ; two online `age` values are missing; three `apply_time` values are 5 minutes off and one is 30 seconds off; and two
`city` values differ.

```python
import logging
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 300

offline = pd.DataFrame({
    "flow_id":     [f"f{i:04d}" for i in range(n)],
    "user_id":     np.arange(n),
    "launch_time": pd.date_range("2026-03-01", periods=n, freq="15min").astype(str),
    "apply_time":  pd.date_range("2026-03-01 08:00", periods=n, freq="15min").astype(str),
    "score":       rng.uniform(300, 900, n),                    # main model score
    "sub_score":   rng.uniform(0, 1, n),                        # a sub-model score
    "age":         rng.integers(20, 60, n).astype(float),
    "income":      rng.lognormal(10, 0.4, n),
    "city":        rng.choice(["Lima", "Quito"], n),
})

online = pd.concat(
    [offline.iloc[4:], offline.iloc[:2].assign(flow_id=["x1", "x2"])], ignore_index=True,
)                                                               # f0000-f0003 missing; x1, x2 are new
online.loc[10:12, "score"] += 0.5
online.loc[20:22, "income"] += 5.0
online.loc[30:31, "age"] = np.nan
online.loc[40:42, "apply_time"] = (pd.to_datetime(online.loc[40:42, "apply_time"]) + pd.Timedelta(minutes=5)).astype(str)
online.loc[43, "apply_time"] = (pd.to_datetime(online.loc[43, "apply_time"]) + pd.Timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
online.loc[50:51, "sub_score"] += 0.1
online.loc[60:61, "city"] = "Cusco"

os.makedirs("sql", exist_ok=True)
with open("sql/pull_offline.sql", "w") as f:
    f.write("SELECT * FROM offline_scores")
with open("sql/pull_online.sql", "w") as f:
    f.write("SELECT * FROM online_scores")


class FrameRunner:
    """Stand-in for ODPSRunner: returns the in-memory table whose name appears in the SQL text."""

    def __init__(self, tables):
        self.tables = tables

    def run_sql(self, sql, n_process=1, **kwargs):
        for table_name, frame in self.tables.items():
            if table_name in sql:
                return frame.copy()
        raise KeyError(sql)


runner = FrameRunner({"offline_scores": offline, "online_scores": online})

# The checker logs its progress at INFO level, and some of those messages are in Chinese. Keep warnings only.
logging.getLogger("Modeling_Tool.UAT.UAT_Consistency_Checker").setLevel(logging.WARNING)
```

## 2. Configuration: `UATConfig`

```python
config = UATConfig(
    main_model_score_col="score",
    sql_dir="sql",
    offline_sql="pull_offline.sql",
    online_sql="pull_online.sql",
    tol_score=1e-6,                           # tolerance for the main and sub-model scores
    tol_feat=1e-2,                            # tolerance for the features
    info_list=["user_id", "launch_time"],     # identifiers added to the detail tables, not compared
    time_featlist=["apply_time"],             # time columns compared by time difference
    tol_time_seconds=60,
    excel_output_path="uat_report.xlsx",
)
```

| Field | Default | Meaning |
|---|---|---|
| `main_model_score_col` | `"credit_risk_ltrs_subomdel_score"` | Main model score column, with the same name in both results. The default is a placeholder from the library author's data, so always set it |
| `include_submodel_scores` | `True` | `True`: the all-feature check already covers the sub-model scores, and the dedicated sub-model check is skipped. `False`: run it on `submodel_pairs` |
| `excel_output_path` | `online_offline_consistency_report_<YYYYmmdd_HHMMSS>.xlsx` | Path of the Excel report. The timestamp is fixed when the `UATConfig` object is created |
| `sql_dir` | `"sql"` | Folder with the SQL files, absolute or relative to the working directory |
| `offline_sql`, `online_sql` | `"pull_offline.sql"`, `"pull_online.sql"` | SQL file names |
| `tol_score` | `1e-06` | Tolerance for the main and sub-model scores |
| `tol_feat` | `0.01` | Tolerance for the features |
| `n_process` | `max(1, cpu_count - 1)` | Passed to `sqlrunner.run_sql(sql, n_process=...)` as the number of download processes |
| `submodel_pairs` | `{}` | Sub-model score columns as `{offline_column: online_column}`, using the column names of the merged table ([section 5](#5-sub-model-and-time-field-checks)). Used when `include_submodel_scores=False` |
| `excel_font` | `"Arial"` | Font of the report |
| `info_list` | `[]` | Identifier columns, such as `user_id`, that are written after `flow_id` in the detail tables and excluded from the comparison. Names not found in the data are ignored with a warning |
| `time_featlist` | `[]` | Time columns, with the same name online and offline, compared by time difference |
| `tol_time_seconds` | `60.0` | Tolerance for the time columns, in seconds |
| `comparison_block_size` | `128` | Feature pairs compared per block when building the per-flow report. Must be positive, otherwise creating the checker raises `ValueError` |

## 3. One-Click Run: `run()`

`run()` executes the whole workflow and returns the summary table. It also writes the Excel report: there is no way to skip
that step in `run()`.

```python
checker = UATConsistencyChecker(config, runner)       # in production: UATConsistencyChecker(config, ODPSRunner())
summary_df = checker.run()
print(summary_df)
```

`summary_df` has the columns `Check Item`, `Detail`, and `Status`. Some `Detail` texts are partly in Chinese (for example,
the clause that says a main-score mismatch can also mean one side is empty).

!!! warning "The report takes minutes"

    Writing the workbook is slow: ExcelMaster formats every row of each worksheet when it creates it, which costs on the order
    of ten seconds and about 2.5 MB per sheet. A report has six to sixteen sheets, so `run()` takes one to three minutes and
    the file can reach tens of megabytes, whatever the number of rows. While you iterate, use the step-by-step calls below
    and leave out `export_excel()`.

`run()` is equivalent to calling these methods in order:

```python
checker.load_data()                # run both SQL files and merge the results
checker.check_coverage()           # flow_id coverage: both, offline only, online only
checker.check_main_score()         # main model score consistency
checker.check_submodel_features()  # sub-model scores (skipped unless include_submodel_scores=False)
checker.check_all_features()       # every other column pair
checker.check_time_fields()        # time columns
checker.build_per_flow_report()    # one row per flow_id
checker.build_summary()            # the summary table
checker.export_excel()             # the Excel report
```

## 4. Step-by-Step Execution and Intermediate Results

Call the steps in the order above: later steps read the state that earlier ones store. Every method except `load_data()`
raises `RuntimeError: Data not loaded. Call load_data() first.` before the data is loaded.

```python
checker = UATConsistencyChecker(config, runner)
checker.load_data()

coverage = checker.check_coverage()               # dict
main_score = checker.check_main_score()           # dict
submodels = checker.check_submodel_features()     # list; empty while include_submodel_scores=True
features = checker.check_all_features()           # DataFrame
time_fields = checker.check_time_fields()         # DataFrame
per_flow = checker.build_per_flow_report()        # DataFrame
summary_df = checker.build_summary()              # DataFrame

print(coverage)
print(main_score)
print(features)
print(time_fields)
print(per_flow[per_flow["n_feature_mismatch"] > 0].head())
```

| Method | Returns |
|---|---|
| `load_data()` | `None`. It stores `df_offline`, `df_online`, the merged `df_compare`, and `df_both`, the rows present on both sides, which is the base of every comparison |
| `check_coverage()` | `dict` with `n_offline`, `n_online`, `n_common`, `n_only_offline`, `n_only_online`, `dup_offline`, `dup_online` |
| `check_main_score()` | `dict` with `offline_score_col`, `online_score_col`, `n_compared`, `n_null`, `n_one_side_null`, `mean_diff`, `max_abs_diff`, `n_mismatch`, `consistent`. The mismatching rows are kept in `checker.main_score_mismatch_df` |
| `check_submodel_features()` | `list` of `dict`, one per pair, with `submodel`, `n_compared`, `n_one_side_null`, `n_mismatch`, `n_mismatch_gt_1e6`, `max_abs_diff` |
| `check_all_features()` | DataFrame `feature`, `n_compared`, `n_one_side_null`, `n_mismatch`, `pct_mismatch`, `mean_diff`, `max_abs_diff`, sorted by `n_mismatch` descending |
| `check_time_fields()` | DataFrame `time_field`, `offline_col`, `online_col`, `n_compared`, `n_one_side_null`, `n_mismatch`, `pct_mismatch`, `mean_diff_sec`, `max_abs_diff_sec`. Empty when `time_featlist` is empty |
| `build_per_flow_report()` | DataFrame with one row per `flow_id` present on both sides: `flow_id`, the `info_list` columns, `main_score_diff`, `main_score_ok`, one `<submodel>_diff` per pair, `n_feature_mismatch`, `mismatch_features` |
| `build_summary()` | DataFrame `Check Item`, `Detail`, `Status` |
| `export_excel()` | Path of the Excel report (`str`) |

If `main_model_score_col` is not found in the merged table, `check_main_score()` only logs a warning and returns
`{'offline_score_col': None, 'online_score_col': None}`, and the summary then has no `Main Model Score` row. Check the
column name when that row is missing.

### How values are compared

All differences are `online - offline`. A pair of values is a mismatch when exactly one side is missing, or when both
exist and `abs(online - offline)` exceeds the tolerance. Two missing values are consistent and are not counted.

- **Main score:** tolerance `tol_score`.
- **Features:** every column that exists both as `col` and as `col_online`, except `info_list` and `time_featlist` columns,
  with tolerance `tol_feat`. `pct_mismatch` is the percentage of `n_compared + n_one_side_null`. The main score and the
  sub-model scores appear in this table as well, compared with `tol_feat`.
- **Time columns:** both sides are parsed with `pd.to_datetime(errors="coerce")`. A pair mismatches when exactly one side
  does not parse, or when the difference exceeds `tol_time_seconds`.

!!! warning "Only numeric columns are compared"

    Every column is converted with `pd.to_numeric(errors="coerce")`, so a text column becomes all-missing on both sides:
    it shows `n_compared = 0` and no mismatch, and counts as consistent even when its values differ. In the example,
    `city` differs in two rows yet passes. Encode categorical inputs as numbers in the SQL, or compare them separately, for
    example with [`proc_compare`](proc_compare.md). `load_data` also converts any object column that has at least one numeric
    value to numbers, which turns its text values into missing values.

`Modeling_Tool.UAT` also exports the comparison helpers the checker uses, so you can reuse the same rules on your own
columns: `safe_diff(a, b)`, `safe_eq(a, b)`, `mismatch_mask(a, b, tol)`, `time_diff_seconds(a, b)`, and
`time_mismatch_mask(a, b, tol_seconds)`. Each takes two Series.

```python
from Modeling_Tool.UAT import mismatch_mask, time_mismatch_mask

a = pd.Series([1.0, 2.0, np.nan, np.nan])
b = pd.Series([1.0, 2.5, 3.0, np.nan])
print(mismatch_mask(a, b, tol=0.1).tolist())        # [False, True, True, False]

t_online = pd.Series(["2026-03-01 10:00:00", "2026-03-01 10:10:00"])
t_offline = pd.Series(["2026-03-01 10:00:30", "2026-03-01 10:00:00"])
print(time_mismatch_mask(t_online, t_offline, tol_seconds=60).tolist())      # [False, True]
```

## 5. Sub-Model and Time Field Checks

**Sub-model scores.** With the default `include_submodel_scores=True`, sub-model scores are checked as ordinary features.
Set `include_submodel_scores=False` to run the dedicated check, which compares every pair in `submodel_pairs` with
`tol_score`. Name the **online column as it appears after the merge**: `<name>_online` when both SQL results use the same name,
or the plain name when only the online result has the column.

```python
config_sub = UATConfig(
    main_model_score_col="score",
    sql_dir="sql",
    include_submodel_scores=False,
    submodel_pairs={"sub_score": "sub_score_online"},       # {offline column: online column in the merged table}
    info_list=["user_id", "launch_time"],
)
checker_sub = UATConsistencyChecker(config_sub, runner)
checker_sub.load_data()
checker_sub.check_coverage()
checker_sub.check_main_score()
print(checker_sub.check_submodel_features())
```

!!! warning "Name the merged column"

    `{"sub_score": "sub_score"}` compares the offline column with itself, so it always passes. A pair whose columns are not
    in the merged table is skipped with a warning.

**Time columns.** List a column in `time_featlist`, with the same name in both SQL results. Offline values come from `col` and
online values from `col_online`. The column is compared by time difference and excluded from the feature check. A column
that is missing from the data is skipped with a warning.

## 6. The Merge

`load_data()` merges the two results with an outer join on `flow_id`. A column that appears in both results keeps its name
for the offline value and gets the suffix `_online` for the online value. A column that appears in one result only keeps its
name. Rows present on one side only are counted by `check_coverage()` and left out of every comparison.

## 7. Excel Report

`export_excel()` (and `run()`) writes these sheets to `excel_output_path`:

| Sheet | Content |
|---|---|
| `Executive Summary` | Overall metrics, the sub-model summary, the 20 features with most mismatches, the time-field summary, and the flow_id coverage split |
| `Main Score Mismatch` | The mismatching rows, largest difference first, with the `info_list` columns |
| `Submodel Score Detail` | The same for every sub-model pair, or a note when the check is skipped |
| `Feature Mismatch Summary` | Every feature with mismatches (with the name of its detail sheet), and the list of fully consistent features |
| `Feat_<name>` | Row-level mismatches of the ten features with the most mismatches. Sheet names are cut to 31 characters and made unique |
| `Time Field Consistency` | Time-field summary and mismatching rows. Present only when `time_featlist` is set |
| `Per Flow-ID Report` | flow_ids with feature mismatches, flow_ids with a main-score mismatch, and a sample of 50 clean flow_ids |

Some titles inside the workbook are in Chinese. The font of every cell is set by `excel_font`.

## 8. Compare Two DataFrames You Already Have

`ScoreConsistencyUATPipeline` wraps the same checker for data you already hold. Pass both frames to `run()`, which skips the SQL
step. Both frames need a `flow_id` column, and the `launch_time` rule from section 1 still applies. For the pipeline's
settings, see [Top-Level Pipelines](../pipeline_one_click.md).

```python
from Modeling_Tool import ScoreConsistencyUATPipeline, ScoreConsistencyUATPipelineConfig

pipeline_config = ScoreConsistencyUATPipelineConfig(
    main_model_score_col="score",
    info_list=["user_id", "launch_time"],
    time_featlist=["apply_time"],
    write_outputs=False,
    write_excel=False,
)
pipeline_result = ScoreConsistencyUATPipeline(pipeline_config).run(offline_data=offline, online_data=online)
print(pipeline_result.summary)
```

## FAQ

??? question "`RuntimeError: Data not loaded. Call load_data() first.`"

    Call `load_data()` before any `check_*`, `build_*`, or `export_excel` call, or use `run()`.

??? question "`KeyError: \"['launch_time'] not in index\"` from `check_main_score()`"

    The main scores differ and neither SQL result has a `launch_time` column. Add the column to a SQL result. Put it in
    `info_list` to keep it out of the feature comparison.

??? question "`FileNotFoundError` for a SQL file"

    The file `sql_dir/offline_sql` or `sql_dir/online_sql` must exist. A relative `sql_dir` is resolved against the current
    working directory.

??? question "Online and offline columns do not line up"

    After the merge, an online column whose name also exists offline gets the suffix `_online`. The main score and every
    feature must have the same name in both results, otherwise they are not paired.

??? question "A feature shows `n_compared = 0`"

    The column is not numeric, so it is not compared (see [How values are compared](#how-values-are-compared)). The
    same happens when a column holds only missing values.

??? question "`run()` takes minutes"

    The Excel export dominates the run time (section 3). Use the step-by-step calls without `export_excel()` while you
    iterate, and export once at the end.

??? question "The summary has no `Main Model Score` row"

    `main_model_score_col` was not found in the merged table. Check its spelling against the SQL result columns, and
    remember that its default is a placeholder.
