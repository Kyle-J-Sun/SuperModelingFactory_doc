# Online/Offline Consistency Check (UAT)

Before and after a model goes live, you need to verify that the **online** and **offline** scores and model input features agree for the same set of users. SuperModelingFactory's `UATConsistencyChecker` packages the complete logic of `99_uat_validation.ipynb` into a reusable class.

!!! info "This is a SQL / ODPS-driven checker"

    The real `UATConsistencyChecker` **does not read CSV files**. Instead it runs two SQL files (online and offline) through `ODPSRunner` to pull data → merge → compare, and outputs an Excel report.

## 1. Preparation: SQL Files + ODPSRunner

Prepare two SQL files in the `sql_dir` directory, which pull the **offline** and **online** results respectively. Both use `flow_id` as the primary key and contain the main-model score column and each model input feature (column names must match between online and offline):

```
sql/
├── pull_offline.sql   # offline scoring results (flow_id, main model score, features...)
└── pull_online.sql    # online scoring results (same structure)
```

## 2. Configuration — `UATConfig`

```python
from Modeling_Tool.UAT.UAT_Consistency_Checker import UATConfig

config = UATConfig(
    main_model_score_col="credit_risk_ltrs_subomdel_score",
    sql_dir="sql",
    offline_sql="pull_offline.sql",
    online_sql="pull_online.sql",
    tol_score=1e-6,                  # tolerance for main/sub-model scores
    tol_feat=1e-2,                   # feature tolerance
    info_list=["user_id", "curp"],   # identifier fields output with the report (not compared)
    time_featlist=["apply_time"],    # time fields compared by second-level time difference
    tol_time_seconds=60,
    excel_output_path="uat_report.xlsx",
)
```

### `UATConfig` Fields

| Field | Default | Description |
|------|-------|------|
| `main_model_score_col` | `"credit_risk_ltrs_subomdel_score"` | Main-model score column name (same name in the online/offline SQL) |
| `sql_dir` | `"sql"` | SQL file directory |
| `offline_sql` / `online_sql` | `"pull_offline.sql"` / `"pull_online.sql"` | Offline/online SQL file names |
| `tol_score` | `1e-6` | Comparison tolerance for main/sub-model scores |
| `tol_feat` | `1e-2` | Comparison tolerance for feature variables |
| `n_process` | `cpu_count-1` | Number of processes for concurrent SQL pulls |
| `include_submodel_scores` | `True` | Whether sub-model scores are already covered as features (when `False`, fill in `submodel_pairs`) |
| `submodel_pairs` | `{}` | Sub-model score column pairs `{offline_col: online_col}` |
| `info_list` | `[]` | Identifier fields output with the report (such as `user_id`/`curp`); not compared |
| `time_featlist` | `[]` | Time fields to compare with time semantics |
| `tol_time_seconds` | `60.0` | Time-difference tolerance (seconds) |
| `excel_font` | `"Arial"` | Excel report font |
| `excel_output_path` | Timestamped file name | Excel report output path |

## 3. One-Click Run — `run()`

`run()` executes the full workflow in order, returns the summary table, and writes the Excel report:

```python
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
from Modeling_Tool.UAT.UAT_Consistency_Checker import UATConsistencyChecker

checker = UATConsistencyChecker(config, ODPSRunner())
summary_df = checker.run()   # pull data → coverage → main score → sub-models → all features → time fields → per-flow → summary → Excel
print(summary_df)
```

Internally, `run()` is equivalent to calling the following in order:

```python
checker.load_data()                # §1 run both SQL files, merge online and offline
checker.check_coverage()           # §2 coverage (online only / offline only / both)
checker.check_main_score()         # §3 main-model score consistency
checker.check_submodel_features()  # §5 sub-model checks (optional)
checker.check_all_features()       # §6 consistency of all features
checker.check_time_fields()        # §7 time fields
checker.build_per_flow_report()    # §8 per-flow_id details
checker.build_summary()            # §9 summary
checker.export_excel()             # §10 Excel output
```

## 4. Step-by-Step Execution and Intermediate Results

```python
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
from Modeling_Tool.UAT.UAT_Consistency_Checker import UATConsistencyChecker

checker = UATConsistencyChecker(config, ODPSRunner())
checker.load_data()

coverage   = checker.check_coverage()      # dict
main_score = checker.check_main_score()    # dict
feat_df    = checker.check_all_features()   # DataFrame (per-feature agreement rate)

report_path = checker.export_excel()        # returns the Excel path
print(f"Report generated: {report_path}")
```

## 5. Sub-Model / Time Field Checks

- **Sub-model scores**: if `include_submodel_scores=False`, give the offline↔online column-name mapping in `submodel_pairs`, and `check_submodel_features()` compares them using `tol_score`.
- **Time fields**: fields in `time_featlist` are parsed with `pd.to_datetime` and compared with a second-level tolerance of `tol_time_seconds`, and are excluded from the numeric feature comparison.

## FAQ

??? question "`RuntimeError: Data not loaded. Call load_data() first.`"

    You must call `load_data()` before any `check_*` / `build_*` / `export_excel` (or just use `run()`).

??? question "SQL file not found"

    The path built from `sql_dir` + `offline_sql`/`online_sql` must exist; `sql_dir` is an absolute path or relative to the current working directory.

??? question "Online and offline column names don't line up"

    After merging, online columns automatically get an `_online` suffix and offline columns keep their original names; the main-model score column and feature columns must have the **same name** online and offline.
