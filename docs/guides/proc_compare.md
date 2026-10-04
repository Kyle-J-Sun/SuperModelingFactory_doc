# Data Consistency Comparison with ProcCompare

`ProcCompareEngine` compares two tables the way SAS `PROC COMPARE` does. It aligns the rows by a key (or by row number),
compares the shared columns with tolerances, and reports differences at five levels: schema, coverage, column, row, and
cell. Use it to check a backfill against a fresh extraction, two feature snapshots, or sample details before and after a
model goes live.

- Inputs are pandas `DataFrame` objects or local CSV files. Large CSV files are compared in chunks.
- Numeric, datetime, and text columns are compared with absolute, relative, and per-column tolerances.
- Results are DataFrames. CSV files and an Excel report are optional.

!!! warning "Choose the alignment explicitly"

    `key_cols` is required. If the tables have no key, set `row_order_compare=True` to confirm that they match one to one
    by row number. This prevents an unordered table from being compared by position by accident.

!!! warning "`write_outputs` is on by default"

    `proc_compare(left, right, key_cols=[...])` writes six CSV files to `output/proc_compare/` (relative to the working
    directory) unless you pass `write_outputs=False`. The examples below pass it wherever files are not the point.

## Example data

Every snippet on this page runs top to bottom in one Python session. The two tables share the keys `f1` to `f4`, and each
has one row the other lacks (`f6` on the left, `f7` on the right). The differences are planted:

| Row | Planted difference |
|---|---|
| `f1` | `credit_limit` differs by 0.5, and `apply_time` by 3 seconds |
| `f2` | `score` is 0.20 against 0.25, and `channel` changes from `Facebook` to `Meta` |
| `f3` | `credit_limit` is missing on the right, and `apply_time` differs by 10 minutes |
| `f4` | `credit_limit` is missing on both sides |
| all | `etl_time` differs: a column you would normally ignore |

```python
import numpy as np
import pandas as pd

left = pd.DataFrame({
    "flow_id":      ["f1", "f2", "f3", "f4", "f6"],
    "apply_date":   ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"],
    "score":        [0.10, 0.20, 0.30, 0.40, 0.60],
    "credit_limit": [1000.0, 2000.0, 3000.0, np.nan, 5000.0],
    "channel":      ["Google", "Facebook", "Organic", "Organic", "Google"],
    "apply_time":   ["2026-01-01 10:00:00", "2026-01-01 11:00:00", "2026-01-02 09:30:00",
                     "2026-01-02 12:00:00", "2026-01-03 08:00:00"],
    "etl_time":     ["2026-02-01 00:00:00"] * 5,
})

right = pd.DataFrame({
    "flow_id":      ["f1", "f2", "f3", "f4", "f7"],
    "apply_date":   ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"],
    "score":        [0.10, 0.25, 0.30, 0.40, 0.70],
    "credit_limit": [1000.5, 2000.0, np.nan, np.nan, 6000.0],
    "channel":      ["Google", "Meta", "Organic", "Organic", "Google"],
    "apply_time":   ["2026-01-01 10:00:03", "2026-01-01 11:00:00", "2026-01-02 09:40:00",
                     "2026-01-02 12:00:00", "2026-01-03 08:00:00"],
    "etl_time":     ["2026-02-02 00:00:00"] * 5,
    "risk_level":   ["low", "low", "mid", "mid", "high"],       # a column that exists on the right only
})
```

## 1. Quick Start

### 1.1 DataFrame Mode

```python
from Modeling_Tool import ProcCompareEngine, ProcCompareConfig

cfg = ProcCompareConfig(
    key_cols=["flow_id"],
    write_outputs=False,
    detail_mode="top",
    top_n=1000,
)
result = ProcCompareEngine(cfg).run(left, right)

print(result.coverage_summary)
print(result.column_summary)
print(result.cell_mismatches.head())
```

`ProcCompareEngine(config=None)` takes a `ProcCompareConfig`, and `run(left, right)` returns a `ProcCompareResult`
([the output](#7-interpreting-the-output)). The convenience function `proc_compare(left, right, **kwargs)` builds the config from keyword
arguments and runs it. Every `ProcCompareConfig` field is accepted ([full list](#9-full-parameter-table)), and an unknown
keyword raises `TypeError`.

```python
from Modeling_Tool import proc_compare

result = proc_compare(left, right, key_cols=["flow_id"], write_outputs=False)
```

### 1.2 Large CSV Mode

When `left` or `right` is a path (`str` or `pathlib.Path`), the engine reads the file as a stream, hash-partitions the
rows into temporary CSV chunks by `key_cols`, and compares one partition at a time, so a large file is never held in memory
whole. The two sides can mix a path and a `DataFrame`.

```python
left.to_csv("base_snapshot.csv", index=False)
right.to_csv("new_snapshot.csv", index=False)

result = proc_compare(
    "base_snapshot.csv",
    "new_snapshot.csv",
    key_cols=["flow_id"],
    chunk_size=2,                 # rows read per chunk (200000 by default; tiny here for the demo)
    n_partitions=4,               # hash partitions (16 by default)
    backend="thread",
    output_dir="output/proc_compare_csv",
    write_outputs=True,
    ignore_cols=["etl_time"],
    datetime_cols=["apply_time"],
    datetime_tol_seconds=60,
)
print(result.coverage_summary)
print(sorted(result.output_paths))
```

What to know about CSV mode:

- **Temporary files.** The partition files, about as large as both inputs together, go under `output_dir` when
  `write_outputs` or `write_excel` is on, and under the system temp folder otherwise. They are deleted when the run ends.
- **Backends.** `"sequential"` (default), `"thread"`, and `"process"` return identical results. Use `"process"` only for large
  inputs, because starting the workers costs several seconds.
- **Schema.** `schema_summary` is built from the first 1,000 rows of each file. Every partition is then parsed by pandas on its
  own.
- **Datetimes.** `read_csv` returns date columns as text, so declare them with `datetime_cols`
  ([datetime columns](#43-datetime-columns)).
- **Order.** `row_summary` lists the keys partition by partition, not in file order.

## 2. Alignment Methods

### 2.1 Align by Primary Key

`key_cols` is the recommended alignment. It takes one or several columns, which must have the same dtype on both sides: a
mismatch such as `int64` against `object` raises a pandas `ValueError` from the merge, so cast the key first.

```python
result = proc_compare(left, right, key_cols=["flow_id"], write_outputs=False)
result = proc_compare(left, right, key_cols=["flow_id", "apply_date"], write_outputs=False)    # composite key
print(result.coverage_summary)
```

`coverage_summary` reports the rows on the left (`n_left_rows`), the rows on the right (`n_right_rows`), the rows present
on both sides (`n_common_rows`), the rows only on one side (`n_left_only_rows`, `n_right_only_rows`), and the number of
columns that were compared (`n_compare_columns`). Key columns are never compared as values.

### 2.2 Align by Row Number

If the tables have no key but are certain to have the same row order, set `row_order_compare=True`. The engine numbers the
rows of a copy of each table from 0 and uses that number as the key. It appears as the column `__proc_compare_row_number__`
in `row_summary` and `cell_mismatches`. Your frames are not modified.

```python
result = proc_compare(
    left.drop(columns=["flow_id"]),
    right.drop(columns=["flow_id"]),
    row_order_compare=True,
    write_outputs=False,
)
print(result.row_summary)
```

This mode suits quick checks on small samples. In CSV mode the row numbers run across chunks, so the files must have
exactly the same order.

## 3. Column Selection

By default, every column present in both tables is compared, except the key columns and `ignore_cols`.

```python
result = proc_compare(left, right, key_cols=["flow_id"], ignore_cols=["etl_time"], write_outputs=False)
print(result.column_summary[["column", "n_mismatch"]])
```

To compare only some columns, list them in `compare_cols`:

```python
result = proc_compare(
    left, right,
    key_cols=["flow_id"],
    compare_cols=["score", "credit_limit", "risk_level"],
    write_outputs=False,
)
print(result.schema_summary[["column", "role", "requested_for_compare", "eligible_for_compare", "status"]])
```

A column that exists on one side only is never compared. It is not an error: it appears in `schema_summary` with the
`status` `left_only_column` or `right_only_column` and the `role` `schema_only` (here, `risk_level`).
`requested_for_compare` says whether you asked for the column, and `eligible_for_compare` says whether it exists on both
sides and enters the value comparison. Only eligible columns get the `role` `compare`. The other roles are `key`, `ignored`
(listed in `ignore_cols`), and `not_compared` (not listed in `compare_cols`).

!!! warning "Typos in `compare_cols` and `ignore_cols` are silent"

    A name that exists in neither table does not raise an error and does not appear in `schema_summary`. Check
    `result.column_summary["column"]` to confirm that the columns you meant to compare were compared.

## 4. Tolerances and Missing-Value Rules

A column is compared **as a datetime** when it has a datetime dtype or is configured as one
([4.3](#43-datetime-columns)), **as a number** when either side has a numeric dtype or every non-null value on both sides
parses as a number, and **as text** otherwise. Text is compared exactly, including case and spaces.

### 4.1 Numeric Columns

Two values are equal when

```text
abs(left - right) <= numeric_tol + numeric_rtol * abs(right)
```

The defaults are `numeric_tol=1e-8` and `numeric_rtol=0.0`.

```python
result = proc_compare(
    left, right,
    key_cols=["flow_id"],
    ignore_cols=["etl_time"],
    numeric_tol=1e-6,
    numeric_rtol=1e-3,
    write_outputs=False,
)
print(result.column_summary[["column", "n_mismatch", "max_abs_diff"]])
```

With a relative tolerance of 0.1% the 0.5 difference on `f1.credit_limit` is accepted (the limit is about 1.0), while the
0.05 difference on `f2.score` still counts.

### 4.2 Per-Column Tolerance

`per_column_tolerance` overrides the global values for specific columns. A number is the absolute tolerance
(`rtol` stays global), and a dictionary can set `tol`, `rtol`, and `datetime_tol_seconds`.

```python
result = proc_compare(
    left, right,
    key_cols=["flow_id"],
    ignore_cols=["etl_time"],
    numeric_tol=1e-8,
    per_column_tolerance={
        "score": 0.1,
        "credit_limit": {"tol": 1.0, "rtol": 0.0},
    },
    write_outputs=False,
)
print(result.column_summary[["column", "n_mismatch", "max_abs_diff"]])
```

### 4.3 Datetime Columns

Datetime columns are compared by the difference in seconds, `left - right`, against `datetime_tol_seconds` (default `0.0`).
A column is treated as a datetime when:

- it has a datetime64 dtype on either side (always, whatever the tolerance);
- it is listed in `datetime_cols`;
- its `per_column_tolerance` entry is a dictionary with the key `datetime_tol_seconds`; or
- the global `datetime_tol_seconds` is above 0 and every non-null value on both sides looks like `YYYY-MM-DD`, optionally
  followed by a time, and parses with `pd.to_datetime`. This probe only runs for columns that are not numeric.

Anything else, such as date strings with no configuration, is compared as text.

```python
result = proc_compare(
    left, right,
    key_cols=["flow_id"],
    ignore_cols=["etl_time"],
    datetime_cols=["apply_time"],
    datetime_tol_seconds=60,
    per_column_tolerance={"apply_time": {"datetime_tol_seconds": 5}},
    write_outputs=False,
)
print(result.cell_mismatches[result.cell_mismatches["column"] == "apply_time"])
```

The 3-second difference on `f1` is within the 5-second limit of `apply_time`, so only the 10-minute difference on `f3` is
reported, with `diff = -600.0`.

### 4.4 Missing Values

By default, a cell is a mismatch when exactly one side is null, and consistent when both sides are null. Set
`both_null_equal=False` to count two nulls as a mismatch. `missing_values` lists extra sentinel values to convert to missing
on both sides, in every column, before the comparison.

```python
a = pd.DataFrame({"flow_id": ["f1", "f2", "f3"], "income": [-999.0, 5000.0, np.nan], "city": ["NULL", "Lima", None]})
b = pd.DataFrame({"flow_id": ["f1", "f2", "f3"], "income": [np.nan, 5000.0, np.nan], "city": [None, "Lima", None]})

as_is = proc_compare(a, b, key_cols=["flow_id"], write_outputs=False)
strict = proc_compare(a, b, key_cols=["flow_id"], both_null_equal=False, write_outputs=False)
cleaned = proc_compare(a, b, key_cols=["flow_id"], missing_values=["", "NULL", -999], write_outputs=False)

for name, res in [("as is", as_is), ("both_null_equal=False", strict), ("missing_values", cleaned)]:
    print(name, res.column_summary[["column", "n_mismatch", "n_one_side_null", "n_both_null"]].values.tolist())
```

## 5. Controlling Mismatch Details

`detail_mode` controls `cell_mismatches`, the cell-level list. The other result tables are always complete.

| `detail_mode` | Behavior |
|---|---|
| `"top"` (default) | Keep at most `top_n` cells (default `1000`). At or below `top_n` mismatches you get all of them, grouped by column. Above it, you get the `top_n` cells with the largest `abs_diff`, sorted by `abs_diff` descending. Cells without an `abs_diff` (text mismatches and one-sided nulls) rank last |
| `"full"` | Keep every mismatched cell. Use it for audits and small tables |
| `"none"` | Return an empty `cell_mismatches` with the right columns |

```python
result = proc_compare(left, right, key_cols=["flow_id"], detail_mode="top", top_n=5, write_outputs=False)
print(result.cell_mismatches)
```

!!! tip "Large tables"

    Use `detail_mode="top"` or `"none"`. If you need every difference, use `write_outputs=True` and read the CSV files; do
    not put the full detail into Excel.

## 6. Duplicate Key Handling

With the default `duplicate_key_policy="raise"`, a duplicated key raises `ValueError` naming up to five examples, so a
many-to-many merge cannot distort the result silently.

| `duplicate_key_policy` | Behavior |
|---|---|
| `"raise"` (default) | Raise `ValueError` when either side has duplicate keys |
| `"first"` | Keep the first row of each key on both sides |
| `"all"` | Keep every row. Matching duplicates multiply in the merge, and `n_common_rows` counts the matched pairs |

```python
dup_right = pd.concat([right, right.iloc[[0]]], ignore_index=True)       # f1 appears twice on the right
result = proc_compare(left, dup_right, key_cols=["flow_id"], duplicate_key_policy="first", write_outputs=False)
print(result.duplicate_key_summary)
```

`duplicate_key_summary` has one row per duplicated key and side, with the columns `side` (the `left_name` or `right_name`),
`duplicate_rows` (how many rows share the key), and the key columns. It is empty when there are no duplicates.

## 7. Interpreting the Output

`ProcCompareEngine.run()` returns a `ProcCompareResult` dataclass:

| Field | Content |
|---|---|
| `coverage_summary` | One row: `left_name`, `right_name`, `n_left_rows`, `n_right_rows`, `n_common_rows`, `n_left_only_rows`, `n_right_only_rows`, `n_compare_columns` |
| `schema_summary` | One row per column of either table: `column`, `role`, `requested_for_compare`, `eligible_for_compare`, `in_left`, `in_right`, `dtype_left`, `dtype_right`, `status`, `dtype_equal` |
| `column_summary` | One row per compared column, sorted by `n_mismatch` (descending) then `column`: `column`, `n_compared`, `n_equal`, `n_mismatch`, `pct_mismatch`, `n_one_side_null`, `n_both_null`, `mean_diff`, `max_abs_diff` |
| `row_summary` | One row per key in either table: the key columns, `row_status` (`both`, `left_only`, or `right_only`), `n_cell_mismatch`, and `mismatch_columns` (comma-separated) |
| `cell_mismatches` | One row per mismatched cell: the key columns, `column`, `left_value`, `right_value`, `diff`, `abs_diff`. Controlled by `detail_mode` and `top_n` |
| `duplicate_key_summary` | Duplicated keys per side (section 6) |
| `output_paths` | `dict` from table name to CSV path. Empty unless `write_outputs=True` |
| `report_path` | Path of the Excel report, or `None` |

Reading the numbers:

- `n_compared` is the number of rows present on both sides, the same for every column. `pct_mismatch` is `n_mismatch / n_compared`
  in percent.
- A cell mismatches when its difference exceeds the tolerance, or when exactly one side is null (counted in
  `n_one_side_null`). Two nulls count only when `both_null_equal=False` (counted in `n_both_null`).
- `diff` is `left - right`, in seconds for datetime columns. It and `abs_diff` are `NaN` for text mismatches and for a null
  on one side. `mean_diff` is the mean of `diff` over the rows where it exists, and `max_abs_diff` the largest `abs_diff`; both
  are `NaN` for text columns. In CSV mode they are aggregated over all partitions.

Common checks:

```python
result = proc_compare(left, right, key_cols=["flow_id"], ignore_cols=["etl_time"], write_outputs=False)

# Coverage gaps
print(result.coverage_summary[["n_common_rows", "n_left_only_rows", "n_right_only_rows"]])

# Which columns differ most?
print(result.column_summary.sort_values("n_mismatch", ascending=False).head(20))

# Which keys have several mismatched columns?
print(result.row_summary.sort_values("n_cell_mismatch", ascending=False).head(20))

# The cells behind the counts
print(result.cell_mismatches.head(100))
```

## 8. Output Files and Excel Report

With `write_outputs=True` (the default), the engine creates `output_dir` (default `output/proc_compare`) and writes one CSV
file per result table: `coverage_summary.csv`, `schema_summary.csv`, `column_summary.csv`, `row_summary.csv`,
`cell_mismatches.csv`, and `duplicate_key_summary.csv`. The paths are in `result.output_paths`.

```python
result = proc_compare(
    left, right,
    key_cols=["flow_id"],
    output_dir="output/proc_compare",
    write_outputs=True,
)
print(result.output_paths)
```

`write_excel=True` writes an ExcelMaster report with the sheets `Coverage`, `Schema`, `Column_Summary`, `Row_Summary`,
`Cell_Mismatches`, and `Duplicate_Keys`. The default path is `<output_dir>/Proc_Compare_Report.xlsx`, and
`excel_output_path` overrides it. `max_excel_rows` caps the rows written per sheet.

```python
result = proc_compare(
    left, right,
    key_cols=["flow_id"],
    write_outputs=False,
    write_excel=True,
    excel_output_path="output/proc_compare/report.xlsx",
)
print(result.report_path)
```

!!! warning "The Excel report is slow and large"

    ExcelMaster formats every row of each worksheet when it creates it, so a report takes on the order of ten seconds per
    sheet (about a minute for the six sheets) and is roughly 15 MB, even for a tiny comparison. For large tables, read the
    CSV files instead.

## 9. Full Parameter Table

These are the fields of `ProcCompareConfig`, and the keyword arguments of `proc_compare`.

| Parameter | Default | Description |
|---|---|---|
| `output_dir` | `"output/proc_compare"` | Folder for the CSV files, the Excel report, and CSV-mode temporary files |
| `write_outputs` | `True` | Write the six CSV files |
| `write_excel` | `False` | Write the ExcelMaster report |
| `left_name` | `"left"` | Name of the left table, written to `coverage_summary` and `duplicate_key_summary` |
| `right_name` | `"right"` | Name of the right table |
| `key_cols` | `None` | Key columns. Required unless `row_order_compare=True` |
| `row_order_compare` | `False` | Align by row number when there is no key |
| `compare_cols` | `None` | Columns to compare. `None` compares every column present in both tables |
| `ignore_cols` | `[]` | Columns excluded from the comparison |
| `chunk_size` | `200000` | Rows read per chunk in CSV mode |
| `n_partitions` | `16` | Hash partitions in CSV mode |
| `backend` | `"sequential"` | CSV partition backend: `"sequential"`, `"thread"`, or `"process"` |
| `compare_block_size` | `64` | Columns compared per vectorized block. Lower it for very wide tables to limit memory. It does not change results |
| `numeric_tol` | `1e-08` | Absolute numeric tolerance |
| `numeric_rtol` | `0.0` | Relative numeric tolerance, applied to `abs(right)` |
| `datetime_tol_seconds` | `0.0` | Tolerance for datetime columns, in seconds |
| `datetime_cols` | `[]` | Columns to compare as datetimes. Recommended in CSV mode |
| `per_column_tolerance` | `{}` | Per-column overrides: a number, or a dictionary with `tol`, `rtol`, `datetime_tol_seconds` |
| `both_null_equal` | `True` | Count two nulls as consistent |
| `missing_values` | `[]` | Extra values to treat as missing, on both sides and in every column |
| `detail_mode` | `"top"` | `"top"`, `"full"`, or `"none"` |
| `top_n` | `1000` | Cells kept when `detail_mode="top"` |
| `duplicate_key_policy` | `"raise"` | `"raise"`, `"first"`, or `"all"` |
| `excel_output_path` | `None` | Excel path. `None` uses `<output_dir>/Proc_Compare_Report.xlsx` |
| `max_excel_rows` | `100000` | Maximum rows written per Excel sheet |

Invalid settings raise `ValueError` when the engine is created: `backend`, `detail_mode`, and `duplicate_key_policy` must be
one of the listed values, `chunk_size`, `n_partitions`, `compare_block_size`, and `top_n` must be positive, and the three
tolerances must not be negative. A missing key raises `ValueError` too, either `Provide key_cols or set row_order_compare=True.`
or `left dataset is missing key columns: [...]` (`right` for the other side).

## 10. Recommended Practices

- Prefer `key_cols` for production tables, and do not rely on row order.
- For large CSV files, set `chunk_size` and `n_partitions` first so that no step needs the whole file in memory.
- For very wide tables, lower `compare_block_size` to limit the memory of one column block.
- Keep `detail_mode="top"` for large tables, and use `"full"` only when an audit requires it.
- Set tolerances per column for amounts, probabilities, and timestamps.
- Ignore ETL times, batch IDs, and export times with `ignore_cols`.
- To compare two snapshots pulled from ODPS, first write them to local CSV files with `ODPSRunner` or
  `ParallelODPSManager.pull()` (see [ODPS Data Extraction](odps.md)), then compare the files.
