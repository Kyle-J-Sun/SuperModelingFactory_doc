# Data Consistency Comparison with ProcCompare

`ProcCompareEngine` is the general-purpose dataset consistency comparison tool in the SMF Core layer, positioned like SAS `proc compare`. It suits comparing two wide tables, two CSVs, two feature snapshots, backfilled data against newly extracted data, sample details before and after going live, and similar cases.

The first version focuses on supporting:

- pandas `DataFrame` comparison.
- Chunked comparison of large local CSV tables.
- Alignment by primary key, or explicitly by row number.
- Summaries of schema / coverage / column-level / row-level / cell-level mismatches.
- Tolerance-based comparison of numeric, datetime, and string fields.
- Optional CSV output and ExcelMaster report.

!!! warning "The alignment method must be explicit"

    `key_cols` is required by default. If there is no primary key, you must explicitly set `row_order_compare=True`, which confirms that the two tables can be aligned one-to-one by row number. This prevents unordered large tables from being compared by row number by mistake.

## 1. Quick Start

### 1.1 DataFrame Mode

```python
import pandas as pd
from Modeling_Tool import ProcCompareEngine, ProcCompareConfig

left = pd.DataFrame({
    "flow_id": ["f1", "f2", "f3"],
    "score": [0.10, 0.20, 0.30],
    "channel": ["Google", "Facebook", "Organic"],
})

right = pd.DataFrame({
    "flow_id": ["f1", "f2", "f4"],
    "score": [0.10, 0.25, 0.40],
    "channel": ["Google", "Meta", "Organic"],
})

cfg = ProcCompareConfig(
    key_cols=["flow_id"],
    output_dir="output/proc_compare_demo",
    detail_mode="top",
    top_n=1000,
    write_outputs=True,
    write_excel=True,
)

result = ProcCompareEngine(cfg).run(left, right)

result.coverage_summary
result.column_summary
result.cell_mismatches.head()
```

You can also use the convenience function:

```python
from Modeling_Tool import proc_compare

result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    write_outputs=False,
    write_excel=False,
)
```

### 1.2 Large CSV Mode

When `left` / `right` are CSV paths, `ProcCompareEngine` reads the files as a stream, hash-partitions them into temporary chunks by `key_cols`, and then merges and compares partition by partition, so the large CSVs are never read into memory all at once.

```python
from Modeling_Tool import proc_compare

result = proc_compare(
    "data/base_snapshot.csv",
    "data/new_snapshot.csv",
    key_cols=["flow_id"],
    chunk_size=200000,
    n_partitions=32,
    backend="thread",
    output_dir="output/proc_compare_csv",
    write_outputs=True,
)
```

Chunked CSV comparison suits:

- Two large local CSVs.
- Re-checking snapshots that were first written to disk from ODPS / SQL.
- Acceptance checks on the consistency of wide-table results before and after going live.

## 2. Alignment Methods

### 2.1 Align by Primary Key

This is the recommended method. `key_cols` can be one or several columns:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
)

result = proc_compare(
    left,
    right,
    key_cols=["customer_id", "apply_date"],
)
```

The `coverage_summary` in the output reports:

- Number of rows in the left table.
- Number of rows in the right table.
- Common rows present on both sides.
- Rows present only in the left table.
- Rows present only in the right table.
- Number of columns actually compared.

### 2.2 Align by Row Number

If the two tables have no primary key but you are sure the row order is exactly the same, you can set:

```python
result = proc_compare(
    left,
    right,
    row_order_compare=True,
)
```

A temporary row-number column is generated internally for alignment. This mode suits quick checks on small samples and is not recommended for unordered large production tables.

## 3. Column Selection

By default, the common columns of the two tables are compared, excluding the primary key and `ignore_cols`.

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    ignore_cols=["etl_time", "batch_id"],
)
```

To compare only specified columns:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    compare_cols=["score", "credit_limit", "risk_level"],
)
```

If some columns in `compare_cols` exist on only one side, no error is raised directly; they appear in `schema_summary` with the status `left_only_column` or `right_only_column` and the role `schema_only`. `requested_for_compare` indicates whether the user asked to compare the column, and `eligible_for_compare` indicates whether the column exists on both sides and actually enters the value comparison. Only columns with `eligible_for_compare=True` are marked `role="compare"`.

## 4. Tolerances and Missing-Value Rules

### 4.1 Numeric Columns

The numeric comparison rule:

```text
abs(left - right) <= numeric_tol + numeric_rtol * abs(right)
```

Example:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    numeric_tol=1e-6,
    numeric_rtol=1e-4,
)
```

### 4.2 Per-Column Tolerance

`per_column_tolerance` can override the global tolerance for specific columns.

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    numeric_tol=1e-8,
    per_column_tolerance={
        "score": 1e-6,
        "credit_limit": {"tol": 1.0, "rtol": 0.0},
    },
)
```

### 4.3 Datetime Columns

Datetime columns are compared by second-level difference:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    datetime_cols=["apply_time", "event_time"],
    datetime_tol_seconds=60,
    per_column_tolerance={
        "apply_time": {"datetime_tol_seconds": 5},
    },
)
```

After a CSV is read, datetime columns usually become `object`. In that case, declare them explicitly with `datetime_cols`; you can also configure `datetime_tol_seconds` for a column in `per_column_tolerance`. If you set only a global `datetime_tol_seconds > 0`, the engine conservatively probes non-numeric columns that match common date formats. Once matched, both sides are converted uniformly with `pd.to_datetime(errors="coerce")`, so DataFrame, CSV, and all three backends use the same basis.

### 4.4 Missing Values

Default rules:

- Both sides null: consistent.
- One side null: inconsistent.

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    both_null_equal=True,
)
```

If your business has special missing values, you can convert them uniformly to missing:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    missing_values=["", "NULL", -999],
)
```

## 5. Controlling Mismatch Details

`detail_mode` controls how much cell-level mismatch detail is output.

| `detail_mode` | Behavior |
|---|---|
| `"top"` | Default. Keeps only the Top N mismatch details, to keep the report from exploding. |
| `"full"` | Keeps all cell mismatches; suits audits or small tables. |
| `"none"` | Outputs no cell-level details, only the summaries. |

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    detail_mode="top",
    top_n=5000,
)
```

!!! tip "Advice for large tables"

    For large tables, use `detail_mode="top"` or `"none"`. If you must have the full details, use `write_outputs=True` and mainly consult the CSV files on disk; do not write the full details into Excel.

## 6. Duplicate Key Handling

The default is `duplicate_key_policy="raise"`: duplicate keys raise an error immediately, to avoid misjudgments caused by many-to-many merges.

| `duplicate_key_policy` | Behavior |
|---|---|
| `"raise"` | Default. Raise an error when duplicate keys are found. |
| `"first"` | Keep the first record for each key in the comparison. |
| `"all"` | Keep all duplicate records; use only when you explicitly accept a many-to-many merge. |

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    duplicate_key_policy="first",
)
```

Duplicate-key information goes into `duplicate_key_summary`.

## 7. Interpreting the Output

`ProcCompareEngine.run()` returns a `ProcCompareResult`.

| Field | Description |
|---|---|
| `coverage_summary` | Left/right row counts, common / only-left / only-right row counts, number of compared columns. |
| `schema_summary` | For each column: whether it exists in the left/right tables, dtype, column role, schema status, and whether it was requested / actually comparable. |
| `column_summary` | For each compared column: `n_compared`, `n_mismatch`, missing counts, `mean_diff`, and `max_abs_diff`; in CSV partition mode, these are aggregated weighted by the number of valid diffs. |
| `row_summary` | The row status for each key and the number of mismatched columns in that row. |
| `cell_mismatches` | Cell-level mismatch details, controlled by `detail_mode/top_n`. |
| `duplicate_key_summary` | Duplicate-key statistics. |
| `output_paths` | CSV output paths. |
| `report_path` | ExcelMaster report path. |

Common checks:

```python
# Are there coverage gaps?
result.coverage_summary[
    ["n_common_rows", "n_left_only_rows", "n_right_only_rows"]
]

# Which columns differ the most?
result.column_summary.sort_values("n_mismatch", ascending=False).head(20)

# Which flow_ids have multiple mismatched columns?
result.row_summary.sort_values("n_cell_mismatch", ascending=False).head(20)

# View the top cell mismatches
result.cell_mismatches.head(100)
```

## 8. Output Files and Excel Report

Enable CSV output:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    output_dir="output/proc_compare",
    write_outputs=True,
)
```

Output files:

- `coverage_summary.csv`
- `schema_summary.csv`
- `column_summary.csv`
- `row_summary.csv`
- `cell_mismatches.csv`
- `duplicate_key_summary.csv`

Enable the ExcelMaster report:

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    output_dir="output/proc_compare",
    write_outputs=True,
    write_excel=True,
)

print(result.report_path)
```

Default report name:

```text
output/proc_compare/Proc_Compare_Report.xlsx
```

The Excel file contains the summaries and a controlled amount of mismatch detail. `max_excel_rows` limits the maximum number of rows written to Excel per sheet.

## 9. Full Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `output_dir` | `"output/proc_compare"` | Output directory for CSV and Excel. |
| `write_outputs` | `True` | Whether to output CSV results. |
| `write_excel` | `False` | Whether to output the ExcelMaster report. |
| `left_name` | `"left"` | Name of the left table, written into the summaries. |
| `right_name` | `"right"` | Name of the right table, written into the summaries. |
| `key_cols` | `None` | Primary key columns. Recommended; multi-column keys are supported. |
| `row_order_compare` | `False` | Whether to align by row number when there is no primary key. |
| `compare_cols` | `None` | Columns to compare; if omitted, the comparable common columns are compared. |
| `ignore_cols` | `[]` | Columns excluded from the comparison. |
| `chunk_size` | `200000` | Rows read per chunk in CSV mode. |
| `n_partitions` | `16` | Number of hash partitions in CSV mode. |
| `backend` | `"sequential"` | CSV partition comparison backend: `"sequential"` / `"thread"` / `"process"`. |
| `compare_block_size` | `64` | Number of columns per vectorized comparison; lower it for very wide tables to limit peak memory. |
| `numeric_tol` | `1e-8` | Absolute numeric tolerance. |
| `numeric_rtol` | `0.0` | Relative numeric tolerance. |
| `datetime_tol_seconds` | `0.0` | Second-level tolerance for datetime columns. |
| `datetime_cols` | `[]` | Explicitly specified datetime columns; recommended in CSV mode, to keep the object dtype from losing datetime semantics. |
| `per_column_tolerance` | `{}` | Per-column tolerance overrides. |
| `both_null_equal` | `True` | Whether both sides being null counts as consistent. |
| `missing_values` | `[]` | Extra values to treat as missing. |
| `detail_mode` | `"top"` | Cell mismatch detail mode: `"top"` / `"full"` / `"none"`. |
| `top_n` | `1000` | Number of mismatch details kept when `detail_mode="top"`. |
| `duplicate_key_policy` | `"raise"` | Duplicate-key policy: `"raise"` / `"first"` / `"all"`. |
| `excel_output_path` | `None` | Excel output path; if omitted, `output_dir/Proc_Compare_Report.xlsx` is used. |
| `max_excel_rows` | `100000` | Maximum rows written per Excel sheet. |

## 10. Recommended Practices

- For production wide tables, prefer `key_cols` and do not rely on row order.
- For large CSVs, set `chunk_size` and `n_partitions` first, to avoid reading everything into memory at once.
- For very wide tables, reduce `compare_block_size`; it only affects the column-block memory and does not change tolerances, detail ordering, or result semantics.
- For large tables, use `detail_mode="top"` by default, and turn on `"full"` only when an audit requires it.
- Set sensible per-column tolerances for amounts, probabilities, timestamps, and similar columns.
- Use `ignore_cols` for ETL time, batch IDs, export time, and similar columns.
- If you are comparing two snapshots pulled from ODPS, first write them to local CSV with `ODPSRunner` or `ParallelODPSManager.pull()`, then compare them with `ProcCompareEngine`.
