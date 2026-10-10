# ODPS Data Extraction

SMF talks to Alibaba Cloud MaxCompute (ODPS) through [PyODPS](https://pyodps.readthedocs.io/). `ODPSRunner` runs SQL and
moves DataFrames in and out of tables, `proc_means_odps` computes descriptive statistics inside MaxCompute,
`ParallelODPSManager` pulls and pushes large tables in parallel chunks, and `parse_sql_file` fills SQL templates.

| I want to... | Use | Section |
|---|---|---|
| Run SQL and get a DataFrame, a CSV file, or both | `ODPSRunner.run_sql` | [1](#1-quickstart), [2](#2-run_sql-parameter-semantics) |
| Fill `{placeholders}` in a SQL file | `parse_sql_file` | [4](#4-complete-example-extract-a-sample-to-local-disk) |
| Summarize numeric columns without downloading rows | `proc_means_odps` | [5](#5-proc_means_odps-odps-side-descriptive-statistics) |
| Pull or push a large table in parallel chunks | `ParallelODPSManager` | [6](#6-parallelodpsmanager-concurrent-pullupload) |
| Download a table or partition, create or update a table | `download_table`, `upload_df`, `insert_df` | [8](#8-other-odpsrunner-methods) |
| Pull a table with thousands of columns | `pull_attributes_in_batch` | [9](#9-related-utility-functions) |

## Prerequisites

Install the optional dependency:

```bash
pip install 'supermodelingfactory[odps]'
```

`ODPSRunner()` takes no arguments. It reads its configuration from environment variables when it is created; all four are
required and there are no built-in defaults:

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `ALIBABA_CLOUD_ACCESS_KEY_ID` | Yes | none (`KeyError` if unset) | AccessKey ID |
| `ALIBABA_CLOUD_ACCESS_KEY_SECRET` | Yes | none (`KeyError` if unset) | AccessKey secret |
| `ODPS_PROJECT` | Yes | none (`KeyError` if unset) | Project that unqualified table names refer to |
| `ODPS_ENDPOINT` | Yes | none (`KeyError` if unset) | MaxCompute endpoint of your project's region, from Alibaba Cloud's endpoint list (a VPC endpoint from inside Alibaba Cloud's network, the public one elsewhere) |

```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID="<your-access-key-id>"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="<your-access-key-secret>"
export ODPS_PROJECT="<your-project>"
export ODPS_ENDPOINT="https://service.<region>.maxcompute.aliyun.com/api"    # from Alibaba Cloud's endpoint list
```

Never write AccessKeys into source code, notebooks, or documentation, and keep any `.env` file out of version control.
The [FAQ](../faq.md) shows how to load them from a shared `.env` file. `ODPSRunner` reads only an AccessKey pair; it has
no argument for an STS token.

Up to 0.9.0, `ODPS_PROJECT` and `ODPS_ENDPOINT` were optional and fell back to a project and an endpoint of the
author's environment; since 0.9.1 a missing one raises `KeyError` like the AccessKey variables.

!!! note "Which snippets need a live project"

    Every snippet that creates an `ODPSRunner`, directly or through `proc_means_odps`, `ParallelODPSManager`, or
    `pull_attributes_in_batch`, connects to MaxCompute. Table names such as `my_project.loan_sample` are placeholders for
    your own tables. `parse_sql_file`, `ODPSRunner.cre_table_schema`, and `ParallelODPSConfig` need no connection and
    run offline. The snippets share one Python session: `odps` is created in the quickstart and reused below.

## 1. Quickstart

```python
# check: skip   (needs MaxCompute credentials)
import os

from Modeling_Tool import ODPSRunner

os.makedirs("data", exist_ok=True)
odps = ODPSRunner()                      # reads the credentials from the environment

# 1) Pull a query result into a DataFrame
df = odps.run_sql("SELECT * FROM my_project.loan_sample LIMIT 1000")
print(df.shape)

# 2) Write the result to a CSV file and return no DataFrame
_ = odps.run_sql(
    "SELECT * FROM my_project.loan_sample",
    to_df=False,
    csv_path="data/loan_sample.csv",
)

# 3) Get the DataFrame and the CSV file
df = odps.run_sql(
    "SELECT * FROM my_project.loan_sample",
    csv_path="data/loan_sample.csv",
)
```

`ODPSRunner` logs the SQL text, timestamps, and duration of every call with `logging.info`. Importing SMF configures the
root logger at `INFO`; to silence these lines, call `logging.getLogger().setLevel(logging.WARNING)` after the import.

## 2. `run_sql` Parameter Semantics

`run_sql(sql, to_df=True, n_process=1, csv_path=None)` runs one SQL statement. `to_df` and `csv_path` are independent:
setting either one triggers the download.

| `to_df` | `csv_path` | Downloads the result | Writes a CSV | Returns |
|---|---|---|---|---|
| `True` (default) | `None` (default) | Yes | No | The result DataFrame |
| `True` | A path | Yes | Yes | The result DataFrame |
| `False` | `None` | No: the statement only runs | No | An empty DataFrame |
| `False` | A path | Yes | Yes | An empty DataFrame |

Use `to_df=False` without `csv_path` for statements that return no rows, such as `CREATE TABLE ... AS SELECT` and
`INSERT`. `n_process` is passed to PyODPS's `to_pandas` and enables a multi-process download when it is above 1. The CSV is
written without the pandas index.

!!! warning "The result always passes through memory"

    `to_df=False` only skips returning the DataFrame. The result is still downloaded into memory in one piece and then
    written to the CSV, so peak memory is that of the full result. For results that do not fit in memory, use
    [`ParallelODPSManager.pull`](#6-parallelodpsmanager-concurrent-pullupload).

!!! warning "Create the CSV folder first"

    `run_sql` does not create the folder of `csv_path`. Writing the CSV is part of the download retry loop (see
    [Retry Policy](#31-retry-policy)), so a missing folder causes six downloads and ends in a `SystemError`; the real cause
    appears only in the log.

## 3. Internals

### 3.1 Retry Policy

- **Execution** (`execute_sql`) runs once and is never resubmitted, so a retry cannot run, and bill, a statement twice.
  An error such as a syntax error, a missing permission, or a quota limit propagates unchanged.
- **Download and CSV write** (`to_pandas`, then `to_csv`) make up to six attempts in total. After the sixth failure
  `run_sql` raises `SystemError`; the causes are in the log lines `download failed [i/6]`.
- PyODPS also retries its HTTP requests on its own (`options.retry_times = 6`, which `ODPSRunner` sets).

### 3.2 Wide-Table Schema Patch

When a result has more than 200 columns, the PyODPS tunnel puts every column name in the request URL and the server can
answer `HTTP 414 (Request-URI Too Long)`. `ODPSRunner` then temporarily replaces
`InstanceDownloadSession._build_input_stream` so that the request carries no column list and the server returns all
columns. The patch is applied and removed automatically. A lock and a reference count keep concurrent downloads safe: the
original method comes back only after the last download has finished.

### 3.3 Connection Configuration

`ODPSRunner.__init__` builds `odps.ODPS(access_id, secret, project, endpoint=...)` from the four environment variables in
[Prerequisites](#prerequisites). It then sets four **process-wide** PyODPS options, which apply to every PyODPS client in
the process: `options.retry_times = 6`, `options.pool_maxsize = 200`, `options.connect_timeout = 3600`, and
`options.read_timeout = 3600`.

The PyODPS client is available as `odps.o`. Use it for anything SMF does not wrap, for example
`odps.o.list_instances(status="running")` to see the running jobs of your project.

## 4. Complete Example: Extract a Sample to Local Disk

```python
# check: skip   (needs MaxCompute credentials)
from pathlib import Path

from Modeling_Tool import ODPSRunner
from Modeling_Tool.Core.utils import parse_sql_file, mkdir_if_not_exist

# 1) A SQL template. parse_sql_file removes the comments and fills the {placeholders}.
Path("sql").mkdir(exist_ok=True)
Path("sql/00_sample.sql").write_text(
    "-- development sample\n"
    "SELECT flow_id, {tgt_name}, {varlist}\n"
    "FROM my_project.loan_sample\n"
    "WHERE dt = '{dt}'\n"
)
sql = parse_sql_file(
    sql_path="sql/00_sample.sql",
    tgt_name="bad_flag",
    varlist="score_b, income, age, n_overdue",
    dt="2026-01-01",
)
print(sql)

# 2) The output folder
mkdir_if_not_exist("data")
csv_path = Path("data") / "sample_drv.csv"

# 3) Run the SQL and write only the CSV
odps = ODPSRunner()
_ = odps.run_sql(sql, to_df=False, csv_path=str(csv_path), n_process=4)
print(f"Sample extracted: {csv_path}")
```

`parse_sql_file(sql_path=None, sql_query=None, split=False, format_select=False, **kwargs)`:

- Give exactly one of `sql_path` (a file) and `sql_query` (a string); otherwise it raises `AttributeError`.
- Every keyword argument fills the `{name}` placeholder of the same name. Values must be strings, so write
  `n="20"`, not `n=20`. A placeholder without an argument stays in the SQL and triggers a `UserWarning`.
- It removes `--` and `/* */` comments and returns one string that ends with `;`. With several statements the string joins
  them with `; `, and `split=True` returns a list instead. `run_sql` accepts one statement per call.
- `format_select=True` reformats the `SELECT` list with one column per line.
- `mkdir_if_not_exist(folder_path)` creates the folder and returns `0`, or returns `1` if it already exists.

## 5. `proc_means_odps`: ODPS-Side Descriptive Statistics

`proc_means_odps()` computes descriptive statistics for numeric columns inside MaxCompute and downloads only the small
aggregated result as a DataFrame. Use it for tables with very many rows or columns, where you do not want to `SELECT *`
first.

```python
# check: skip   (needs MaxCompute credentials)
from Modeling_Tool import proc_means_odps

summary = proc_means_odps(
    input_table_name="my_project.loan_sample",
    skip_cols=["flow_id", "bad_flag"],
    batch_size=50,
)
print(summary.head())
```

The result has one row per variable:

```text
attribute, N_ALL, N, MEAN, STD, MIN,
Q5, Q15, Q25, Q50, Q75, Q95, Q99, MAX, MISSING_RATE
```

`N_ALL` is the number of rows, `N` the number of valid values, and `MISSING_RATE = 1 - N / N_ALL`. `STD` is the sample
standard deviation and is `NaN` when `N < 2`.

### 5.1 Statistics by Group

`group` takes a column name or a list of columns. The group columns come first in the result, and the rows are sorted by
the group values, then by variable.

```python
# check: skip   (needs MaxCompute credentials)
grouped = proc_means_odps(
    input_table_name="my_project.loan_sample",
    select_cols=["age", "credit_limit", "income"],
    group=["apply_month", "channel"],
    where_clause="dt >= '2026-01-01'",
)
print(grouped.head())
```

By default (`include_missing_group=False`) rows where any group column is NULL are left out, as in a pandas `groupby`. Set
it to `True` to keep them as their own group.

### 5.2 Column Selection and Batches

- `select_cols=None` analyzes every numeric ordinary column. String columns and partition columns are skipped, and the
  target column is analyzed too if it is numeric, so list it in `skip_cols`.
- `select_cols=[...]` analyzes only those columns. A non-numeric column raises `ValueError`.
- `skip_cols=[...]` removes columns from the candidates and wins over `select_cols`. Every listed column must exist.
- `group` columns are never analyzed as variables.
- `batch_size=50` is the number of variables per aggregation statement. It limits the width of the SQL; it never splits
  or downloads source rows.

Each batch scans the table once and computes `COUNT`, `AVG`, `STDDEV_SAMP`, `MIN`, the percentiles, and `MAX` for all its
variables together, then the small wide result is reshaped into a long table locally. If any batch fails,
`proc_means_odps` raises a `RuntimeError` that names the batch and its first and last variable, and it writes neither the
CSV nor the result table.

### 5.3 Quantiles and Special Missing Values

By default the percentiles are approximate, which suits large tables:

```python
# check: skip   (needs MaxCompute credentials)
approx = proc_means_odps(
    "my_project.loan_sample",
    q=[0.05, 0.5, 0.95],
    quantile_method="approx",
    percentile_accuracy=10000,
)
```

For linear interpolation as in pandas, ask for the exact mode:

```python
# check: skip   (needs MaxCompute credentials)
exact = proc_means_odps(
    "my_project.loan_sample",
    q=[0.05, 0.5, 0.95],
    quantile_method="exact",
)
```

`"approx"` uses MaxCompute's `PERCENTILE_APPROX` and `"exact"` uses `PERCENTILE_CONT`, which can need much more compute on
very large tables or on many groups.

The quantile columns are named `Q<percent>`: `0.05` becomes `Q5`. The percent is `int(q * 100)`, which truncates, so
floating-point error lowers a few labels by one (`0.29`, `0.57`, and `0.58` become `Q28`, `Q56`, and `Q57`). Each quantile
must map to a distinct whole percent, so values such as `0.001` or `0.505` raise `ValueError`.

Special missing values are turned into NULL before the aggregation, so they count neither in `N` nor in the statistics:

```python
# check: skip   (needs MaxCompute credentials)
summary = proc_means_odps(
    "my_project.loan_sample",
    select_cols=["age", "income"],
    spec_missing_value={
        "age": [-1, -999],
        "income": -999,
    },
)
```

`spec_missing_value` is a number, a list of numbers (applied to every analyzed column), or a dictionary from column to a
number or list. Dictionary keys must be analyzed columns.

### 5.4 CSV and ODPS Output

By default the function only returns a DataFrame and creates no file or table:

```python
# check: skip   (needs MaxCompute credentials)
summary = proc_means_odps("my_project.loan_sample")
```

Write a CSV, without the pandas index; missing parent folders are created:

```python
# check: skip   (needs MaxCompute credentials)
summary = proc_means_odps(
    "my_project.loan_sample",
    output_csv="output/feature_means.csv",
)
```

Write the result back to MaxCompute. The mode is required, so a production table is never overwritten by mistake:

```python
# check: skip   (needs MaxCompute credentials)
summary = proc_means_odps(
    "my_project.loan_sample",
    output_table_name="my_project.feature_means_report",
    output_table_mode="overwrite",           # or "append"
)
```

- `"overwrite"` creates the table, or replaces it atomically through `ODPSRunner.upload_df(..., atomic=True)`, with a schema
  inferred from the result. It adds no `py_inserttime` column.
- `"append"` needs an existing table whose column names, order, and types match the result; otherwise it raises
  `ValueError`.
- The output table must differ from the input table. `"append"` needs an unpartitioned target, and `"overwrite"` always
  creates an unpartitioned table.

### 5.5 Parameter Table

`proc_means_odps(input_table_name, skip_cols=None, select_cols=None, batch_size=50, group=None, *, q=None,
quantile_method='approx', percentile_accuracy=10000, where_clause=None, spec_missing_value=None,
include_missing_group=False, sqlrunner=None, output_csv=None, output_table_name=None, output_table_mode=None)`.
Every parameter after `group` is keyword-only.

| Parameter | Default | Description |
|---|---:|---|
| `input_table_name` | Required | Table identifier: `table`, `project.table`, or `project.schema.table`. Letters, digits, and underscores only |
| `skip_cols` | `None` | Columns excluded from the variables |
| `select_cols` | `None` | Explicit variables; if omitted, the numeric ordinary columns are used |
| `batch_size` | `50` | Variables per aggregation statement |
| `group` | `None` | One or several grouping columns; omitted means global statistics |
| `q` | `None` | Quantile points in `[0, 1]`, strictly increasing. `None` means `[0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]` |
| `quantile_method` | `"approx"` | `"approx"` or `"exact"` |
| `percentile_accuracy` | `10000` | Accuracy argument of `PERCENTILE_APPROX` |
| `where_clause` | `None` | One SQL condition, wrapped in parentheses and combined with `AND`; useful for partition pruning. It must not contain `;` |
| `spec_missing_value` | `None` | Numeric sentinel(s) treated as missing |
| `include_missing_group` | `False` | Keep the combinations where a group column is NULL |
| `sqlrunner` | `None` | An `ODPSRunner` to reuse; if omitted, one is created from the environment |
| `output_csv` | `None` | Path of an optional CSV file |
| `output_table_name` | `None` | Name of an optional result table |
| `output_table_mode` | `None` | `"overwrite"` or `"append"`; required with `output_table_name` |

`proc_means_odps` analyzes numeric variables only. It does not compute `UNIQUE`, `TOP`, or `FREQ` for categorical columns.

## 6. `ParallelODPSManager` Concurrent Pull/Upload

`ParallelODPSManager(config, odps_runner=None)` combines `ODPSRunner` with `ParallelApplyEngine` to process a large table
chunk by chunk. If you pass no `odps_runner`, it creates one. `ParallelODPSPuller` is an alias of the same class.

- `pull()` splits one query into chunks by hash of `unique_key` or, without a key, by a ROW_NUMBER staging table. It runs
  the chunks concurrently and merges them into one local CSV.
- `push()` takes a DataFrame or a CSV path, splits it by rows, uploads every chunk to a temporary ODPS table, and writes
  all chunks into the target table with `UNION ALL`.

`ParallelODPSConfig` fields:

| Field | Default | Meaning |
|---|---|---|
| `unique_key` | `None` | Column used for hash bucketing in `pull()` |
| `chunk_size` | `None` | Rows per chunk. Cannot be combined with `n_chunks` |
| `n_chunks` | `None` | Number of chunks |
| `n_jobs` | `3` | Number of parallel workers |
| `backend` | `"thread"` | `"thread"`, `"process"`, or `"sequential"` |
| `pull_split_strategy` | `"auto"` | `"auto"` (hash with a `unique_key`, ROW_NUMBER without), `"hash"`, or `"row_number"` |
| `row_number_order_by` | `None` | `ORDER BY` expression of the ROW_NUMBER staging table; `None` means `ORDER BY 1` |
| `row_number_col` | `"__smf_parallel_odps_rn__"` | Name of the helper row-number column |
| `validate_unique_key` | `True` | In hash mode, run a probe query before pulling |
| `chunk_filter_key` | `"chunk_filter"` | Name of the placeholder in the SQL template |
| `tmp_dir` | `Path("data/_chunks")` | Local folder for chunk files |
| `tmp_table_prefix` | `"tmp_parallel_odps"` | Prefix of temporary ODPS tables |
| `cleanup_tmp` | `True` | Drop temporary tables when the run ends |
| `keep_tmp_on_error` | `False` | Keep temporary tables when the run fails |

`pull()` and `push()` need `n_chunks` or `chunk_size`, and they use the same configuration.

### 6.1 Concurrent pull

`pull(sql_path, out_path, count_query=None, **template_kwargs)` reads a SQL template that **must contain
`{chunk_filter}`**, placed in the `WHERE` clause of the table you split. A template without it raises `ValueError` before
any query runs, which prevents every chunk from pulling the whole table. Extra keyword arguments fill other placeholders
of the template and must be strings.

#### Hash Bucketing: Recommended When You Have a Stable Key

With a `unique_key` and `pull_split_strategy="auto"`, each chunk receives this filter:

```sql
ABS(HASH(flow_id)) % 20 = <chunk_id>
```

Before the chunks run, SMF sends one probe query to check that `unique_key` is visible in the SQL scope. If the probe
fails, `pull()` raises `ValueError` that quotes the first line of the ODPS error and stops. Set `validate_unique_key=False`
to skip the probe.

```python
# check: skip   (needs MaxCompute credentials)
from pathlib import Path

from Modeling_Tool import ParallelODPSConfig, ParallelODPSManager

Path("sql").mkdir(exist_ok=True)
Path("sql/pull_sample.sql").write_text(
    "SELECT flow_id, score_b, apply_month\n"
    "FROM my_project.loan_sample\n"
    "WHERE 1 = 1\n"
    "  AND {chunk_filter}\n"
)

manager = ParallelODPSManager(
    ParallelODPSConfig(
        unique_key="flow_id",
        pull_split_strategy="auto",   # auto + unique_key => hash
        n_chunks=20,
        n_jobs=5,
        backend="thread",
        tmp_dir="data/_chunks",
    )
)

summary = manager.pull(
    sql_path="sql/pull_sample.sql",
    out_path="data/sample.csv",
)
print(summary)
```

`pull()` returns a dictionary:

| Key | Value |
|---|---|
| `pull_strategy` | `"hash"` or `"row_number"` |
| `staging_table` | Name of the ROW_NUMBER staging table, or `None` in hash mode |
| `n_chunks` | Number of chunks |
| `total_rows` | Rows in the merged CSV |
| `out_path` | Path of the merged CSV |
| `per_chunk_rows` | Rows of each chunk, in chunk order |

The CSV holds the chunks in chunk order, with one header. If you pass `chunk_size` instead of `n_chunks`, `pull()` first
runs a count query to derive the number of chunks. For a complex join, pass a cheaper `count_query` yourself.

If a chunk fails, `pull()` raises `RuntimeError` (`1/4 ODPS pull chunks failed`, followed by the errors), writes no CSV,
and leaves the files of the finished chunks in `tmp_dir`.

#### ROW_NUMBER Bucketing: Used Automatically When There Is No `unique_key`

With `unique_key=None` and `pull_split_strategy="auto"`, SMF:

1. Renders the SQL with `{chunk_filter}` set to `1=1`.
2. Materializes the result in an ODPS staging table with an extra row-number column.
3. Pulls each chunk with `WHERE (row_number_col - 1) % n_chunks = chunk_id`.
4. Drops the row-number column before writing the CSV.
5. Drops the staging table according to `cleanup_tmp` and `keep_tmp_on_error`.

```python
# check: skip   (needs MaxCompute credentials)
manager = ParallelODPSManager(
    ParallelODPSConfig(
        unique_key=None,
        pull_split_strategy="auto",   # auto + no unique_key => row_number
        n_chunks=20,
        n_jobs=5,
        backend="thread",
        tmp_table_prefix="tmp_parallel_odps",
    )
)

summary = manager.pull(
    sql_path="sql/pull_sample.sql",
    out_path="data/sample_rn.csv",
)
print(summary["staging_table"])
```

The default row-number expression is `ROW_NUMBER() OVER (ORDER BY 1)`. For a reproducible numbering, set
`row_number_order_by`:

```python
# check: skip   (continues a snippet that needs MaxCompute credentials)
config = ParallelODPSConfig(
    unique_key=None,
    pull_split_strategy="row_number",
    row_number_order_by="apply_month, flow_id",
    chunk_size=500000,
)
```

ROW_NUMBER mode creates a temporary table, so it costs more compute and storage than hash mode. Prefer `unique_key` when
you have a stable, evenly distributed key.

### 6.2 Concurrent push

`push(data, target_table, write_mode=None)` accepts a DataFrame or a CSV path. `write_mode` is required, `"overwrite"` or
`"append"`, so a table is never replaced by accident.

```python
# check: skip   (continues a snippet that needs MaxCompute credentials or local files)
import pandas as pd

scores = pd.read_csv("data/sample.csv")          # the file written by pull() above

summary = manager.push(
    data=scores,
    target_table="my_project.loan_scores",
    write_mode="overwrite",
)
print(summary["total_rows"], summary["n_chunks"])

summary = manager.push(
    data="data/sample.csv",                      # a CSV path is read in chunks
    target_table="my_project.loan_scores",
    write_mode="append",
)
```

`push()` returns a dictionary with `n_chunks`, `total_rows`, `target_table`, `write_mode`, `tmp_tables`, `per_chunk_rows`,
`union_sql`, and `final_sql`. Execution flow:

1. A DataFrame is split by rows. A CSV is read with `pd.read_csv(..., chunksize=...)` and written to temporary local chunk
   files.
2. Each chunk is uploaded with `ODPSRunner.upload_df` to its own ODPS table, such as `tmp_parallel_odps_<run_id>_0000`.
   `upload_df` returns only after the table is visible, so the final `UNION ALL` never runs too early and you need no
   `sleep` or polling.
3. All temporary tables are combined into the target table with `UNION ALL`.
4. Temporary tables are dropped after success and, by default, after a failure too. With `keep_tmp_on_error=True` they stay
   after a failure.

| `write_mode` | Final statement |
|---|---|
| `"overwrite"` | Drops the target table, then runs `CREATE TABLE target AS SELECT ... UNION ALL ...` |
| `"append"` | Runs `INSERT INTO TABLE target SELECT ... UNION ALL ...` |

!!! warning "`overwrite` is not atomic"

    The target table is dropped before the final `CREATE TABLE ... AS`. If that statement fails, the target is gone. Set
    `keep_tmp_on_error=True` to keep the temporary tables, which hold the data, for recovery.

!!! warning "The target table gets an extra `py_inserttime` column"

    The chunks are uploaded without an explicit schema, so each temporary table, and therefore the table created by
    `"overwrite"`, ends with a string column `py_inserttime` (the upload time). For `"append"`, the existing target must
    already have the DataFrame's columns in the same order, followed by `py_inserttime`; a table without it makes the
    `INSERT` fail with a column-count error.

`push()` does not support partitioned target tables.

### 6.3 Backend Recommendations

| `backend` | Recommendation |
|---|---|
| `"thread"` | The default choice for ODPS I/O. The workers share the manager's `ODPSRunner` and its connection pool, and the wide-table patch is thread-safe |
| `"sequential"` | One chunk after the other. Use it to debug chunk SQL, upload logic, and cleanup |
| `"process"` | Every chunk task creates its own `ODPSRunner()`, so the credentials must be in the environment of the worker processes. Stronger isolation at a higher overhead |

## 7. Common Pitfalls

### Pitfall 1: `to_df=False` does not save memory while downloading

`run_sql` with `to_df=False` and a `csv_path` still holds the whole result in memory until the CSV is written. Use
`ParallelODPSManager.pull` or narrower queries for large results. See [section 2](#2-run_sql-parameter-semantics).

### Pitfall 2: A table with more than 200 columns

`ODPSRunner` applies the wide-schema patch by itself, so `HTTP 414 (Request-URI Too Long)` does not reach you. See
[section 3.2](#32-wide-table-schema-patch).

### Pitfall 3: Every `run_sql` call submits a new job

`run_sql` submits the SQL again each time, even if nothing changed. Cache results you reuse:

```python
# check: skip   (continues a snippet that needs MaxCompute credentials or local files)
from pathlib import Path

import pandas as pd

csv_path = Path("data/loan_sample.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    df = odps.run_sql("SELECT * FROM my_project.loan_sample", csv_path=str(csv_path))
```

A CSV loses dtypes such as dates; `df.to_pickle("data/loan_sample.pkl")` and `pd.read_pickle` keep them.

### Pitfall 4: `upload_df` replaces the table and `insert_df` overwrites by default

`upload_df` always replaces the whole table, even with `partition=`. `insert_df` replaces the table or partition contents
by default (`overwrite=True`). See [section 8](#8-other-odpsrunner-methods).

### Pitfall 5: Long queries block the process

`run_sql` returns when the query has finished and the result is downloaded. Run jobs that take minutes or hours in the
background, or from a scheduler, and write their output to a log file. For example, save the code of
[section 4](#4-complete-example-extract-a-sample-to-local-disk) as `extract_sample.py` and run:

```bash
nohup python -u extract_sample.py > "odps_$(date +%s).log" 2>&1 &
```

`ODPSRunner` does not return the job id. To see the running jobs of your project, use
`odps.o.list_instances(status="running")`.

### Error lookup

| Message | Cause | Fix |
|---|---|---|
| `KeyError: 'ALIBABA_CLOUD_ACCESS_KEY_ID'` | The credentials are not in the environment when `ODPSRunner()` is created | Export the variables from [Prerequisites](#prerequisites) before you start Python |
| `ModuleNotFoundError: No module named 'odps'` | PyODPS is not installed | `pip install 'supermodelingfactory[odps]'` |
| `SystemError: break: ...` from `run_sql` | Six download attempts failed | Read the log lines `download failed [i/6]`. Typical causes: a missing folder for `csv_path`, or a network or endpoint problem |
| `ValueError: pull SQL template must contain {chunk_filter}` | The SQL file has no `{chunk_filter}` placeholder | Add it to the `WHERE` clause |
| `ValueError: unique_key validation failed for pull SQL: ...` | The probe query failed: the key is not visible in the SQL scope, or the SQL itself is wrong | Fix the key or the SQL; the message quotes the first line of the ODPS error |
| `ValueError: chunk_size or n_chunks is required ...` | Neither is set in `ParallelODPSConfig` | Set one of them |
| `ValueError: write_mode is required ...` | `push()` was called without a valid `write_mode` | Pass `"overwrite"` or `"append"` |
| `UserWarning: Missing argument(s) ... in the given SQL file` | A `{placeholder}` got no keyword argument | Pass it to `parse_sql_file` |
| `TypeError: replace() argument 2 must be str, not int` | A template value is not a string | Pass `"20"` instead of `20` |
| `ValueError: The values set to records are against the schema, expect len N, got len M` | The DataFrame has other columns than the table, for example it still contains the partition column | Pass exactly the table's non-partition columns, in order |

## 8. Other `ODPSRunner` Methods

### `download_table(table_name, partition=None, n_process=1, csv_path=None)`

Reads a whole table, or one partition, through the table tunnel instead of running a query:

```python
# check: skip   (continues a snippet that needs MaxCompute credentials or local files)
df = odps.download_table(
    "my_project.loan_sample",
    partition="dt=2026-01-01",
    csv_path="data/loan_sample_0101.csv",
)
print(df.shape)
```

`partition` is a partition spec string such as `"dt=2026-01-01"` (several levels: `"dt=2026-01-01,hr=01"`) or a
dictionary such as `{"dt": "2026-01-01"}`. `n_process` above 1 turns on a parallel download; in the current
implementation that always uses 10 processes. `csv_path` is written without the index.

### `upload_df(df, table_name, table_schema=None, partition=None, atomic=True)`

Creates a table from a DataFrame and writes all rows. **An existing table of the same name is replaced.**

```python
# check: skip   (continues a snippet that needs MaxCompute credentials or local files)
import pandas as pd

scores = pd.DataFrame({
    "flow_id": ["F001", "F002", "F003"],
    "score": [0.12, 0.34, None],
    "scored_at": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
})

schema = ODPSRunner.cre_table_schema(scores)
odps.upload_df(scores, "my_project.my_scores", table_schema=schema)
```

- `atomic=True` (default) writes the data to a temporary table and renames it over the target, so a failure before the
  swap leaves the original table untouched. `atomic=False` drops the old table first, so a failure leaves no table.
- Without `table_schema` the schema is inferred with `cre_table_schema`, and a string column `py_inserttime` (the upload
  time) is **appended to the data**. Pass a schema to avoid it.
- `np.nan`, `pd.NA`, and `NaT` become `NULL` in a copy of the records. The DataFrame you pass is not modified, and there is
  no need to call `npnan2none()` first.
- For a partitioned table, build the schema from a frame that contains the partition column and upload the data without
  it, because the partition value comes from `partition`:

```python
# check: skip   (continues a snippet that needs MaxCompute credentials or local files)
schema = ODPSRunner.cre_table_schema(scores.assign(dt="2026-01-01"), partition_name="dt")
odps.upload_df(scores, "my_project.my_scores_by_day", table_schema=schema, partition="dt=2026-01-01")
```

`upload_df` replaces the whole table also with `partition=`: afterwards the table holds only that partition. To add or
replace one partition of an existing table, use `insert_df`.

### `insert_df(df, table_name, overwrite=True, partition=None, atomic=True)`

Writes a DataFrame into an **existing** table. The DataFrame's columns must match the table's non-partition columns in
number, order, and type, because values are written by position.

```python
# check: skip   (continues a snippet that needs MaxCompute credentials or local files)
# Append to a partition of the table created above; other partitions stay untouched
odps.insert_df(scores, "my_project.my_scores_by_day", overwrite=False, partition="dt=2026-01-02")

# Replace the contents of one partition (the default overwrite=True)
odps.insert_df(scores, "my_project.my_scores_by_day", partition="dt=2026-01-02")
```

- `overwrite=True` is the default and **replaces** the data: without `partition` the table is truncated first (this path is
  not atomic); with `partition` that partition is replaced, atomically through a staging partition when `atomic=True`.
  `overwrite=False` appends.
- If the table has a `py_inserttime` column and the DataFrame does not, the column is filled with the insert time.

### `cre_table_schema(df, partition_name=None)` (static method)

Infers an ODPS schema from a DataFrame:

```python
# check: skip   (continues a snippet that needs MaxCompute credentials)
schema = ODPSRunner.cre_table_schema(scores)
print([(column.name, str(column.type)) for column in schema.columns])
```

| pandas dtype | ODPS type |
|---|---|
| bool | `boolean` |
| any integer, including nullable `Int64` | `bigint` |
| `float32` | `float` |
| other floats, including nullable `Float64` | `double` |
| datetime | `datetime` |
| string, `object`, category | `string` |

Complex and timedelta columns, and any other unsupported dtype, raise `TypeError` instead of being converted silently.
`partition_name` marks the column of that name as a partition column; if `df` has no such column it has no effect.

## 9. Related Utility Functions

`pull_attributes_in_batch(table_name, varlist, batch_num=6, unikey='flow_id', main_info_select=['*'], add_query='')` pulls a
table whose attribute columns are too many for one query. It creates its own `ODPSRunner()`.

```python
# check: skip   (needs MaxCompute credentials)
from Modeling_Tool import pull_attributes_in_batch

varlist = [f"x{i:03d}" for i in range(1, 1001)]      # the attribute columns of the table

result_df = pull_attributes_in_batch(
    table_name="my_project.wide_attrs",
    varlist=varlist,
    unikey="flow_id",
)
print(result_df.shape)
```

It splits `varlist` into about six vertical batches and runs `SELECT <unikey>, <batch columns> FROM <table> <add_query>` for
each. A last query, `SELECT <main_info_select> EXCEPT (<varlist>) FROM <table> <add_query>`, reads the remaining columns.
The pieces are merged on `unikey` with an inner join.

- `varlist` needs at least six names; fewer raise `ValueError`.
- `batch_num` is accepted but ignored: the number of batches is fixed at about six. Do not rely on it.
- `unikey` must be unique per row and must be spelled as in the downloaded result; MaxCompute returns lower-case column
  names, so write `flow_id`.
- Keep `main_info_select` at `['*']`: the query uses MaxCompute's `SELECT * EXCEPT (...)` form.
- `add_query` is text appended after `FROM <table>` in every query, for example `"WHERE dt = '2026-01-01'"`.

## 10. Next Steps

- Want the complete modeling pipeline? Read [End-to-End Modeling Pipeline](../pipeline.md)
- Want downstream processing such as WOE / IV? Read [WOE Encoding](woe.md) and [Feature Screening](feature.md)
- Want specific API signatures? Visit [API Reference → Core](../api/core.md)
