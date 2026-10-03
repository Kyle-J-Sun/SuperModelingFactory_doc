# ODPS Data Extraction

SuperModelingFactory's [`Modeling_Tool.Core.ODPS_Tool`](../api/core.md) provides `ODPSRunner` — a wrapper around SQL execution, data download, and upload for Alibaba Cloud MaxCompute (ODPS).

## 1. Quickstart

```python
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner

odps = ODPSRunner()

# 1) Pull data into a DataFrame
df = odps.run_sql("SELECT * FROM mex_anls.drv LIMIT 1000")
print(df.head())

# 2) Pull data straight to a CSV on disk (recommended for large tables)
_ = odps.run_sql(
    "SELECT * FROM mex_anls.drv",
    to_df=False,
    csv_path="/data/drv.csv",
)

# 3) Get both the DataFrame and the file on disk
df = odps.run_sql(
    "SELECT * FROM mex_anls.drv",
    csv_path="/data/drv.csv",
)
```

## 2. `run_sql` Parameter Semantics

| `to_df` | `csv_path` | Behavior | Returns |
|---------|-----------|------|------|
| `True` | `None` | Download into memory | DataFrame |
| `True` | Set | Download + write CSV | DataFrame |
| `False` | `None` | **No download** (for DDL/INSERT) | Empty DataFrame |
| `False` | Set | Download + write CSV (frees memory) | Empty DataFrame |

!!! warning "The counterintuitive `to_df=False + csv_path`"

    Historically, `to_df=False` caused `csv_path` to be **silently ignored** as well — see [§7 Historical pitfalls](#7-common-pitfalls).
    The current version has fixed this: `to_df` and `csv_path` are independent of each other, and setting either one triggers the download.

## 3. Internals

### 3.1 Retry Policy

- **Execution stage** (`execute_sql`) — runs only once, with no retry (to avoid being billed twice)
- **Download stage** (`to_pandas` + `to_csv`) — up to 6 retries, suited to network jitter

### 3.2 Wide-Table Schema Patch

When a SQL result has more than 200 columns, `ODPSRunner` automatically enables a thread-safe wide-schema patch:

```text
HTTP 414 (URI Too Long) ← the original request carries all column names as URL query parameters
                         after the patch → the columns parameter is removed, and the server returns everything
```

The patch is restored automatically and is transparent to the caller. When several threads download wide tables at the same time, an internal lock and reference count ensure the original ODPS method is restored only after the last download task exits, so concurrent patch/unpatch calls cannot overwrite each other.

### 3.3 Connection Configuration in `__init__`

```python
class ODPSRunner:
    def __init__(self):
        self.o = ODPS(
            "<ALIBABA_CLOUD_ACCESS_KEY_ID>",      # AccessKey ID
            "<ALIBABA_CLOUD_ACCESS_KEY_SECRET>",  # AccessKey Secret
            "mex_anls",                          # ← default project
            endpoint="https://service.ap-southeast-1-vpc.maxcompute.aliyun-inc.com/api",
        )
        options.retry_times = 6
        options.pool_maxsize = 200
        options.connect_timeout = 3600
        options.read_timeout = 3600
```

!!! warning "Credential configuration"

    The current implementation reads the Alibaba Cloud credentials from environment variables. Never write real credentials into source code or documentation.
    
    1. Inject through environment variables:
       ```python
       self.o = ODPS(
           os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
           os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
           os.environ["ODPS_PROJECT"],
           endpoint=os.environ["ODPS_ENDPOINT"],
       )
       ```
    2. Add `.env` to `.gitignore` to avoid leaking it
    3. In the long run, consider switching to a RAM Role / STS Token

## 4. Complete Example — Extract a Sample to Local Disk

```python
from pathlib import Path
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
from Modeling_Tool.Core.utils import parse_sql_file, mkdir_if_not_exist

odps = ODPSRunner()

# 1) Render the SQL template
sql = parse_sql_file(
    sql_path="sql/00_sample.sql",
    tgt_name="IS_DPD7",
    varlist="score_b, income, age, n_overdue",  # placeholder substitution
)

# 2) Output directory
out_dir = Path("data/")
mkdir_if_not_exist(str(out_dir))
csv_path = out_dir / "sample_drv.csv"

# 3) Run the SQL, write only the CSV, use no memory
_ = odps.run_sql(sql, to_df=False, csv_path=str(csv_path), n_process=4)
print(f"Sample extracted: {csv_path}")
```

## 5. `proc_means_odps`: ODPS-Side Descriptive Statistics

`proc_means_odps()` computes descriptive statistics for numeric variables directly in MaxCompute, and downloads only the small aggregated result as a pandas DataFrame. It suits scenarios with very many rows or very many features, where you do not want to `SELECT *` the full table first.

```python
from Modeling_Tool import proc_means_odps

summary = proc_means_odps(
    input_table_name="mex_anls.feature_wide_table",
    skip_cols=["flow_id", "badflag"],
    batch_size=50,
)
```

The default output has one row per variable:

```text
attribute, N_ALL, N, MEAN, STD, MIN,
Q5, Q15, Q25, Q50, Q75, Q95, Q99, MAX, MISSING_RATE
```

### 5.1 Statistics by Group

`group` accepts a string or a list of columns, and the output structure matches the numeric `proc_means_by_grp()`:

```python
grouped = proc_means_odps(
    input_table_name="mex_anls.feature_wide_table",
    select_cols=["age", "credit_limit", "income"],
    group=["apply_month", "channel"],
    where_clause="dt >= '2026-01-01'",
)
```

The default is `include_missing_group=False`: records where any group column is NULL do not enter the grouped result, consistent with the default of pandas `groupby`. Set it to `True` to keep NULL groups.

### 5.2 Column Selection and Batches

- `select_cols=None`: automatically choose numeric columns from the ordinary table columns; partition columns are not analyzed automatically.
- `select_cols=[...]`: analyze only the specified numeric columns; explicitly passing a non-numeric column raises an error.
- `skip_cols=[...]`: remove columns from the candidates; when it overlaps with `select_cols`, `skip_cols` takes precedence.
- `group` columns take part only in grouping and are not analyzed again as metric variables.
- `batch_size=50`: each ODPS aggregation SQL handles 50 features. It limits the SQL width; it is not a row-level chunk and never downloads source data rows.

Each feature batch scans the source table only once, computing `COUNT/AVG/STDDEV_SAMP/MIN/PERCENTILE/MAX` for all variables in that batch at the same time, and then converts the small wide aggregated result into a long table locally. If any batch fails, the function raises immediately, and neither the CSV nor the ODPS result table gets a half-finished output.

### 5.3 Quantiles and Special Missing Values

By default, approximate quantiles suited to large tables are used:

```python
approx = proc_means_odps(
    "mex_anls.feature_wide_table",
    q=[0.05, 0.5, 0.95],
    quantile_method="approx",
    percentile_accuracy=10000,
)
```

When you need something closer to pandas linear interpolation, choose the exact mode explicitly:

```python
exact = proc_means_odps(
    "mex_anls.feature_wide_table",
    q=[0.05, 0.5, 0.95],
    quantile_method="exact",
)
```

`approx` uses MaxCompute `PERCENTILE_APPROX`, and `exact` uses `PERCENTILE_CONT`. The exact mode needs more compute resources and is not recommended as the default on very large tables or high-cardinality group combinations.

Special missing values are converted to NULL before the SQL aggregation:

```python
summary = proc_means_odps(
    "mex_anls.feature_wide_table",
    select_cols=["age", "income"],
    spec_missing_value={
        "age": [-1, -999],
        "income": -999,
    },
)
```

`N_ALL` is the total sample count of the filtered group, `N` is the valid sample count after excluding SQL NULL and special missing values, and `MISSING_RATE = 1 - N / N_ALL`.

### 5.4 CSV and ODPS Output

By default, only a DataFrame is returned, and no local file or remote table is created:

```python
summary = proc_means_odps("mex_anls.feature_wide_table")
```

You can optionally write a CSV, always without the pandas index:

```python
summary = proc_means_odps(
    "mex_anls.feature_wide_table",
    output_csv="output/feature_means.csv",
)
```

Writing back to MaxCompute requires an explicitly specified mode:

```python
summary = proc_means_odps(
    "mex_anls.feature_wide_table",
    output_table_name="mex_anls.feature_means_report",
    output_table_mode="overwrite",  # or "append"
)
```

- `overwrite` uses `ODPSRunner.upload_df(..., atomic=True)` to replace the target table atomically.
- `append` requires the target table to already exist, with column names, order, and types fully compatible with the result.
- The input table and output table cannot be the same table.
- The first version supports writing only to non-partitioned result tables.

### 5.5 Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `input_table_name` | Required | MaxCompute table name; `table` or `project.table` is supported. |
| `skip_cols` | `None` | Columns excluded from the candidate metric variables. |
| `select_cols` | `None` | Explicit metric variables; if omitted, numeric ordinary columns are chosen automatically. |
| `batch_size` | `50` | Number of features handled by each aggregation SQL. |
| `group` | `None` | One or several grouping columns; omitted means global statistics. |
| `q` | `[.05,.15,.25,.5,.75,.95,.99]` | Quantile points; must be strictly increasing and within `[0,1]`. |
| `quantile_method` | `"approx"` | `"approx"` or `"exact"`. |
| `percentile_accuracy` | `10000` | Accuracy parameter of `PERCENTILE_APPROX`. |
| `where_clause` | `None` | A single SQL filter condition, suited to partition pruning; must not contain a semicolon. |
| `spec_missing_value` | `None` | A global numeric sentinel, or per-column numeric sentinels. |
| `include_missing_group` | `False` | Whether to keep combinations where a group column is NULL. |
| `sqlrunner` | `None` | An already-initialized `ODPSRunner`; created lazily if omitted. |
| `output_csv` | `None` | Optional CSV output path. |
| `output_table_name` | `None` | Optional MaxCompute result table. |
| `output_table_mode` | `None` | Required when writing a result table: `"overwrite"` or `"append"`. |

The first version of `proc_means_odps` analyzes only numeric variables; `UNIQUE/TOP/FREQ` for categorical variables are not computed by this function.

## 6. `ParallelODPSManager` Concurrent Pull/Upload

`ParallelODPSManager` is a high-level wrapper around `ODPSRunner + ParallelApplyEngine`, suited to processing a large table concurrently by chunk:

- `pull()`: hash-buckets by `unique_key`, or, when there is no `unique_key`, automatically materializes a ROW_NUMBER temp table and buckets by it; it runs the SQL concurrently to pull data and merges it into a local CSV.
- `push()`: takes a pandas DataFrame or a local CSV, splits it into chunks by row, uploads them to ODPS temp tables, writes them into the target table with `UNION ALL`, and cleans up the temp tables.

### 6.1 Concurrent pull

The SQL template must contain `{chunk_filter}`, placed in the `WHERE` clause of the base table you want to split. If the template lacks this placeholder, `pull()` raises `ValueError` before any ODPS query, to avoid every chunk repeatedly pulling the full data:

```sql
SELECT flow_id, score, apply_time
FROM mex_anls.source_table
WHERE 1 = 1
  AND {chunk_filter}
```

#### Hash Bucketing: Recommended When You Have a Stable Key

When `unique_key` is configured, `pull_split_strategy="auto"` uses hash bucketing. Before running the concurrent chunks, SMF first runs a probe SQL to verify that `unique_key` is usable in the current SQL scope; if the check fails, it raises `ValueError` and does not enter the concurrent pull.

Python call:

```python
from Modeling_Tool import ParallelODPSConfig, ParallelODPSManager

manager = ParallelODPSManager(
    ParallelODPSConfig(
        unique_key="flow_id",
        pull_split_strategy="auto",  # auto + unique_key => hash
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
```

Each chunk automatically gets this injected:

```sql
ABS(HASH(flow_id)) % 20 = <chunk_id>
```

If you do not pass `n_chunks` directly, you can pass `chunk_size`; `pull()` then first runs `count_query` to derive the number of chunks. For complex wide-table joins, it is advisable to hand-write a lighter `count_query`.

#### ROW_NUMBER Bucketing: Used Automatically When There Is No unique_key

When `unique_key=None` and `pull_split_strategy="auto"`, SMF automatically uses ROW_NUMBER bucketing:

1. First render the original SQL with `{chunk_filter}=1=1`.
2. Materialize the result into an ODPS temp table and add an internal row-number column.
3. Pull each chunk concurrently with `WHERE (row_number_col - 1) % n_chunks = chunk_id`.
4. Delete the internal row-number column before writing the local CSV.
5. Clean up the temp table according to `cleanup_tmp` / `keep_tmp_on_error`.

```python
manager = ParallelODPSManager(
    ParallelODPSConfig(
        unique_key=None,
        pull_split_strategy="auto",  # auto + no unique_key => row_number
        n_chunks=20,
        n_jobs=5,
        backend="thread",
        tmp_table_prefix="tmp_parallel_odps",
    )
)

summary = manager.pull(
    sql_path="sql/pull_sample.sql",
    out_path="data/sample.csv",
)
```

The default row-number expression is:

```sql
ROW_NUMBER() OVER (ORDER BY 1)
```

If you want a more stable row-number order, pass `row_number_order_by`:

```python
ParallelODPSConfig(
    unique_key=None,
    pull_split_strategy="row_number",
    row_number_order_by="apply_time, flow_id",
    chunk_size=500000,
)
```

ROW_NUMBER mode creates an ODPS temporary staging table, with higher performance and storage cost than hash mode; as long as you can provide a stable and reasonably evenly distributed key, `unique_key` hash bucketing is still preferred.

### 6.2 Concurrent push

`push()` accepts a DataFrame or a CSV path. The write mode for the target table must be specified explicitly, to avoid overwriting a production table by mistake:

```python
summary = manager.push(
    data=df_or_csv_path,
    target_table="mex_anls.target_table",
    write_mode="overwrite",  # required: "overwrite" or "append"
)
```

Execution flow:

1. Split the DataFrame by row; for CSV input, `pd.read_csv(..., chunksize=...)` splits it as a stream into local temporary chunk files.
2. Each chunk is uploaded to its own ODPS tmp table, for example `tmp_parallel_odps_<run_id>_0000`.
3. All tmp tables are written into the final target table through `UNION ALL`.
4. On success, the tmp tables are cleaned up; on failure they are also cleaned up by default, unless `keep_tmp_on_error=True`.

Each chunk's `upload_df()` returns only after the atomic rename of the temp table has completed, so the final `UNION ALL` never runs before the tmp tables are visible. Callers do not need to add any extra `sleep` or polling.

Write modes:

| `write_mode` | Behavior |
|---|---|
| `"overwrite"` | Drop the target table first, then `CREATE TABLE target AS SELECT ... UNION ALL ...`. |
| `"append"` | Append to the existing target table with `INSERT INTO TABLE target SELECT ... UNION ALL ...`. |

### 6.3 Backend Recommendations

| backend | Recommendation |
|---|---|
| `"thread"` | The default recommendation for ODPS IO tasks; it shares the connection pool, and the `ODPSRunner` wide-table download patch is already thread-safe. |
| `"sequential"` | Use when debugging chunk SQL, upload logic, and tmp-table cleanup. |
| `"process"` | Each worker creates a new `ODPSRunner()` and does not pass live connections across processes; suited to scenarios needing stronger isolation at higher overhead. |

The first version of `push()` does not support partitioned target tables; if partitioned writes are needed, a `partition` parameter can be added later.

## 7. Common Pitfalls

### ❌ Pitfall 1: `to_df=False + csv_path` historically wrote no CSV

```python
# The "illusion" before the fix (≤ v1.0.0):
odps.run_sql(sql, to_df=False, csv_path="x.csv")
# → the SQL ran, the CSV was not written, a silent failure
```

**After the fix (current version)**: setting either one triggers the download.

### ❌ Pitfall 2: A table with 200+ columns triggers HTTP 414

```text
odps.errors.InternalServerError: HTTP 414 (Request-URI Too Long)
```

This is handled automatically by `ODPSRunner`'s thread-safe wide-schema patch, with no manual intervention needed.

### ❌ Pitfall 3: An ODPS Instance is one-shot

`execute_sql` re-issues the SQL every time, even if the dataset has not changed. To avoid repeated cost/time:

```python
# Pattern A: cache on disk
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    odps.run_sql(sql, to_df=False, csv_path=str(csv_path))
    df = pd.read_csv(csv_path)

# Pattern B: cache intermediate results with Modeling_Tool.Core.utils.save_model
from Modeling_Tool.Core.utils import save_model, load_model
save_model(df, "data/cached.pkl")
```

### ❌ Pitfall 4: Long-running big queries hit the Bash 120s timeout

`run_sql` is synchronous and blocking; for queries that take minutes you should:

1. **Start in the background** — with `nohup` + `&`, see the [appendix](#appendix-starting-a-long-query-in-the-background)
2. **Poll the status** — check progress through `odps.instances`
3. **Write to a log file** — redirect to `/tmp/odps_<ts>.log` for easy tracing

### Appendix: Starting a Long Query in the Background

```bash
cd /path/to/project
export PYTHONPATH="$(pwd):$PYTHONPATH"

nohup python3 -u -c "
import sys
sys.path.insert(0, '.')
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
odps = ODPSRunner()
_ = odps.run_sql(open('big_query.sql').read(), to_df=False, csv_path='out.csv')
" > /tmp/odps_$(date +%s).log 2>&1 &
PID=$!
echo "ODPS job PID=$PID, log=/tmp/odps_*.log"
```

## 8. Other `ODPSRunner` Methods

### `download_table(table_name, partition=None, n_process=1, csv_path=None)`

Pulls an **entire table** directly (rather than a SQL query), inferring the schema automatically:

```python
df = odps.download_table(
    "mex_anls.drv",
    partition={"dt": "2025-08-18"},
    csv_path="out.csv",
)
```

### `upload_df(df, table_name, table_schema=None, partition=None)`

Uploads a DataFrame to a new ODPS table:

```python
schema = ODPSRunner.cre_table_schema(df, partition_name="dt")
odps.upload_df(df, "mex_anls.my_table", table_schema=schema, partition="dt=2025-08-18")
```

On upload, the schema is first inferred from the original pandas dtypes, and then `np.nan`, `pd.NA`, and `NaT` are converted to ODPS `NULL` in a copy of the records; the DataFrame passed in by the caller is not modified, and there is no need to call `npnan2none()` beforehand. The default atomic replacement uses blocking DDL: the target-table backup, the tmp-table rename, and failure recovery all wait for the ODPS Instance to succeed before moving to the next step.

### `insert_df(df, table_name, overwrite=True, partition=None)`

Appends to an **existing** table:

```python
odps.insert_df(df, "mex_anls.my_table", overwrite=False, partition="dt=2025-08-19")
```

### `cre_table_schema(df, partition_name=None)` (staticmethod)

Infers the ODPS schema from a DataFrame:

```python
schema = ODPSRunner.cre_table_schema(df, partition_name="dt")
# integer → bigint, float32 → float, float64 → double
# boolean → boolean, datetime → datetime, object/string/category → string
```

Currently unsupported dtypes such as complex and timedelta raise a clear `TypeError` and are not silently downcast.

## 9. Related Utility Functions

[`Modeling_Tool.Core.utils.pull_attributes_in_batch`](../api/core.md) provides the ability to split `{varlist}` into batches — strongly recommended when a single SQL pull has more than 2000 columns:

```python
from Modeling_Tool.Core.utils import pull_attributes_in_batch

# Internally splits varlist into N batches, runs run_sql several times and concatenates the results
result_df = pull_attributes_in_batch(
    table_name="mex_anls.drv",
    varlist=big_varlist,            # 1000+ columns
    batch_num=6,                    # ~167 columns per batch
    unikey="FLOW_ID",
    main_info_select=["*"],
)
```

## 10. Next Steps

- Want the complete modeling pipeline? Read [End-to-End Modeling Pipeline](../pipeline.md)
- Want downstream processing such as WOE / IV? Read [WOE Encoding](woe.md) and [Feature Screening](feature.md)
- Want specific API signatures? Visit [API Reference → Core](../api/core.md)
