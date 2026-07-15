# ODPS 数据抽取

SuperModelingFactory 的 [`Modeling_Tool.Core.ODPS_Tool`](../api/core.md) 提供 `ODPSRunner` —— 封装阿里云 MaxCompute (ODPS) 的 SQL 执行、数据下载与上传。

## 1. 快速上手

```python
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner

odps = ODPSRunner()

# 1) 拉数据到 DataFrame
df = odps.run_sql("SELECT * FROM mex_anls.drv LIMIT 1000")
print(df.head())

# 2) 拉数据直接落盘 CSV (大表推荐)
_ = odps.run_sql(
    "SELECT * FROM mex_anls.drv",
    to_df=False,
    csv_path="/data/drv.csv",
)

# 3) 既要 DataFrame 也要落盘
df = odps.run_sql(
    "SELECT * FROM mex_anls.drv",
    csv_path="/data/drv.csv",
)
```

## 2. `run_sql` 参数语义

| `to_df` | `csv_path` | 行为 | 返回 |
|---------|-----------|------|------|
| `True` | `None` | 下载到内存 | DataFrame |
| `True` | 设置 | 下载 + 写 CSV | DataFrame |
| `False` | `None` | **不下载** (DDL/INSERT 用) | 空 DataFrame |
| `False` | 设置 | 下载 + 写 CSV (释放内存) | 空 DataFrame |

!!! warning "反直觉的 `to_df=False + csv_path`"

    历史上 `to_df=False` 会让 `csv_path` 也被**静默忽略** — 见 [§7 历史坑位](#7-常见坑位)。
    当前版本已修复: `to_df` 与 `csv_path` 互相独立, 只要任一被设置就会触发下载。

## 3. 内部机制

### 3.1 重试策略

- **执行阶段** (`execute_sql`) — 只跑 1 次, 无重试 (避免重复扣费)
- **下载阶段** (`to_pandas` + `to_csv`) — 最多 6 次重试, 适合网络抖动

### 3.2 宽表 schema 补丁

当 SQL 返回列数 > 200 时, `ODPSRunner` 会自动启用线程安全的 wide-schema patch:

```text
HTTP 414 (URI Too Long) ← 原始请求中包含全部列名作为 URL query 参数
                         修补后 → 移除 columns 参数, 由 server 全量返回
```

补丁会自动恢复，对调用方透明。多线程同时下载宽表时，内部用锁和引用计数保证最后一个下载任务退出后才恢复 ODPS 原始方法，避免并发 patch/unpatch 互相覆盖。

### 3.3 `__init__` 中的连接配置

```python
class ODPSRunner:
    def __init__(self):
        self.o = ODPS(
            "<ALIBABA_CLOUD_ACCESS_KEY_ID>",      # AccessKey ID
            "<ALIBABA_CLOUD_ACCESS_KEY_SECRET>",  # AccessKey Secret
            "mex_anls",                          # ← 默认 project
            endpoint="https://service.ap-southeast-1-vpc.maxcompute.aliyun-inc.com/api",
        )
        options.retry_times = 6
        options.pool_maxsize = 200
        options.connect_timeout = 3600
        options.read_timeout = 3600
```

!!! warning "凭证配置"

    当前实现通过环境变量读取阿里云凭证。请勿把真实凭证写入源码或文档。
    
    1. 通过环境变量注入:
       ```python
       self.o = ODPS(
           os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
           os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
           os.environ["ODPS_PROJECT"],
           endpoint=os.environ["ODPS_ENDPOINT"],
       )
       ```
    2. 配合 `.gitignore` 把 `.env` 加入忽略, 避免泄漏
    3. 长期可改用 RAM Role / STS Token

## 4. 完整示例 — 抽取样本到本地

```python
from pathlib import Path
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
from Modeling_Tool.Core.utils import parse_sql_file, mkdir_if_not_exist

odps = ODPSRunner()

# 1) 渲染 SQL 模板
sql = parse_sql_file(
    sql_path="sql/00_sample.sql",
    tgt_name="IS_DPD7",
    varlist="score_b, income, age, n_overdue",  # 占位符替换
)

# 2) 落盘目录
out_dir = Path("data/")
mkdir_if_not_exist(str(out_dir))
csv_path = out_dir / "sample_drv.csv"

# 3) 跑 SQL, 只写 CSV, 不占用内存
_ = odps.run_sql(sql, to_df=False, csv_path=str(csv_path), n_process=4)
print(f"样本已抽取: {csv_path}")
```

## 5. `proc_means_odps` ODPS 端描述性统计

`proc_means_odps()` 直接在 MaxCompute 中计算数值变量的描述性统计，只把聚合后的小结果下载为 pandas DataFrame。它适合行数很大或特征很多、不希望先 `SELECT *` 拉取全表的场景。

```python
from Modeling_Tool import proc_means_odps

summary = proc_means_odps(
    input_table_name="mex_anls.feature_wide_table",
    skip_cols=["flow_id", "badflag"],
    batch_size=50,
)
```

默认输出一行一个变量：

```text
attribute, N_ALL, N, MEAN, STD, MIN,
Q5, Q15, Q25, Q50, Q75, Q95, Q99, MAX, MISSING_RATE
```

### 5.1 按 group 统计

`group` 支持字符串或字段列表，输出结构和数值型 `proc_means_by_grp()` 一致：

```python
grouped = proc_means_odps(
    input_table_name="mex_anls.feature_wide_table",
    select_cols=["age", "credit_limit", "income"],
    group=["apply_month", "channel"],
    where_clause="dt >= '2026-01-01'",
)
```

默认 `include_missing_group=False`，任一 group 字段为 NULL 的记录不进入分组结果，与 pandas `groupby` 的默认口径一致。设为 `True` 可保留 NULL group。

### 5.2 字段选择与批次

- `select_cols=None`：从普通表字段中自动选择数值列，不自动分析分区字段。
- `select_cols=[...]`：只分析指定数值列；显式传入非数值列会报错。
- `skip_cols=[...]`：从候选列中剔除字段；与 `select_cols` 重叠时 `skip_cols` 优先。
- `group` 字段只参与分组，不会作为指标变量重复分析。
- `batch_size=50`：每条 ODPS 聚合 SQL 处理 50 个特征。它限制 SQL 宽度，不是行级 chunk，也不会下载源数据行。

每个 feature batch 只扫描源表一次，同时计算该批所有变量的 `COUNT/AVG/STDDEV_SAMP/MIN/PERCENTILE/MAX`，然后在本地把小型宽聚合结果转成长表。任一 batch 失败时函数立即抛错，CSV 和 ODPS 结果表都不会写出半成品。

### 5.3 分位数与特殊缺失值

默认使用适合大表的近似分位数：

```python
approx = proc_means_odps(
    "mex_anls.feature_wide_table",
    q=[0.05, 0.5, 0.95],
    quantile_method="approx",
    percentile_accuracy=10000,
)
```

需要更接近 pandas 线性插值时可显式选择精确模式：

```python
exact = proc_means_odps(
    "mex_anls.feature_wide_table",
    q=[0.05, 0.5, 0.95],
    quantile_method="exact",
)
```

`approx` 使用 MaxCompute `PERCENTILE_APPROX`，`exact` 使用 `PERCENTILE_CONT`。精确模式需要更多计算资源，不建议在超大表和高基数组合分组上默认开启。

特殊缺失值会在 SQL 聚合前转换为 NULL：

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

`N_ALL` 是过滤后 group 的总样本数，`N` 是排除 SQL NULL 和特殊缺失值后的有效样本数，`MISSING_RATE = 1 - N / N_ALL`。

### 5.4 CSV 与 ODPS 输出

默认只返回 DataFrame，不创建本地文件或远端表：

```python
summary = proc_means_odps("mex_anls.feature_wide_table")
```

可选写 CSV，固定不写 pandas 索引：

```python
summary = proc_means_odps(
    "mex_anls.feature_wide_table",
    output_csv="output/feature_means.csv",
)
```

写回 MaxCompute 时必须显式指定模式：

```python
summary = proc_means_odps(
    "mex_anls.feature_wide_table",
    output_table_name="mex_anls.feature_means_report",
    output_table_mode="overwrite",  # 或 "append"
)
```

- `overwrite` 使用 `ODPSRunner.upload_df(..., atomic=True)` 原子替换目标表。
- `append` 要求目标表已存在，且字段名称、顺序和类型与结果完全兼容。
- 输入表和输出表不能是同一张表。
- 第一版只支持写入非分区结果表。

### 5.5 参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `input_table_name` | 必填 | MaxCompute 表名，支持 `table` 或 `project.table`。 |
| `skip_cols` | `None` | 从候选指标变量中排除的字段。 |
| `select_cols` | `None` | 显式指标变量；不传时自动选择数值型普通字段。 |
| `batch_size` | `50` | 每条聚合 SQL 处理的特征数。 |
| `group` | `None` | 单个或多个分组字段；不传表示全局统计。 |
| `q` | `[.05,.15,.25,.5,.75,.95,.99]` | 分位点，必须严格递增且位于 `[0,1]`。 |
| `quantile_method` | `"approx"` | `"approx"` 或 `"exact"`。 |
| `percentile_accuracy` | `10000` | `PERCENTILE_APPROX` 精度参数。 |
| `where_clause` | `None` | 单条 SQL 过滤条件，适合分区裁剪；不能含分号。 |
| `spec_missing_value` | `None` | 全局数值哨兵，或按字段配置的数值哨兵。 |
| `include_missing_group` | `False` | 是否保留 group 字段为 NULL 的组合。 |
| `sqlrunner` | `None` | 已初始化的 `ODPSRunner`；不传时延迟创建。 |
| `output_csv` | `None` | 可选 CSV 输出路径。 |
| `output_table_name` | `None` | 可选 MaxCompute 结果表。 |
| `output_table_mode` | `None` | 写结果表时必填：`"overwrite"` 或 `"append"`。 |

`proc_means_odps` 第一版只分析数值变量；类别变量的 `UNIQUE/TOP/FREQ` 不在本函数中计算。

## 6. `ParallelODPSManager` 并发拉取/上传

`ParallelODPSManager` 是 `ODPSRunner + ParallelApplyEngine` 的高层封装，适合把大表按 chunk 并发处理：

- `pull()`：按 `unique_key` 哈希分桶，或在没有 `unique_key` 时自动物化 ROW_NUMBER 临时表再分桶，并发执行 SQL 拉数，合并成本地 CSV。
- `push()`：接收 pandas DataFrame 或本地 CSV，按行拆 chunk 上传到 ODPS 临时表，再 `UNION ALL` 写入目标表，并清理临时表。

### 6.1 并发 pull

SQL 模板必须包含 `{chunk_filter}`，并放在希望切分的基础表 `WHERE` 子句里。若模板里没有这个占位符，`pull()` 会在任何 ODPS 查询前直接抛 `ValueError`，避免每个 chunk 都重复拉全量数据：

```sql
SELECT flow_id, score, apply_time
FROM mex_anls.source_table
WHERE 1 = 1
  AND {chunk_filter}
```

#### Hash 分桶：推荐有稳定 key 时使用

当配置了 `unique_key`，`pull_split_strategy="auto"` 会走 hash 分桶。执行并发 chunk 前，SMF 会先跑一次 probe SQL，校验 `unique_key` 在当前 SQL 作用域里可用；校验失败会抛 `ValueError`，不会进入并发拉取。

Python 调用：

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

每个 chunk 会自动注入：

```sql
ABS(HASH(flow_id)) % 20 = <chunk_id>
```

如果不直接传 `n_chunks`，可以传 `chunk_size`；此时 `pull()` 会先跑 `count_query` 推导分块数。对复杂宽表 join，建议手写更轻量的 `count_query`。

#### ROW_NUMBER 分桶：没有 unique_key 时自动使用

当 `unique_key=None` 且 `pull_split_strategy="auto"`，SMF 会自动走 ROW_NUMBER 分桶：

1. 先用 `{chunk_filter}=1=1` 渲染原 SQL。
2. 将结果物化为 ODPS 临时表，并新增内部行号列。
3. 并发执行 `WHERE (row_number_col - 1) % n_chunks = chunk_id` 拉取各 chunk。
4. 本地输出 CSV 前删除内部行号列。
5. 根据 `cleanup_tmp` / `keep_tmp_on_error` 清理临时表。

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

默认行号表达式是：

```sql
ROW_NUMBER() OVER (ORDER BY 1)
```

如果希望行号顺序更稳定，可传 `row_number_order_by`：

```python
ParallelODPSConfig(
    unique_key=None,
    pull_split_strategy="row_number",
    row_number_order_by="apply_time, flow_id",
    chunk_size=500000,
)
```

ROW_NUMBER 模式会创建一张 ODPS 临时 staging 表，性能和存储成本高于 hash 模式；只要能提供稳定且分布较均匀的 key，仍推荐优先使用 `unique_key` hash 分桶。

### 6.2 并发 push

`push()` 支持 DataFrame 或 CSV 路径。目标表写入模式必须显式指定，避免误覆盖生产表：

```python
summary = manager.push(
    data=df_or_csv_path,
    target_table="mex_anls.target_table",
    write_mode="overwrite",  # 必填: "overwrite" 或 "append"
)
```

执行流程：

1. 按行拆分 DataFrame；CSV 输入使用 `pd.read_csv(..., chunksize=...)` 流式切分到本地临时 chunk 文件。
2. 每个 chunk 上传到独立 ODPS tmp 表，例如 `tmp_parallel_odps_<run_id>_0000`。
3. 所有 tmp 表通过 `UNION ALL` 写入最终目标表。
4. 成功后清理 tmp 表；失败时默认也清理，除非 `keep_tmp_on_error=True`。

每个 chunk 的 `upload_df()` 会等待临时表原子 rename 完成后才返回，因此最终 `UNION ALL` 不会抢在 tmp 表可见之前执行。调用方无需额外添加 `sleep` 或轮询。

写入模式：

| `write_mode` | 行为 |
|---|---|
| `"overwrite"` | 先删除目标表，再 `CREATE TABLE target AS SELECT ... UNION ALL ...`。 |
| `"append"` | 使用 `INSERT INTO TABLE target SELECT ... UNION ALL ...` 追加到已有目标表。 |

### 6.3 backend 建议

| backend | 建议 |
|---|---|
| `"thread"` | ODPS IO 任务默认推荐；共享连接池，`ODPSRunner` 宽表下载补丁已做线程安全保护。 |
| `"sequential"` | 调试 chunk SQL、上传逻辑和 tmp 表清理时使用。 |
| `"process"` | 每个 worker 内新建 `ODPSRunner()`，不跨进程传活连接；适合隔离性更强但开销更高的场景。 |

第一版 `push()` 不支持分区目标表；如果需要分区写入，后续可扩展 `partition` 参数。

## 7. 常见坑位

### ❌ 坑 1: `to_df=False + csv_path` 历史上 CSV 不会写

```python
# 修复前 (≤ v1.0.0) 的"假象":
odps.run_sql(sql, to_df=False, csv_path="x.csv")
# → SQL 跑了, CSV 没写, 沉默失败
```

**修复后 (当前版本)**: 任一被设置都会触发下载。

### ❌ 坑 2: 200+ 列宽表触发 HTTP 414

```text
odps.errors.InternalServerError: HTTP 414 (Request-URI Too Long)
```

已由 `ODPSRunner` 的线程安全 wide-schema patch 自动处理, 无需手动干预。

### ❌ 坑 3: ODPS Instance 是一次性的

`execute_sql` 每次都会重新发起 SQL, 即便数据集没变。要避免重复扣费/时长:

```python
# 模式 A: 落盘缓存
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    odps.run_sql(sql, to_df=False, csv_path=str(csv_path))
    df = pd.read_csv(csv_path)

# 模式 B: 用 Modeling_Tool.Core.utils.save_model 缓存中间结果
from Modeling_Tool.Core.utils import save_model, load_model
save_model(df, "data/cached.pkl")
```

### ❌ 坑 4: 大查询耗时长导致 Bash 120s 超时

`run_sql` 是同步阻塞, 几分钟级别的查询应:

1. **在后台启动** — 用 `nohup` + `&`, 见 [附录](#附录-后台启动长查询)
2. **轮询状态** — 通过 `odps.instances` 查进度
3. **写入日志文件** — 重定向到 `/tmp/odps_<ts>.log` 方便回溯

### 附录: 后台启动长查询

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

## 8. `ODPSRunner` 其他方法

### `download_table(table_name, partition=None, n_process=1, csv_path=None)`

直接拉取**整张表** (而非 SQL 查询), 自动推断 schema:

```python
df = odps.download_table(
    "mex_anls.drv",
    partition={"dt": "2025-08-18"},
    csv_path="out.csv",
)
```

### `upload_df(df, table_name, table_schema=None, partition=None)`

上传 DataFrame 到 ODPS 新表:

```python
schema = ODPSRunner.cre_table_schema(df, partition_name="dt")
odps.upload_df(df, "mex_anls.my_table", table_schema=schema, partition="dt=2025-08-18")
```

上传时先按原始 pandas dtype 推断 schema，再在 records 副本中把 `np.nan`、`pd.NA` 和 `NaT` 转成 ODPS `NULL`；不会修改调用方传入的 DataFrame，也不需要预先调用 `npnan2none()`。默认原子替换使用阻塞 DDL：目标表备份、tmp 表 rename 和失败恢复都会等待 ODPS Instance 成功后再进入下一步。

### `insert_df(df, table_name, overwrite=True, partition=None)`

追加写入**已存在**的表:

```python
odps.insert_df(df, "mex_anls.my_table", overwrite=False, partition="dt=2025-08-19")
```

### `cre_table_schema(df, partition_name=None)` (staticmethod)

从 DataFrame 推断 ODPS Schema:

```python
schema = ODPSRunner.cre_table_schema(df, partition_name="dt")
# integer → bigint, float32 → float, float64 → double
# boolean → boolean, datetime → datetime, object/string/category → string
```

复数和 timedelta 等当前不支持的 dtype 会抛出清晰的 `TypeError`，不会静默降型。

## 9. 相关工具函数

[`Modeling_Tool.Core.utils.pull_attributes_in_batch`](../api/core.md) 提供按批切分 `{varlist}` 的能力 — 当单次 SQL 拉取列数 > 2000 时强烈建议使用:

```python
from Modeling_Tool.Core.utils import pull_attributes_in_batch

# 内部把 varlist 切成 N 批, 多次 run_sql 拼接
result_df = pull_attributes_in_batch(
    table_name="mex_anls.drv",
    varlist=big_varlist,            # 1000+ 列
    batch_num=6,                    # 每批 ~ 167 列
    unikey="FLOW_ID",
    main_info_select=["*"],
)
```

## 10. 下一步

- 想要完整的建模流水线？请阅读 [端到端建模流水线](../pipeline.md)
- 想看 WOE / IV 等下游处理？请阅读 [WOE 编码](woe.md) 与 [特征筛选](feature.md)
- 想看具体 API 签名？请访问 [API 参考 → Core](../api/core.md)
