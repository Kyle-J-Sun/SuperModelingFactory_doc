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

    历史上 `to_df=False` 会让 `csv_path` 也被**静默忽略** — 见 [§6 历史坑位](#6-常见坑位)。
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

## 5. `ParallelODPSManager` 并发拉取/上传

`ParallelODPSManager` 是 `ODPSRunner + ParallelApplyEngine` 的高层封装，适合把大表按 chunk 并发处理：

- `pull()`：按 `unique_key` 哈希分桶，并发执行 SQL 拉数，合并成本地 CSV。
- `push()`：接受 pandas DataFrame 或本地 CSV，按行拆 chunk 上传为 ODPS 临时表，再 `UNION ALL` 写入目标表，并清理临时表。

### 5.1 并发 pull

SQL 模板必须包含 `{chunk_filter}`，并放在希望切分的基础表 `WHERE` 子句里：

```sql
SELECT flow_id, score, apply_time
FROM mex_anls.source_table
WHERE 1 = 1
  AND {chunk_filter}
```

Python 调用：

```python
from Modeling_Tool import ParallelODPSConfig, ParallelODPSManager

manager = ParallelODPSManager(
    ParallelODPSConfig(
        unique_key="flow_id",
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

### 5.2 并发 push

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

写入模式：

| `write_mode` | 行为 |
|---|---|
| `"overwrite"` | 先删除目标表，再 `CREATE TABLE target AS SELECT ... UNION ALL ...`。 |
| `"append"` | 使用 `INSERT INTO TABLE target SELECT ... UNION ALL ...` 追加到已有目标表。 |

### 5.3 backend 建议

| backend | 建议 |
|---|---|
| `"thread"` | ODPS IO 任务默认推荐；共享连接池，`ODPSRunner` 宽表下载补丁已做线程安全保护。 |
| `"sequential"` | 调试 chunk SQL、上传逻辑和 tmp 表清理时使用。 |
| `"process"` | 每个 worker 内新建 `ODPSRunner()`，不跨进程传活连接；适合隔离性更强但开销更高的场景。 |

第一版 `push()` 不支持分区目标表；如果需要分区写入，后续可扩展 `partition` 参数。

## 6. 常见坑位

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

## 7. `ODPSRunner` 其他方法

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

### `insert_df(df, table_name, overwrite=True, partition=None)`

追加写入**已存在**的表:

```python
odps.insert_df(df, "mex_anls.my_table", overwrite=False, partition="dt=2025-08-19")
```

### `cre_table_schema(df, partition_name=None)` (staticmethod)

从 DataFrame 推断 ODPS Schema:

```python
schema = ODPSRunner.cre_table_schema(df, partition_name="dt")
# int64 → bigint, float64 → double, object → string
```

## 8. 相关工具函数

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

## 8. 下一步

- 想要完整的建模流水线？请阅读 [端到端建模流水线](../pipeline.md)
- 想看 WOE / IV 等下游处理？请阅读 [WOE 编码](woe.md) 与 [特征筛选](feature.md)
- 想看具体 API 签名？请访问 [API 参考 → Core](../api/core.md)
