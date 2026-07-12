# 数据一致性对比 ProcCompare

`ProcCompareEngine` 是 SMF Core 层的通用 dataset 一致性对比工具，定位类似 SAS `proc compare`。它适合比较两张宽表、两个 CSV、两版特征快照、回溯数据与新抽取数据、上线前后样本明细等。

第一版重点支持：

- pandas `DataFrame` 对比。
- 本地 CSV 大表分块对比。
- 按主键对齐，或显式按行号对齐。
- schema / 覆盖度 / 字段级 / 行级 / cell 级 mismatch 汇总。
- 数值、时间、字符字段容差比较。
- 可选 CSV 落盘和 ExcelMaster 报告。

!!! warning "必须明确对齐方式"

    默认要求传入 `key_cols`。如果没有主键，必须显式设置 `row_order_compare=True`，表示你确认两张表可以按行号一一对齐。这样可以避免无序大表被误按行号硬比较。

## 1. 快速开始

### 1.1 DataFrame 模式

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

也可以使用便捷函数：

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

### 1.2 CSV 大表模式

当 `left` / `right` 是 CSV 路径时，`ProcCompareEngine` 会流式读取文件，按 `key_cols` hash 分区到临时 chunk，再逐分区 merge 和 compare，避免一次性把大 CSV 全部读进内存。

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

CSV 分块对比适合：

- 两张本地大 CSV。
- 从 ODPS / SQL 先落盘后的快照复核。
- 宽表上线前后结果一致性验收。

## 2. 对齐方式

### 2.1 按主键对齐

推荐方式。`key_cols` 可以是一列或多列：

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

输出里的 `coverage_summary` 会统计：

- 左表行数。
- 右表行数。
- 两边都有的 common rows。
- 只在左表出现的 rows。
- 只在右表出现的 rows。
- 实际参与对比的字段数。

### 2.2 按行号对齐

如果两张表没有主键，但你确认行顺序完全一致，可以设置：

```python
result = proc_compare(
    left,
    right,
    row_order_compare=True,
)
```

内部会生成临时行号列用于对齐。这个模式适合小样本 quick check，不建议用于无序生产大表。

## 3. 字段选择

默认会比较两张表中除主键和 `ignore_cols` 之外的共同字段。

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    ignore_cols=["etl_time", "batch_id"],
)
```

只比较指定字段：

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    compare_cols=["score", "credit_limit", "risk_level"],
)
```

如果 `compare_cols` 中某些字段只存在于一侧，不会直接报错；它们会出现在 `schema_summary` 中，状态为 `left_only_column` 或 `right_only_column`，角色为 `schema_only`。`requested_for_compare` 表示用户是否要求比较该字段，`eligible_for_compare` 表示该字段是否两侧都存在且实际进入值比较。只有 `eligible_for_compare=True` 的字段才会标记为 `role="compare"`。

## 4. 容差和缺失值规则

### 4.1 数值字段

数值比较规则：

```text
abs(left - right) <= numeric_tol + numeric_rtol * abs(right)
```

示例：

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    numeric_tol=1e-6,
    numeric_rtol=1e-4,
)
```

### 4.2 单字段容差

`per_column_tolerance` 可以覆盖某些字段的全局容差。

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

### 4.3 时间字段

datetime 字段按秒级差异比较：

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

CSV 读取后时间列通常会变成 `object`。此时推荐用 `datetime_cols` 显式声明时间字段；也可以在 `per_column_tolerance` 中为某列配置 `datetime_tol_seconds`。若只设置全局 `datetime_tol_seconds > 0`，引擎会对非数值且符合常见日期格式的字段做保守探测。命中后左右两侧统一用 `pd.to_datetime(errors="coerce")` 转换，因此 DataFrame、CSV 和三种 backend 使用相同口径。

### 4.4 缺失值

默认规则：

- 双侧都为空：一致。
- 单侧为空：不一致。

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    both_null_equal=True,
)
```

如果业务里有特殊缺失值，可以统一转成缺失：

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    missing_values=["", "NULL", -999],
)
```

## 5. mismatch 明细控制

`detail_mode` 控制 cell 级 mismatch 明细的输出量。

| `detail_mode` | 行为 |
|---|---|
| `"top"` | 默认。只保留 Top N mismatch 明细，避免报告爆炸。 |
| `"full"` | 保留所有 cell mismatch，适合审计或小表。 |
| `"none"` | 不输出 cell 级明细，只保留 summary。 |

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    detail_mode="top",
    top_n=5000,
)
```

!!! tip "大表建议"

    大表建议使用 `detail_mode="top"` 或 `"none"`。如果一定要全量明细，建议 `write_outputs=True`，并主要查看落盘 CSV，不要把全量明细写进 Excel。

## 6. 重复主键处理

默认 `duplicate_key_policy="raise"`，发现重复主键会直接报错，避免多对多 merge 造成误判。

| `duplicate_key_policy` | 行为 |
|---|---|
| `"raise"` | 默认。发现重复 key 直接报错。 |
| `"first"` | 保留每个 key 第一条记录参与比较。 |
| `"all"` | 保留全部重复记录；仅在你明确接受多对多 merge 时使用。 |

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    duplicate_key_policy="first",
)
```

重复 key 信息会进入 `duplicate_key_summary`。

## 7. 输出结果解读

`ProcCompareEngine.run()` 返回 `ProcCompareResult`。

| 字段 | 说明 |
|---|---|
| `coverage_summary` | 左右表行数、common / only-left / only-right 行数、比较字段数。 |
| `schema_summary` | 每个字段在左右表是否存在、dtype、字段角色、schema 状态，以及是否被请求/是否实际可比较。 |
| `column_summary` | 每个比较字段的 `n_compared`、`n_mismatch`、缺失数、`mean_diff` 和 `max_abs_diff`；CSV 分区模式会按有效 diff 数量加权汇总。 |
| `row_summary` | 每个 key 的行状态和该行不一致字段数。 |
| `cell_mismatches` | cell 级 mismatch 明细，受 `detail_mode/top_n` 控制。 |
| `duplicate_key_summary` | 重复主键统计。 |
| `output_paths` | CSV 落盘路径。 |
| `report_path` | ExcelMaster 报告路径。 |

常见检查方式：

```python
# 是否有覆盖缺口
result.coverage_summary[
    ["n_common_rows", "n_left_only_rows", "n_right_only_rows"]
]

# 哪些字段差异最多
result.column_summary.sort_values("n_mismatch", ascending=False).head(20)

# 哪些 flow_id 有多个字段不一致
result.row_summary.sort_values("n_cell_mismatch", ascending=False).head(20)

# 查看 Top cell mismatch
result.cell_mismatches.head(100)
```

## 8. 输出文件和 Excel 报告

开启 CSV 落盘：

```python
result = proc_compare(
    left,
    right,
    key_cols=["flow_id"],
    output_dir="output/proc_compare",
    write_outputs=True,
)
```

输出文件：

- `coverage_summary.csv`
- `schema_summary.csv`
- `column_summary.csv`
- `row_summary.csv`
- `cell_mismatches.csv`
- `duplicate_key_summary.csv`

开启 ExcelMaster 报告：

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

默认报告名：

```text
output/proc_compare/Proc_Compare_Report.xlsx
```

Excel 会写 summary 和受控的 mismatch 明细。`max_excel_rows` 用于限制每个 sheet 写入 Excel 的最大行数。

## 9. 完整参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `output_dir` | `"output/proc_compare"` | CSV 和 Excel 输出目录。 |
| `write_outputs` | `True` | 是否输出 CSV 结果。 |
| `write_excel` | `False` | 是否输出 ExcelMaster 报告。 |
| `left_name` | `"left"` | 左表名称，写入 summary。 |
| `right_name` | `"right"` | 右表名称，写入 summary。 |
| `key_cols` | `None` | 主键列。推荐传入；支持多列主键。 |
| `row_order_compare` | `False` | 无主键时是否按行号对齐。 |
| `compare_cols` | `None` | 指定比较字段；不传则比较可比共同字段。 |
| `ignore_cols` | `[]` | 不参与比较的字段。 |
| `chunk_size` | `200000` | CSV 模式每次读取行数。 |
| `n_partitions` | `16` | CSV 模式 hash 分区数。 |
| `backend` | `"sequential"` | CSV partition 比较后端：`"sequential"` / `"thread"` / `"process"`。 |
| `numeric_tol` | `1e-8` | 数值绝对容差。 |
| `numeric_rtol` | `0.0` | 数值相对容差。 |
| `datetime_tol_seconds` | `0.0` | 时间字段秒级容差。 |
| `datetime_cols` | `[]` | 显式指定时间字段；CSV 模式推荐配置，避免 object dtype 丢失时间语义。 |
| `per_column_tolerance` | `{}` | 单字段容差覆盖。 |
| `both_null_equal` | `True` | 双侧为空是否视为一致。 |
| `missing_values` | `[]` | 额外视为缺失的取值。 |
| `detail_mode` | `"top"` | cell mismatch 明细模式：`"top"` / `"full"` / `"none"`。 |
| `top_n` | `1000` | `detail_mode="top"` 时保留的 mismatch 明细数。 |
| `duplicate_key_policy` | `"raise"` | 重复 key 策略：`"raise"` / `"first"` / `"all"`。 |
| `excel_output_path` | `None` | Excel 输出路径；不传则使用 `output_dir/Proc_Compare_Report.xlsx`。 |
| `max_excel_rows` | `100000` | 每个 Excel sheet 最多写入行数。 |

## 10. 推荐实践

- 生产宽表优先使用 `key_cols`，不要依赖行顺序。
- 大 CSV 优先设置 `chunk_size` 和 `n_partitions`，避免一次性读入内存。
- 大表默认使用 `detail_mode="top"`，只有审计需要时再开 `"full"`。
- 对金额、概率、时间戳等字段设置合理的单字段容差。
- 对 ETL 时间、批次号、导出时间等字段使用 `ignore_cols`。
- 如果比较的是从 ODPS 拉下来的两张快照，建议先用 `ODPSRunner` 或 `ParallelODPSManager.pull()` 落成本地 CSV，再用 `ProcCompareEngine` 比较。
