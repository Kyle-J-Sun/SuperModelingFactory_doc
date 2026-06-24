# 线上线下一致性校验（UAT）

模型上线前后，需验证**线上**与**线下**对同一批用户的打分与入模特征一致。SuperModelingFactory 的 `UATConsistencyChecker` 把 `99_uat_validation.ipynb` 的完整逻辑封装为可复用类。

!!! info "这是 SQL / ODPS 驱动的校验器"

    真实的 `UATConsistencyChecker` **不读取 CSV**，而是通过 `ODPSRunner` 执行两个 SQL 文件（线上、线下）拉数 → 合并 → 比对，并输出 Excel 报告。

## 1. 准备：SQL 文件 + ODPSRunner

在 `sql_dir` 目录下准备两个 SQL 文件，分别拉取**线下**与**线上**结果。二者都以 `flow_id` 为主键，并包含主模型分列与各入模特征（线上线下列名需一致）：

```
sql/
├── pull_offline.sql   # 线下打分结果（flow_id, 主模型分, 特征...）
└── pull_online.sql    # 线上打分结果（同结构）
```

## 2. 配置 —— `UATConfig`

```python
from Modeling_Tool.UAT.UAT_Consistency_Checker import UATConfig

config = UATConfig(
    main_model_score_col="credit_risk_ltrs_subomdel_score",
    sql_dir="sql",
    offline_sql="pull_offline.sql",
    online_sql="pull_online.sql",
    tol_score=1e-6,                  # 主/子模型分容差
    tol_feat=1e-2,                   # 特征容差
    info_list=["user_id", "curp"],   # 随报告输出的标识字段（不参与比对）
    time_featlist=["apply_time"],    # 需按秒级时间差比对的时间字段
    tol_time_seconds=60,
    excel_output_path="uat_report.xlsx",
)
```

### `UATConfig` 字段

| 字段 | 默认值 | 说明 |
|------|-------|------|
| `main_model_score_col` | `"credit_risk_ltrs_subomdel_score"` | 主模型分列名（线上/线下 SQL 同名） |
| `sql_dir` | `"sql"` | SQL 文件目录 |
| `offline_sql` / `online_sql` | `"pull_offline.sql"` / `"pull_online.sql"` | 线下/线上 SQL 文件名 |
| `tol_score` | `1e-6` | 主/子模型分比较容差 |
| `tol_feat` | `1e-2` | 特征变量比较容差 |
| `n_process` | `cpu_count-1` | SQL 并发拉取进程数 |
| `include_submodel_scores` | `True` | 子模型分是否已作为特征被覆盖（`False` 时需填 `submodel_pairs`） |
| `submodel_pairs` | `{}` | 子模型分列对 `{offline_col: online_col}` |
| `info_list` | `[]` | 随报告输出的标识字段（如 `user_id`/`curp`），不参与比对 |
| `time_featlist` | `[]` | 需按时间语义比对的时间字段 |
| `tol_time_seconds` | `60.0` | 时间差容差（秒） |
| `excel_font` | `"Arial"` | Excel 报告字体 |
| `excel_output_path` | 带时间戳文件名 | Excel 报告输出路径 |

## 3. 一键运行 —— `run()`

`run()` 按顺序执行全流程，返回汇总表，并落地 Excel 报告：

```python
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
from Modeling_Tool.UAT.UAT_Consistency_Checker import UATConsistencyChecker

checker = UATConsistencyChecker(config, ODPSRunner())
summary_df = checker.run()   # 拉数→覆盖度→主模型分→子模型→全量特征→时间字段→Per-Flow→汇总→Excel
print(summary_df)
```

`run()` 内部等价于按顺序调用：

```python
checker.load_data()                # §1 执行两个 SQL，合并线上线下
checker.check_coverage()           # §2 覆盖度（仅线上/仅线下/双方都有）
checker.check_main_score()         # §3 主模型分一致性
checker.check_submodel_features()  # §5 子模型专项（可选）
checker.check_all_features()       # §6 全量特征一致性
checker.check_time_fields()        # §7 时间字段
checker.build_per_flow_report()    # §8 逐 flow_id 明细
checker.build_summary()            # §9 汇总
checker.export_excel()             # §10 Excel 输出
```

## 4. 分步执行与取中间结果

```python
from Modeling_Tool.Core.ODPS_Tool import ODPSRunner
from Modeling_Tool.UAT.UAT_Consistency_Checker import UATConsistencyChecker

checker = UATConsistencyChecker(config, ODPSRunner())
checker.load_data()

coverage   = checker.check_coverage()      # dict
main_score = checker.check_main_score()    # dict
feat_df    = checker.check_all_features()   # DataFrame（逐特征一致率）

report_path = checker.export_excel()        # 返回 Excel 路径
print(f"报告已生成：{report_path}")
```

## 5. 子模型 / 时间字段校验

- **子模型分**：若 `include_submodel_scores=False`，在 `submodel_pairs` 中给出离线↔线上列名映射，`check_submodel_features()` 会按 `tol_score` 比对。
- **时间字段**：`time_featlist` 中的字段会用 `pd.to_datetime` 解析后按 `tol_time_seconds` 秒级容差比较，并从数值特征比对中排除。

## 常见问题

??? question "`RuntimeError: Data not loaded. Call load_data() first.`"

    调用 `check_*` / `build_*` / `export_excel` 前必须先 `load_data()`（或直接用 `run()`）。

??? question "SQL 文件找不到"

    `sql_dir` + `offline_sql`/`online_sql` 拼出的路径必须存在；`sql_dir` 为绝对路径或相对当前工作目录。

??? question "线上线下列名对不齐"

    合并后线上列自动加 `_online` 后缀，线下列保持原名；主模型分列与特征列要求线上线下**同名**。
