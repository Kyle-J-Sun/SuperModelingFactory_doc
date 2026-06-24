# Excel 报告生成

[`ExcelMaster`](../api/excelmaster.md) 是通用 Excel 写入引擎，[`Report`](../api/report.md) 是风控专属模板。本节展示如何在建模流水线中输出专业级中文报告。

## 1. ExcelMaster 核心概念

### 三层类继承

```
ExcelFormat    (ExcelFormatTool.py)   50+ 预设单元格格式
    │
    ▼
ExcelWorkbook  (ExcelMaster.py)       工作簿级：条件格式 / 边框 / 图表基架
    │
    ▼
ExcelMaster    (ExcelMaster.py)       工作表级：光标流式 / DataFrame / 图表
```

### 光标追踪模式

每次写入操作自动推进 `curr_row` / `curr_col`，无需手工算坐标：

```python
em.write_dataframe(ws, df1, title="指标")    # 写入后光标下移
em.write_dataframe(ws, df2, title="结果")    # 紧接 df1 下方
```

通过 `skipby="col"` 改为按列推进；通过 `gap_number` 控制间距。

### 预设格式

50+ 别名格式直接引用：

| 类别 | 别名 | 效果 |
|------|------|------|
| 标题 | `H1`, `H2`, `H3`, `H4` | 18/16/14/12 pt 加粗 |
| 彩色标题 | `BLUE_H1~4`, `ORANGE_H1~4`, `GREEN_H1~4` | 蓝/橙/绿底标题 |
| 高亮 | `YELLOW_BG` | 黄色背景 |
| 数字 | `NUM`, `NUM%.1`, `NUM%.2`, `NUM%.4` | 小数百分比 |
| 分隔 | `COMMA` | 千分位 |
| 边框 | `----` | 全边框 |

## 2. 基本用法

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster("report.xlsx", verbose=False)
ws = em.add_worksheet("Performance", zoom_perc=100)

# 1) 合并单元格写标题
em.merge_col(ws, ncols=5, text="LightGBM 模型性能", cformat="BLUE_H2")

# 2) 写 DataFrame（自动推进光标）
em.write_dataframe(
    ws, perf,
    title="性能指标",
    titleformat="BLUE_H2",
    headerformat="ORANGE_H4",
    valueformat="NUM%.4",
)

# 3) 插图
em.insert_image(ws, "roc_curve.png", figScale=(600, 400))

# 4) 双 Y 轴组合图
em.write_duo_chart(
    ws, chart_df,
    y1_list=["bad_count", "good_count"],
    y2_list=["bad_rate"],
    x="score_bin",
    c1_type="column",
    c2_type="line",
    title="分数分布与坏账率",
    chart_size=(800, 400),
)

# 5) 条件格式（热力图）
em.set_color_scale(ws, [3, 2, 7, 5], colors=("#F8696B", "#FFEB84", "#63BE7B"))

em.close_workbook()
```

## 3. Report 模板函数

`Report/Report_Tool.py` 提供风控专用的高阶模板。

### 单模型性能报告

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import single_model_perf

em = ExcelMaster("report.xlsx", verbose=False)
ws = em.add_worksheet("LGB性能")

single_model_perf(
    em, ws,
    fig_path="./output/lgb_perf.jpg",
    res_path="./output/lgb_perf.csv",
    model_name="LGB",
    image_size=(600, 400),
    text="LightGBM 模型性能评估",
)
em.close_workbook()
```

自动写入：**图片 → 性能 CSV（含 Top10%_Lift、AUC_Shift 自动计算）**

### 多模型对比报告

```python
from Report.Report_Tool import get_multi_model_perf_report

get_multi_model_perf_report(
    em, ws,
    eval_img_path="./output/eval_img/",
    eval_res_path="./output/eval_res/",
)
```

### WOE 批量图报告

```python
from Report.Report_Tool import get_woe_plot_report_new

get_woe_plot_report_new(
    em, ws,
    woe_plot_dir="./output/woe_plot/",
    grp_name="month",
    varlist=features,
)
```

目录中需有 `{var}.png`（参考 WOE 图）和 `{var}_{month}.png`（分组对比图）。

### 终版模型报告

```python
from Report.Report_Tool import get_fnl_model_report

get_fnl_model_report(em, ws, result_dir="./output/final/")
```

## 4. 完整流水线 —— 输出建模报告

```python
"""建模 → 评估 → 报告 一键脚本"""
from ExcelMaster.ExcelMaster import ExcelMaster
from Modeling_Tool import (
    PerformanceEvaluator, GainsTableCalculator, GradientBoostingModel,
)
from Report.Report_Tool import (
    single_model_perf, get_multi_model_perf_report,
    get_woe_plot_report_new, get_multi_model_varimp,
)

# 假设已有训练结果
em = ExcelMaster("model_evaluation_report.xlsx", verbose=False)

# ---- Sheet 1: 单模型性能 ----
ws1 = em.add_worksheet("LGB性能")
single_model_perf(
    em, ws1,
    fig_path="./output/lgb_roc.jpg",
    res_path="./output/lgb_perf.csv",
    model_name="LightGBM",
    image_size=(600, 400),
)

# ---- Sheet 2: 多模型对比 ----
ws2 = em.add_worksheet("模型对比")
get_multi_model_perf_report(
    em, ws2,
    eval_img_path="./output/eval_img/",
    eval_res_path="./output/eval_res/",
)

# ---- Sheet 3: 变量重要性 ----
ws3 = em.add_worksheet("变量重要性")
get_multi_model_varimp(
    em, ws3,
    raw_varimp="./output/varimp_raw.csv",
    woe_varimp="./output/varimp_woe.csv",
)

# ---- Sheet 4: WOE 分析 ----
ws4 = em.add_worksheet("WOE分析")
get_woe_plot_report_new(
    em, ws4,
    woe_plot_dir="./output/woe_plot/",
    grp_name="apply_month",
    varlist=features,
)

em.close_workbook()
print("已生成 model_evaluation_report.xlsx")
```

## 5. 自定义报告模板

参考 `ExcelMaster/Template.py` 中 `get_pva_report` 的写法：

```python
from ExcelMaster.ExcelMaster import ExcelMaster
import pandas as pd

def my_var_perf_report(em: ExcelMaster, ws, data: pd.DataFrame, var_name: str):
    em.gap_number = 1

    # 标题
    em.merge_col(ws, ncols=5, text=f"{var_name} 单变量性能", cformat="BLUE_H2")

    # DataFrame
    em.write_dataframe(ws, data, title="性能汇总",
                       titleformat="ORANGE_H3",
                       headerformat="ORANGE_H4",
                       valueformat="NUM%.4")

    # 绑图（如果已生成 PNG）
    png_path = f"./output/perf/{var_name}.png"
    if os.path.exists(png_path):
        em.insert_image(ws, png_path, figScale=(800, 400))

    return em.get_curr_loc()
```

## 6. 字体 / 颜色自定义

```python
# 新增自定义格式
em.add_new_format(
    {"font_name": "微软雅黑", "font_size": 12, "bold": True, "bg_color": "#FFE699"},
    "MY_TITLE",
)
em.merge_col(ws, ncols=5, text="自定义标题", cformat="MY_TITLE")

# 统一替换字体
from Modeling_Tool.UAT.UAT_Consistency_Checker import _apply_excel_font
_apply_excel_font(em, "SimSun")
```

## 常见问题

??? question "中文显示为方框"

    安装中文字体：

    ```bash
    sudo apt install fonts-noto-cjk fonts-wqy-zenhei
    fc-cache -fv
    ```

    或从项目自带的 `ref_font/` 目录安装。

??? question "图表源数据污染主表"

    ExcelMaster 默认把图表数据写入**隐藏工作表** `__CHRT_DATA_<N>`，主表保持整洁。
    如需查看，取消隐藏：

    ```python
    em.worksheets()["__CHRT_DATA_1"].show()
    ```

??? question "写入 DataFrame 时行号不从 0 开始"

    `write_dataframe(df, index=True)` 会把索引作为第一列写入；
    `index=False` 不写索引。

??? question "关闭后文件被占用"

    `close_workbook()` 会自动释放。如强制打开，使用 `with` 上下文：

    ```python
    with ExcelMaster("report.xlsx", verbose=False) as em:
        ws = em.add_worksheet("Performance")
        em.write_dataframe(ws, df)
    ```
