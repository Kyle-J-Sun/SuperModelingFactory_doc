# Excel Report Generation

[`ExcelMaster`](../api/excelmaster.md) is a general-purpose Excel writing engine, and [`Report`](../api/report.md) holds the templates specific to risk control. This section shows how to produce professional reports in a modeling pipeline.

## 1. ExcelMaster Core Concepts

### Three-Layer Class Inheritance

```
ExcelFormat    (ExcelFormatTool.py)   50+ preset cell formats
    │
    ▼
ExcelWorkbook  (ExcelMaster.py)       workbook level: conditional formatting / borders / chart scaffolding
    │
    ▼
ExcelMaster    (ExcelMaster.py)       worksheet level: cursor streaming / DataFrame / charts
```

### Cursor Tracking Mode

Each write operation automatically advances `curr_row` / `curr_col`, so you never have to compute coordinates by hand:

```python
em.write_dataframe(ws, df1, title="Metrics")    # cursor moves down after writing
em.write_dataframe(ws, df2, title="Results")    # placed right below df1
```

Use `skipby="col"` to advance by column instead; use `gap_number` to control the spacing.

### Preset Formats

50+ aliased formats can be referenced directly:

| Category | Aliases | Effect |
|------|------|------|
| Titles | `H1`, `H2`, `H3`, `H4` | 18/16/14/12 pt bold |
| Colored titles | `BLUE_H1~4`, `ORANGE_H1~4`, `GREEN_H1~4` | Blue/orange/green background titles |
| Highlight | `YELLOW_BG` | Yellow background |
| Numbers | `NUM`, `NUM%.1`, `NUM%.2`, `NUM%.4` | Decimal percentages |
| Separators | `COMMA` | Thousands separator |
| Borders | `----` | Full border |

## 2. Basic Usage

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster("report.xlsx", verbose=False)
ws = em.add_worksheet("Performance", zoom_perc=100)

# 1) Write a title in merged cells
em.merge_col(ws, ncols=5, text="LightGBM Model Performance", cformat="BLUE_H2")

# 2) Write a DataFrame (the cursor advances automatically)
em.write_dataframe(
    ws, perf,
    title="Performance Metrics",
    titleformat="BLUE_H2",
    headerformat="ORANGE_H4",
    valueformat="NUM%.4",
)

# 3) Insert an image
em.insert_image(ws, "roc_curve.png", figScale=(600, 400))

# 4) Dual-Y-axis combo chart
em.write_duo_chart(
    ws, chart_df,
    y1_list=["bad_count", "good_count"],
    y2_list=["bad_rate"],
    x="score_bin",
    c1_type="column",
    c2_type="line",
    title="Score Distribution and Bad Rate",
    chart_size=(800, 400),
)

# 5) Conditional formatting (heat map)
em.set_color_scale(ws, [3, 2, 7, 5], colors=("#F8696B", "#FFEB84", "#63BE7B"))

em.close_workbook()
```

## 3. Report Template Functions

`Report/Report_Tool.py` provides high-level templates specific to risk control.

### Single-Model Performance Report

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import single_model_perf

em = ExcelMaster("report.xlsx", verbose=False)
ws = em.add_worksheet("LGB Performance")

single_model_perf(
    em, ws,
    fig_path="./output/lgb_perf.jpg",
    res_path="./output/lgb_perf.csv",
    model_name="LGB",
    image_size=(600, 400),
    text="LightGBM Model Performance Evaluation",
)
em.close_workbook()
```

It writes automatically: **image → performance CSV (with Top10%_Lift and AUC_Shift computed automatically)**

### Multi-Model Comparison Report

```python
from Report.Report_Tool import get_multi_model_perf_report

get_multi_model_perf_report(
    em, ws,
    eval_img_path="./output/eval_img/",
    eval_res_path="./output/eval_res/",
)
```

### Bulk WOE Plot Report

```python
from Report.Report_Tool import get_woe_plot_report_new

get_woe_plot_report_new(
    em, ws,
    woe_plot_dir="./output/woe_plot/",
    grp_name="month",
    varlist=features,
)
```

The directory must contain `{var}.png` (the reference WOE plot) and `{var}_{month}.png` (the by-group comparison plot).

### Final Model Report

```python
from Report.Report_Tool import get_fnl_model_report

get_fnl_model_report(em, ws, result_dir="./output/final/")
```

## 4. Full Pipeline — Output a Modeling Report

```python
"""Modeling → evaluation → report, one-click script"""
from ExcelMaster.ExcelMaster import ExcelMaster
from Modeling_Tool import (
    PerformanceEvaluator, GainsTableCalculator, GradientBoostingModel,
)
from Report.Report_Tool import (
    single_model_perf, get_multi_model_perf_report,
    get_woe_plot_report_new, get_multi_model_varimp,
)

# Assumes training results already exist
em = ExcelMaster("model_evaluation_report.xlsx", verbose=False)

# ---- Sheet 1: Single-model performance ----
ws1 = em.add_worksheet("LGB Performance")
single_model_perf(
    em, ws1,
    fig_path="./output/lgb_roc.jpg",
    res_path="./output/lgb_perf.csv",
    model_name="LightGBM",
    image_size=(600, 400),
)

# ---- Sheet 2: Multi-model comparison ----
ws2 = em.add_worksheet("Model Comparison")
get_multi_model_perf_report(
    em, ws2,
    eval_img_path="./output/eval_img/",
    eval_res_path="./output/eval_res/",
)

# ---- Sheet 3: Variable importance ----
ws3 = em.add_worksheet("Variable Importance")
get_multi_model_varimp(
    em, ws3,
    raw_varimp="./output/varimp_raw.csv",
    woe_varimp="./output/varimp_woe.csv",
)

# ---- Sheet 4: WOE analysis ----
ws4 = em.add_worksheet("WOE Analysis")
get_woe_plot_report_new(
    em, ws4,
    woe_plot_dir="./output/woe_plot/",
    grp_name="apply_month",
    varlist=features,
)

em.close_workbook()
print("Generated model_evaluation_report.xlsx")
```

## 5. Custom Report Templates

See how `get_pva_report` is written in `ExcelMaster/Template.py`:

```python
from ExcelMaster.ExcelMaster import ExcelMaster
import pandas as pd

def my_var_perf_report(em: ExcelMaster, ws, data: pd.DataFrame, var_name: str):
    em.gap_number = 1

    # Title
    em.merge_col(ws, ncols=5, text=f"{var_name} Univariate Performance", cformat="BLUE_H2")

    # DataFrame
    em.write_dataframe(ws, data, title="Performance Summary",
                       titleformat="ORANGE_H3",
                       headerformat="ORANGE_H4",
                       valueformat="NUM%.4")

    # Plot (if the PNG has already been generated)
    png_path = f"./output/perf/{var_name}.png"
    if os.path.exists(png_path):
        em.insert_image(ws, png_path, figScale=(800, 400))

    return em.get_curr_loc()
```

## 6. Customizing Fonts / Colors

```python
# Add a custom format
em.add_new_format(
    {"font_name": "Microsoft YaHei", "font_size": 12, "bold": True, "bg_color": "#FFE699"},
    "MY_TITLE",
)
em.merge_col(ws, ncols=5, text="Custom Title", cformat="MY_TITLE")

# Replace the font everywhere
from Modeling_Tool.UAT.UAT_Consistency_Checker import _apply_excel_font
_apply_excel_font(em, "SimSun")
```

## FAQ

??? question "Chinese characters appear as empty boxes"

    Install a CJK font:

    ```bash
    sudo apt install fonts-noto-cjk fonts-wqy-zenhei
    fc-cache -fv
    ```

    Or install one from the project's bundled `ref_font/` directory.

??? question "Chart source data pollutes the main sheet"

    ExcelMaster writes chart data into a **hidden worksheet** `__CHRT_DATA_<N>` by default, keeping the main sheet clean.
    To inspect it, unhide it:

    ```python
    em.worksheets()["__CHRT_DATA_1"].show()
    ```

??? question "Row numbers don't start from 0 when writing a DataFrame"

    `write_dataframe(df, index=True)` writes the index as the first column;
    `index=False` does not write the index.

??? question "The file is locked after closing"

    `close_workbook()` releases it automatically. If you force it open, use a `with` context:

    ```python
    with ExcelMaster("report.xlsx", verbose=False) as em:
        ws = em.add_worksheet("Performance")
        em.write_dataframe(ws, df)
    ```
