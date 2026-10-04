# Excel Report Generation

[`ExcelMaster`](../api/excelmaster.md) is a cursor-based Excel writer built on [xlsxwriter](https://xlsxwriter.readthedocs.io/).
Every write starts at the current cursor position and moves the cursor past what it wrote, so you can stack DataFrames,
images, charts, and text on a sheet without computing cell coordinates. [`Report`](../api/report.md) builds on it with
ready-made sheet layouts for the artifacts of a modeling run: performance tables and plots, WOE plots, and variable
importance.

Both are top-level packages installed with SMF, next to `Modeling_Tool`:

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import single_model_perf
```

Every snippet on this page runs top to bottom in one Python session and one working directory, on synthetic data. The
snippets write `.xlsx` files and an `output/` folder into that directory.

## 1. ExcelMaster Core Concepts

### Class layers

```text
ExcelFormat    (ExcelFormatTool.py)   creates the workbook and registers the preset cell formats
    │
    ▼
ExcelWorkbook  (ExcelMaster.py)       workbook-level helpers: conditional formats, borders, chart scaffolding
    │
    ▼
ExcelMaster    (ExcelMaster.py)       worksheet-level API: cursor, tables, text, images, charts
```

You only use `ExcelMaster`. `em.workbook` is the underlying `xlsxwriter.Workbook` and `add_worksheet` returns a plain
xlsxwriter worksheet, so every xlsxwriter feature stays available on both.

### Lifecycle

- `ExcelMaster(filepath, verbose, gap_number=2, init_loc=(0, 0))` opens the output file. `verbose` has no default, so
  pass it explicitly (`True` logs every write through `logging.info`).
- `em.add_worksheet(name)` adds a sheet and returns the worksheet. It also resets the cursor to `(0, 0)`, which replaces
  `init_loc`, unless you pass `reset_loc=False`.
- `em.close_workbook()` saves the workbook. The constructor creates the file empty and `close_workbook()` fills it, so
  a script that never calls it leaves a 0-byte file.

`ExcelMaster` is not a context manager. To save the workbook even when the code in between fails, use `try` / `finally`:

```python
em = ExcelMaster("lifecycle.xlsx", verbose=False)
try:
    ws = em.add_worksheet("Sheet", cell_scale=False)
    em.write_text_content(ws, input_text="{HEADER_2} Report\n")
finally:
    em.close_workbook()
```

### Cursor

`ExcelMaster` keeps a zero-based cursor in `em.curr_row` and `em.curr_col`. Every write method starts at the cursor, then
moves it past the block it wrote plus a gap.

| To... | Use |
|---|---|
| Stack blocks downward (the default) | `skipby="row"` |
| Place the next block to the right | `skipby="col"` |
| Start one call at a fixed position | `loc=(row, col)` |
| Read or move the cursor | `em.get_curr_loc()`, `em.reset_curr_loc((row, col))` |
| Change the spacing | `em.gap_number = n` |

!!! note "`gap_number`"

    `gap_number` is the number of blank rows (or columns) between consecutive blocks. The constructor stores
    `gap_number + 1`, so the default `ExcelMaster(..., gap_number=2)` leaves three blank rows. Assigning the attribute
    (`em.gap_number = 1`) sets exactly that many.

### Return values

The writing methods return `0`; the chart methods return the chart objects instead when you pass `retChart=True`. Pass
`retCellRange="value"` to get `[first_row, first_col, last_row, last_col]` of what was written, or `retCellRange="text"` to
get an A1-style string such as `"A1:D6"`. Use the result to format the block you just wrote with `set_cell_format`,
`set_color_scale`, or `set_data_bar`. For images and charts the range is approximate: it can extend one row and column
past the object.

### Cell counts and scale factors

Sizes use two conventions:

| Argument | Unit |
|---|---|
| `figScale=(x, y)` in `insert_image` | Scale factors for the picture's width and height. `1` keeps the original size and `0.8` shrinks it by 20 percent |
| `chart_size=(rows, columns)` in `write_chart` and `write_duo_chart` | Worksheet cells. A cell is 20 px high and 64 px wide by default |
| `image_size=(rows, columns)` in the [Report functions](#3-report-template-functions) | Worksheet cells. The image file is resized to fit |

### Preset formats

Every method that takes a format (`cformat`, `titleformat`, `headerformat`, `valueformat`) accepts a name from
`em.dict_cell_format`, which holds 87 names; many are aliases of the same format. List them with
`sorted(em.dict_cell_format)`. An unknown name raises `KeyError`. Commonly used names:

| Names | Effect |
|---|---|
| `BLUE_H1`, `BLUE_H2`, `BLUE_H3`, `BLUE_H4` | Bold, centered, bordered title on a light-blue background (`#C5D9F1`) with dark-blue text; 18, 16, 14, and 12 pt |
| `ORANGE_H1` ... `ORANGE_H4` | The same sizes on an orange background (`#FABF8F`) with black text |
| `HEADER_1` ... `HEADER_4` (aliases `#`, `##`, `###`, `####`) | Bold left-aligned heading text without a background; 18, 16, 14, and 12 pt |
| `TABLE_HEADER` (alias `HEADER`) | Bold, centered, bordered table header. Default `headerformat` |
| `----` (aliases `BORDER`, `TEXT_NO_FORMAT_BORDER`) | Plain cell with a border. Default `valueformat` |
| `----C` (alias `BORDER_CENTER`) | Plain bordered cell, centered |
| `NUM%.1`, `NUM%.2`, `NUM%.3`, `NUM%.4` | Bordered percentage with 1 to 4 decimals (`0.0%` ... `0.0000%`) |
| `NUM_COMMA` (alias `NUM,`) | Bordered integer with a thousands separator (`#,##0`) |
| `B`, `I`, `U`, `BU`, `IU`, `BIU` | Bold, italic, underline, and their combinations |
| `RED`, `**RED` | Red text; bold red text |
| `YELLOW_BG` | Yellow highlight |

!!! warning "Apply percentage formats to the right columns"

    `valueformat` applies to every value of the table, so `valueformat="NUM%.1"` would also turn a count column into a
    percentage (`4200` shows as `420000.0%`). Keep the default `valueformat` and format only the rate columns with
    `set_cell_format`, as in the next section.

## 2. Basic Usage

The example builds a small report from a metrics table, a figure, and score-band counts.

```python
import matplotlib
matplotlib.use("Agg")                       # headless backend; not needed in a notebook
import matplotlib.pyplot as plt
import pandas as pd

from ExcelMaster.ExcelMaster import ExcelMaster

perf = pd.DataFrame({
    "sample": ["train", "test"],
    "N": [4200, 1800],
    "bad_rate": [0.081, 0.084],
    "KS": [0.44, 0.41],
})
bands = pd.DataFrame({
    "score_bin": [f"B{i}" for i in range(1, 6)],
    "bad_count": [5, 12, 30, 55, 90],
    "good_count": [95, 88, 70, 45, 10],
})
bands["bad_rate"] = bands["bad_count"] / (bands["bad_count"] + bands["good_count"])

plt.figure(figsize=(4, 3))
plt.plot([0, 1], [0, 1])
plt.savefig("roc.png", dpi=100)
plt.close()

em = ExcelMaster("report.xlsx", verbose=False)               # `verbose` is required
ws = em.add_worksheet("Performance", cell_scale=False)       # returns the xlsxwriter worksheet

# 1) A title across merged cells
em.merge_col(ws, ncols=8, text="Model performance summary", cformat="BLUE_H2")

# 2) A DataFrame with a title row; the cursor moves below it
box = em.write_dataframe(
    ws, perf,
    title="KS and bad rate",
    titleformat="BLUE_H4",
    headerformat="ORANGE_H4",
    retCellRange="value",                                    # [first_row, first_col, last_row, last_col]
)
# The data rows start two rows below the first row (title + header). Format the rate columns only.
em.set_cell_format(ws, [box[0] + 2, 2, box[2], 3], "NUM%.1")

# 3) A text line; end it with a newline so the next block starts below it
em.write_text_content(ws, input_text="{I} Metrics are computed on the hold-out sample.\n")

# 4) An image (scale factors, not pixels)
em.insert_image(ws, "roc.png", figScale=(0.8, 0.8))

# 5) A column + line chart on two y axes (chart_size is rows x columns of cells)
em.write_duo_chart(
    ws, bands,
    y1_list=["bad_count", "good_count"],
    y2_list=["bad_rate"],
    x="score_bin",
    c1_type="column",
    c2_type="line",
    y1_axis_range=None,                                      # automatic scale for the counts
    y2_axis_range=(0, 1),
    title="Score bands: counts and bad rate",
    chart_size=(15, 9),
)

# 6) Another table with a heat map on its bad-rate column
rng = em.write_dataframe(ws, bands, title="Score bands", retCellRange="value")
rate_col = [rng[0] + 2, 3, rng[2], 3]
em.set_cell_format(ws, rate_col, "NUM%.1")
em.set_color_scale(ws, rate_col, colors=("#63BE7B", "#FFEB84", "#F8696B"))   # low = green, high = red

em.close_workbook()
```

!!! warning "`y1_axis_range` defaults to `(0, 1)`"

    `write_duo_chart` fixes the primary axis to 0 to 1 unless you pass `y1_axis_range`, which suits rates but clips counts.
    Pass `None` for an automatic scale or explicit `(min, max)` limits. `y2_axis_range` defaults to the primary range.

!!! tip "Faster, smaller workbooks: `cell_scale`"

    By default `add_worksheet` writes an explicit size for all 1,048,576 rows (`cell_scale=True`). That takes a few
    seconds and adds about 2.5 MB per sheet, and it changes nothing at the default scale, because the explicit size
    (20 px by 64 px) equals Excel's default. Pass `cell_scale=False` to skip it. Pass a tuple such as `(1, 2)` to scale
    the row heights and column widths instead (here: columns twice as wide).

The methods used above:

| Method | Purpose |
|---|---|
| `add_worksheet(name, hide_grid=True, reset_loc=True, cell_scale=True, auto_fit=False, zoom_perc=100, tab_color=None)` | Add a sheet and, by default, reset the cursor |
| `merge_col(worksheet, loc=None, nrows=1, ncols=1, text='', skipby='row', cformat='BLUE_H4', retCellRange=None)` | Merge cells and write a heading |
| `write_dataframe(worksheet, df, loc=None, title=None, index=False, header=True, skipby='row', titleformat='BLUE_H4', headerformat='TABLE_HEADER', valueformat='----', retCellRange=None)` | Write a DataFrame with an optional title row. `index=True` writes the index as the first column |
| `write_text_content(worksheet, input_text=None, txt_path=None, loc=None, retCellRange=None)` | Write multi-line text. Start a line with `{FORMAT_NAME}` to style it |
| `insert_image(worksheet, figPath, figScale=(1, 1), loc=None, skipby='row', retCellRange=None)` | Insert a picture |
| `write_chart(worksheet, df, y_list, x=None, title='', chart_size=(30, 13), chart_type='line', ...)` | One chart; `chart_type` is `'line'`, `'column'`, `'stacked_column'`, or `'pie'`. `outputData=True` writes the chart data above the chart |
| `write_duo_chart(worksheet, df, y1_list, y2_list=None, x=None, c1_type='column', c2_type='line', y1_axis_range=(0, 1), y2_axis_range=None, ..., title='', chart_size=(30, 13))` | Two chart types on one x axis, with a secondary y axis |
| `set_color_scale(worksheet, cell_range, colors=('#F8696B', '#FFEB84', '#63BE7B'))` | Two- or three-color scale; `cell_range` is `"B2:B6"` or `[first_row, first_col, last_row, last_col]` |
| `set_data_bar(worksheet, cell_range, bar_color='#63C384')` | Data bars |
| `set_cell_format(worksheet, cell_range, cformat, cell_condition=None)` | Apply a preset or custom format, optionally only where `cell_condition=(">", 0.2)` or `("between", (0.1, 0.3))` holds |
| `set_border_line(worksheet, valuerange, border_line=1)` | Draw borders around every cell of a range |
| `add_new_format(format_dict, format_name)` | Register a custom format |

The [ExcelMaster API reference](../api/excelmaster.md) lists every parameter.

## 3. Report Template Functions

`Report.Report_Tool` takes the artifacts of a modeling run (performance CSVs and plots, WOE plots, variable-importance
tables) and lays them out as sheets. Every function takes an `ExcelMaster` instance `em` and a worksheet `ws` first and
writes at the cursor. They do no modeling themselves.

### Example artifacts

The templates need fixed inputs, so this block creates them: a synthetic sample, WOE encoding, five models, and one
evaluation figure plus metrics table per model.

```python
import os

import numpy as np
import pandas as pd

from Modeling_Tool import SampleSplitter, WOE_Master, GradientBoostingModel, LRMaster, PerformanceEvaluator

rng = np.random.default_rng(42)
n = 6000
data = pd.DataFrame({
    "apply_month": rng.choice([f"2025-{m:02d}" for m in range(1, 7)], n),
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
})
logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"] - 0.8 * data["utilization"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
features = ["age", "income", "score_b", "utilization", "n_overdue"]

train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True).split_df(data, target="bad_flag")
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag", graph_save_dir="output")
woe.fit(nbins=10, equal_freq=True)
train_woe, test_woe = woe.transform(train_df), woe.transform(test_df)
woe_features = [f"{f}_woe" for f in features]

# Five models: XGBoost and LightGBM on the raw and the WOE features, and a logistic regression on the WOE features
gbm_params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "early_stopping_rounds": 20, "eval_metric": "auc"}
models = {
    "xgb_original": GradientBoostingModel("xgb", dict(gbm_params)),
    "lgb_original": GradientBoostingModel("lgb", {**gbm_params, "verbose": -1}),
    "xgb_woe":      GradientBoostingModel("xgb", dict(gbm_params)),
    "lgb_woe":      GradientBoostingModel("lgb", {**gbm_params, "verbose": -1}),
    "lr_woe":       LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}),
}
columns_and_samples = {                                  # model -> (feature columns, train frame, test frame)
    "xgb_original": (features, train_df, test_df),
    "lgb_original": (features, train_df, test_df),
    "xgb_woe":      (woe_features, train_woe, test_woe),
    "lgb_woe":      (woe_features, train_woe, test_woe),
    "lr_woe":       (woe_features, train_woe, test_woe),
}

# Fit each model, then save one figure (.jpg) and one metrics table (.csv) for it
os.makedirs("output/eval_img", exist_ok=True)
os.makedirs("output/eval_res", exist_ok=True)
for name, model in models.items():
    cols, train, test = columns_and_samples[name]
    if name == "lr_woe":
        model.fit(train, cols, "bad_flag")
    else:
        model.fit(train[cols], train["bad_flag"], test[cols], test["bad_flag"])
    (PerformanceEvaluator(tgt_name="bad_flag", model=model, feature_cols=cols)
     .add_dataset("train", train).add_dataset("test", test)
     .evaluate(fig_save_path=f"output/eval_img/{name}_perf.jpg",
               rpt_save_path=f"output/eval_res/{name}_perf.csv", display=False))

print(sorted(os.listdir("output/eval_img")))
```

The WOE plot report needs one more input: the plots written by `WOE_Master.plot_bivar_graph`, which are
`output/woe_plot/{var}.png` and `output/woe_plot/{var}_apply_month.png` for every variable.

```python
woe.plot_bivar_graph(data=train_df, group="apply_month", dirname="woe_plot")
print(sorted(os.listdir("output/woe_plot")))
```

### Single-model performance

`single_model_perf` writes a heading (optional), the performance figure, and the metrics table, and returns the cell
ranges of the figure and the table.

```python
import shutil

from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import single_model_perf

shutil.copy("output/eval_img/lgb_woe_perf.jpg", "output/lgb_perf.jpg")     # the function resizes the image in place

em = ExcelMaster("single_model.xlsx", verbose=False)
ws = em.add_worksheet("LightGBM", cell_scale=False)
img_loc, df_loc = single_model_perf(
    em, ws,
    fig_path="output/lgb_perf.jpg",
    res_path="output/eval_res/lgb_woe_perf.csv",
    model_name="LightGBM",
    image_size=(25, 12),                         # (rows, columns) in worksheet cells, not pixels
    text="{##} LightGBM performance\n",          # optional heading; end it with a newline
)
em.close_workbook()
print(img_loc, df_loc)                           # [first_row, first_col, last_row, last_col] each
```

- `fig_path` is resized **in place** to `image_size`, so pass a copy if you need the original.
- `res_path` is the unweighted table written by `PerformanceEvaluator.evaluate(rpt_save_path=...)`. It must contain
  `avgTrue`, `Top10%_TargetRate`, and `AUC`; the weighted table lacks `Top10%_TargetRate` and raises `KeyError`.
- The table is titled `Performance for <model_name>`, rounded to three decimals, with `Top10%_Lift` and `AUC_Shift`
  recomputed (`AUC_Shift` is the previous row's AUC divided by this row's, minus 1).
- `text` goes through `write_text_content`, which leaves the cursor on its last line. Without the trailing `\n` the
  image is placed over the heading.

### Multi-model comparison

`get_multi_model_perf_report` lays out five models in two rows: XGBoost and LightGBM on the original features, then
logistic regression, XGBoost, and LightGBM on the WOE features.

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import get_multi_model_perf_report

em = ExcelMaster("multi_model.xlsx", verbose=False)
ws = em.add_worksheet("Model Comparison", cell_scale=False)
get_multi_model_perf_report(em, ws, eval_img_path="output/eval_img", eval_res_path="output/eval_res")
em.close_workbook()
```

The file names are fixed. `eval_img_path` must hold `xgb_original_perf.jpg`, `lgb_original_perf.jpg`,
`lr_woe_perf.jpg`, `xgb_woe_perf.jpg`, and `lgb_woe_perf.jpg`, and `eval_res_path` the five matching `.csv` files. The
function resizes the images in place.

### WOE plot report

`get_woe_plot_report_new` puts, for every variable, the overall WOE plot next to the plot by group.

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import get_woe_plot_report_new

means_rpt = train_df[features].describe().T.rename_axis("attribute").reset_index()    # optional: one row per variable

em = ExcelMaster("woe_report.xlsx", verbose=False)
ws = em.add_worksheet("WOE Analysis", cell_scale=False)
get_woe_plot_report_new(
    em, ws,
    woe_plot_dir="output/woe_plot",
    grp_name="apply_month",
    varlist=features,
    means_rpt=means_rpt,
)
em.close_workbook()
```

- The directory holds `{var}.png` (the overall plot) and `{var}_{grp_name}.png` (the plot by group), as written by
  `WOE_Master.plot_bivar_graph`.
- A variable is skipped when `{var}_{grp_name}.png` is missing. For every other variable `{var}.png` must exist too, or
  the call raises `FileNotFoundError`.
- `means_rpt` is a DataFrame with an `attribute` column and one row per variable, such as the output of
  [`proc_means_odps`](odps.md#5-proc_means_odps-odps-side-descriptive-statistics). Its row for the variable is written to
  the right of the plots as `Means for <var>`.
- Column A repeats the variable name on every row of its block.
- The images are resized in place.

### Final model report

`get_fnl_model_report` writes one fixed layout for a final XGBoost model: the heading and the table title say
`XGBoost (Without Monotonic Constraints)` and `XGBoost (Without MC)`, and the file names are fixed. For any other model
use `single_model_perf`.

```python
import os
import shutil

from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import get_fnl_model_report

os.makedirs("output/final", exist_ok=True)
shutil.copy("output/eval_img/xgb_original_perf.jpg", "output/final/xgb_fnl_model_perf.jpg")
shutil.copy("output/eval_res/xgb_original_perf.csv", "output/final/xgb_fnl_model_perf.csv")

em = ExcelMaster("final_model.xlsx", verbose=False)
ws = em.add_worksheet("Final Model", cell_scale=False)
get_fnl_model_report(em, ws, result_dir="output/final")
em.close_workbook()
```

### Variable importance

`get_model_varimp` writes one importance table. `get_multi_model_varimp` writes two side by side: the original features
and the WOE features. Both take DataFrames, not file paths, and `get_multi_model_varimp` skips an argument that is
`None`.

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import get_model_varimp, get_multi_model_varimp


def importance(model, prefix):
    """Columns `variable`, `<prefix>_varimp`, and `<prefix>_rank` from a fitted gradient boosting model."""
    table = model.get_feature_importance().rename(columns={"feature": "variable", "importance": f"{prefix}_varimp"})
    table[f"{prefix}_rank"] = table[f"{prefix}_varimp"].rank(ascending=False, method="first").astype(int)
    return table


raw_varimp = importance(models["xgb_original"], "xgb").merge(importance(models["lgb_original"], "lgb"), on="variable")

lr_table = models["lr_woe"].get_variable_importance().rename(columns={"varlist": "variable", "coef": "coefficient"})
lr_table["lr_rank"] = lr_table["importance"].rank(ascending=False, method="first").astype(int)
woe_varimp = (
    lr_table[["variable", "lr_rank", "coefficient"]]
    .merge(importance(models["xgb_woe"], "xgb"), on="variable")
    .merge(importance(models["lgb_woe"], "lgb"), on="variable")
)

em = ExcelMaster("varimp.xlsx", verbose=False)
if "CUS_#" not in em.dict_cell_format:                  # the heading format of both varimp templates
    em.add_new_format({"bold": True, "font_size": 18}, "CUS_#")

get_model_varimp(em, em.add_worksheet("Single", cell_scale=False), raw_varimp[["variable", "xgb_rank", "xgb_varimp"]])
get_multi_model_varimp(em, em.add_worksheet("Multi", cell_scale=False), raw_varimp=raw_varimp, woe_varimp=woe_varimp)
em.close_workbook()
```

- Both templates style their heading with a custom format named `CUS_#` that you must register first (`KeyError:
  'CUS_#'` otherwise). `get_multi_model_perf_report` and `get_fnl_model_report` register it themselves.
- `get_multi_model_varimp` selects fixed columns and rounds them to four decimals: `variable`, `xgb_rank`, `xgb_varimp`,
  `lgb_rank`, `lgb_varimp` for `raw_varimp`, and `variable`, `lr_rank`, `coefficient`, `xgb_rank`, `xgb_varimp`,
  `lgb_rank`, `lgb_varimp` for `woe_varimp`.
- `get_feature_importance()` returns split counts for XGBoost and gain for LightGBM, and ignores its `importance_type`
  argument. Compare the ranks across models, not the raw values.

### Native WOE chart and data dictionary

`plot_woe` draws a WOE chart as a native Excel chart (stacked good and bad counts, bad-rate line, and mean reference
lines) from a table of bins, and `write_var_info` writes one variable's row of a data dictionary and returns its
description.

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import plot_woe, write_var_info

woe_bins = woe.get_mapping_table().rename(columns={
    "N_BAD": "n1", "N_GOOD": "n0", "AVG_BAD": "tr", "IV": "iv", "BIN_RANGE": "bin_value", "VAR": "var_name",
})
data_dict = pd.DataFrame({
    "var_name": features,
    "description": ["Applicant age", "Monthly income", "Bureau score", "Credit utilization", "Overdue count"],
})

em = ExcelMaster("woe_chart.xlsx", verbose=False)
ws = em.add_worksheet("score_b", cell_scale=False)
description, info_loc = write_var_info(em, ws, "score_b", "var_name", data_dict, var_info_title="score_b")
chart_loc = plot_woe(em, ws, "score_b", woe_bins, x_col="bin_value", description=description)
em.close_workbook()
```

`woe_bins` needs the columns `n1` (bad count), `n0` (good count), `tr` (bad rate), `iv`, a label column named by `x_col`,
and a `var_name` column (rename it with `plot_woe(var_name=...)`). A bin whose label starts with
`[<spec_missing_value>` (default `-99999`) counts as the missing-value bin when the "mean without missing" line is
drawn. `data_dict` needs a `description` column and a column named by the `var_name` argument.

### Function reference

| Function | Reads | Writes |
|---|---|---|
| `single_model_perf(em, ws, fig_path, res_path, model_name, image_size, text=None)` | `fig_path` image; `res_path` CSV from `PerformanceEvaluator.evaluate(rpt_save_path=...)` | Optional heading, image, and metrics table; returns `(img_loc, df_loc)` |
| `get_multi_model_perf_report(em, ws, eval_img_path, eval_res_path)` | Five fixed `*_perf.jpg` and `*_perf.csv` pairs | Five models in two rows, original versus WOE features |
| `get_fnl_model_report(em, ws, result_dir)` | `{result_dir}/xgb_fnl_model_perf.jpg` and `.csv` | Final XGBoost model sheet |
| `get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt=None)` | `{woe_plot_dir}/{var}.png` and `{var}_{grp_name}.png` | One row per variable: overall plot, plot by group, optional means table |
| `get_woe_plot_report(em, ws, analysis_dir, varlist, means_rpt=None)` | `{analysis_dir}/woe_plot/{var}_woe.png` and `{var}_woe_group.png`; `numvars_woe.csv` and `numvars_woe_group.csv` in `analysis_dir` | Older layout of the same report |
| `get_model_varimp(em, ws, varimp)` | `varimp` DataFrame | One importance table |
| `get_multi_model_varimp(em, ws, raw_varimp=None, woe_varimp=None)` | Two DataFrames (columns above) | Importance of several models |
| `plot_woe(em, ws, var, woe_bins, x_col, spec_missing_value=-99999, chart_size=(20, 5), var_name='var_name', description='', skipby='row')` | `woe_bins` DataFrame | Native Excel WOE chart; returns its cell range |
| `write_var_info(em, ws, var, var_name, data_dict, var_info_title='', skipby='row')` | `data_dict` DataFrame | A variable's dictionary row; returns `(description, cell_range)` |

`Modeling_Tool.WOE.WOE_Report_Builder` contains a richer variant of `get_woe_plot_report_new` (it also writes a
one-line explanation above each variable, from `var_dict`) and a `WoeReportBuilder` class that builds one sheet per
grouping column.

!!! warning "Behavior shared by all Report functions"

    - **Gap.** Each function sets `em.gap_number` for its own layout and leaves it at `0` or `1`. Set it again if your
      code depends on a specific gap.
    - **Images.** The functions resize their image files in place.
    - **Chinese headings.** Several sections are titled with fixed Chinese strings that cannot be changed through
      arguments. The multi-model sheet is headed *multi-model evaluation (untuned version)* with the sub-headings *using
      third-party features directly* and *modeling after WOE processing of the features*. The final-model sheet is
      headed *final model evaluation* and *modeling with the original features*. Both variable-importance templates are
      headed *feature importance evaluation*. Tables, plots, and the other titles are in English.

## 4. Full Pipeline: Output a Modeling Report

This script assembles the artifacts of the previous section into one workbook with four sheets. It reuses `features`,
`raw_varimp`, `woe_varimp`, and the files in `output/` created above.

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import (
    single_model_perf, get_multi_model_perf_report, get_multi_model_varimp, get_woe_plot_report_new,
)

em = ExcelMaster("model_evaluation_report.xlsx", verbose=False)

# Sheet 1: single-model performance
ws1 = em.add_worksheet("LGB Performance", cell_scale=False)
single_model_perf(
    em, ws1,
    fig_path="output/lgb_perf.jpg",
    res_path="output/eval_res/lgb_woe_perf.csv",
    model_name="LightGBM",
    image_size=(25, 12),
    text="{##} LightGBM performance\n",
)

# Sheet 2: model comparison (also registers the CUS_# heading format)
ws2 = em.add_worksheet("Model Comparison", cell_scale=False)
get_multi_model_perf_report(em, ws2, eval_img_path="output/eval_img", eval_res_path="output/eval_res")

# Sheet 3: variable importance
ws3 = em.add_worksheet("Variable Importance", cell_scale=False)
get_multi_model_varimp(em, ws3, raw_varimp=raw_varimp, woe_varimp=woe_varimp)

# Sheet 4: WOE analysis
ws4 = em.add_worksheet("WOE Analysis", cell_scale=False)
get_woe_plot_report_new(em, ws4, woe_plot_dir="output/woe_plot", grp_name="apply_month", varlist=features)

em.close_workbook()
print("Generated model_evaluation_report.xlsx")
```

## 5. Custom Report Templates

A template is a function that takes `em` and `ws` and writes with the `ExcelMaster` calls. Built-in examples live in
`ExcelMaster.Template`: `get_pva_report`, `get_bivar_report`, `get_means_chart_report`, `get_grid_search_report`,
`get_grid_boxplot_report`, `get_var_reduct_report`, and `get_seg_perf_comparison_report`. Each expects specific input
tables; read the source of `get_pva_report` as a model for your own.

```python
import os

import pandas as pd

from ExcelMaster.ExcelMaster import ExcelMaster


def my_var_perf_report(em: ExcelMaster, ws, table: pd.DataFrame, var_name: str, png_path: str | None = None):
    """One block per variable: a heading, a metrics table, and an optional figure. Returns the cursor."""
    em.gap_number = 1                                    # one blank row between blocks

    em.merge_col(ws, ncols=table.shape[1], text=f"{var_name}: univariate performance", cformat="BLUE_H2")
    em.write_dataframe(
        ws, table,
        title="Performance summary",
        titleformat="ORANGE_H3",
        headerformat="ORANGE_H4",
    )
    if png_path and os.path.exists(png_path):            # add the figure only if it was generated
        em.insert_image(ws, png_path, figScale=(0.8, 0.8))

    return em.get_curr_loc()


bins = woe.get_mapping_table().query("VAR == 'score_b'")[["BIN_RANGE", "N", "AVG_BAD", "LIFT", "WOE", "IV"]]

em = ExcelMaster("custom_report.xlsx", verbose=False)
ws = em.add_worksheet("score_b", cell_scale=False)
print(my_var_perf_report(em, ws, bins, "score_b", "output/woe_plot/score_b.png"))
em.close_workbook()
```

## 6. Customizing Fonts and Colors

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster("custom_format.xlsx", verbose=False)
ws = em.add_worksheet("Custom", cell_scale=False)

# Register a custom format: a dictionary of xlsxwriter format properties and a new name
em.add_new_format({"font_name": "Arial", "font_size": 12, "bold": True, "bg_color": "#FFE699"}, "MY_TITLE")
em.merge_col(ws, ncols=5, text="Custom title", cformat="MY_TITLE")

# Replace the font of every format in the workbook. Call it after the last write and before close_workbook(),
# because pandas creates additional formats (for table headers and dates) while writing a DataFrame.
for fmt in em.workbook.formats:
    fmt.font_name = "Arial"

em.close_workbook()
```

`add_new_format` returns `0` on success. If the name is already taken, which includes every preset, it returns `1`,
logs a message, and keeps the existing format, so presets cannot be overridden.

## FAQ

??? question "The workbook is a 0-byte file or Excel reports it as corrupt"

    `close_workbook()` was never called, or the script stopped before reaching it. `ExcelMaster` creates the file when it
    is constructed and writes the content only on close. Call it once at the end, or wrap the report in `try` / `finally`
    (see [Lifecycle](#lifecycle)).

    Because the file is opened at construction, a path that cannot be written, for example a workbook that is still open
    in Excel on Windows, fails right there with an `OSError`. Close the file in Excel and run again.

??? question "My heading is hidden behind an image"

    `write_text_content` leaves the cursor on the last line it wrote. End the text with a newline so that the next block
    starts on the following row: `"{HEADER_2} Title\n"`, not `"{HEADER_2} Title"`.

??? question "Bars or lines are cut off in a chart"

    `write_duo_chart` fixes the primary axis to `(0, 1)` by default. Pass `y1_axis_range=None` for an automatic scale, or
    explicit limits. `write_chart` has no fixed default: its `y_axis_range=(None, None)` is automatic.

??? question "`add_worksheet` is slow and the file is large"

    The default `cell_scale=True` writes an explicit size for every row. Pass `cell_scale=False`; see
    [Basic Usage](#2-basic-usage).

??? question "Chinese text in the figures appears as empty boxes"

    This concerns figures only; text written to cells is stored as Unicode and drawn by Excel. matplotlib needs a font
    that contains the glyphs. Register one before you create the figures. SMF ships three Chinese fonts in
    `Modeling_Tool/ref_font/` (`KaiTi.ttf`, `WeiRuanYaHei.ttf`, and `simsun.ttc`):

    ```python
    from pathlib import Path

    import matplotlib
    from matplotlib import font_manager

    import Modeling_Tool

    font_file = Path(Modeling_Tool.__file__).parent / "ref_font" / "KaiTi.ttf"
    font_manager.fontManager.addfont(str(font_file))
    matplotlib.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_file)).get_name()]
    matplotlib.rcParams["axes.unicode_minus"] = False         # keep the minus sign readable with this font
    ```

    You can also install a system CJK font (for example `fonts-noto-cjk` on Debian or Ubuntu) and put its name first in
    `rcParams["font.sans-serif"]`.

??? question "Where did the chart data go?"

    Each chart's source data is written to a hidden worksheet named `__CHRT_DATA_<N>` (`N` starts at 0), which keeps the
    visible sheet clean. `write_duo_chart` creates two of them. The sheets are listed in `em.ws_dict`. To see one, right-click
    a sheet tab in Excel and choose Unhide, or call `em.ws_dict["__CHRT_DATA_0"].activate()` before `close_workbook()`.
    To keep the data on the visible sheet instead, use `write_chart(..., outputData=True)`, which writes it above the
    chart.

??? question "How do I write the DataFrame index?"

    `write_dataframe` omits the index by default. Pass `index=True` to write it as the first column (one column per
    index level), and `header=False` to omit the column names.

??? question "`KeyError` for a format name"

    The name is not in `em.dict_cell_format`. List the valid names with `sorted(em.dict_cell_format)`, or register your own
    with `em.add_new_format`. Valid names include `BLUE_H1` to `BLUE_H4`, `ORANGE_H1` to `ORANGE_H4`, `HEADER_1` to
    `HEADER_4`, and `NUM%.1` to `NUM%.4`; see [Preset formats](#preset-formats).
