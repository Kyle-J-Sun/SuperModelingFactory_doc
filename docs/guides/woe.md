# WOE Encoding

Weight of evidence (WOE) replaces each bin of a variable with `ln(share of all bads in the bin / share of all goods in the
bin)`. SMF uses this sign convention, so **a positive WOE marks a riskier bin**. The encoded column is on the log-odds scale
that logistic-regression scorecards expect, and the information value (IV) of a variable is the sum over its bins of
`(bad share - good share) * WOE`.

The [`Modeling_Tool.WOE`](../api/woe.md) subpackage provides two binning engines, helpers to apply and persist the bins,
plotting tools, and an adapter that lets screening and monitoring tools reuse a fitted engine.

| | `WOE_Master` | `MonotoneWOEBinner` |
|---|---|---|
| Features | Numeric | Numeric, and categorical through `cate_feats` |
| Binning | Equal-frequency or equal-width bins, optionally decision-tree or chi-square. Not forced to be monotone | Greedy merging until the WOE is monotone. Optional chi-square merging, decision-tree re-cutting, and clustering of categories |
| Special values | `spec_values` in `fit` | `special_values` in the constructor, with a `[Missing]` bin and governance options |
| Fit, then apply | `fit(...)`, `transform(df)` | `fit(df, ...)`, `apply_woe(df)` |
| Persist | `save_mapping_table`, `load_mapping_table` | `get_final_bins`, `load_woe_bins` |
| Typical use | Quick exploration and general encoding | Scorecards that need monotone, explainable bins |

[`as_woe_engine`](#3-unified-binning-engine-as_woe_engine) gives both engines one interface.

## Example data

Every snippet on this page runs top to bottom in one Python session. The setup creates a synthetic sample with the data
quirks that WOE binning has to handle: `-1` means "no record" in every numeric feature, and about 5% of the incomes are
missing.

```python
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 8000

df = pd.DataFrame({
    "age":         rng.normal(35, 8, n).clip(18, 70),
    "income":      rng.lognormal(10, 0.4, n),
    "score_b":     rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue":   rng.poisson(0.3, n),
    "city_grade":  rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
    "apply_month": rng.choice(["2025-01", "2025-02", "2025-03"], n),
})
grade_effect = df["city_grade"].map({"A": -0.4, "B": 0.0, "C": 0.3, "D": 0.7})
logit = -2.2 - 0.02 * (df["score_b"] - 600) + 0.5 * df["n_overdue"] - 0.8 * df["utilization"] + grade_effect
df["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "utilization", "n_overdue"]

# Typical data quirks: -1 means "no record" in every numeric feature, and some incomes are missing
for col in features:
    df.loc[rng.random(n) < 0.02, col] = -1
df.loc[rng.random(n) < 0.05, "income"] = np.nan

train_df = df.iloc[:5000].copy()
test_df = df.iloc[5000:6500].copy()
oot_df = df.iloc[6500:].copy()
print(f"train={len(train_df)}  bad rate={train_df['bad_flag'].mean():.3f}  missing income={train_df['income'].isna().mean():.3f}")
```

## 1. Master Class: `WOE_Master`

`WOE_Master` bins numeric variables into equal-frequency (or equal-width) bins and keeps one mapping table per variable.

```python
from Modeling_Tool import WOE_Master

woe = WOE_Master(
    train_data=train_df,
    varlist=features,
    dep="bad_flag",
    graph_save_dir="output/woe_plots",      # used only by plot_bivar_graph
)
woe.fit(nbins=10, equal_freq=True, spec_values=[-1, -999999])

train_woe = woe.transform(train_df)         # adds one `<feature>_woe` column per variable
test_woe = woe.transform(test_df)
oot_woe = woe.transform(oot_df)

woe_features = [f"{f}_woe" for f in features]
print(train_woe[woe_features].head())
print(woe.get_mapping_table().query("VAR == 'income'")[["BIN_NUM", "BIN_RANGE", "N", "AVG_BAD", "WOE", "IV"]])
```

`WOE_Master` bins **numeric** columns only (booleans work). A string column raises `TypeError: Expected numeric dtype`. Use
`MonotoneWOEBinner` with `cate_feats` (section 2) or encode the categories as numbers first.

### Parameters

`WOE_Master(train_data, varlist, dep=None, graph_save_dir="", woe_suffix="_woe", missing_ref_value=SMF_MISSING_BIN, remove_exist_dir=False)`

| Parameter | Default | Meaning |
|---|---|---|
| `train_data` | required | Training frame. `fit` bins its columns, and `transform()` scores it when you pass no data |
| `varlist` | required | Numeric columns to encode |
| `dep` | `None` | Binary target column (1 = bad). `fit` needs it |
| `graph_save_dir` | `""` | Base folder of `plot_bivar_graph` |
| `woe_suffix` | `"_woe"` | Suffix of the output columns |
| `missing_ref_value` | `SMF_MISSING_BIN` | Value that replaces missing data when `fit` and `transform` bin it. Keep the default (see [Missing values and special values](#missing-values-and-special-values)). A small finite value triggers a `UserWarning`, and a value that occurs in the data raises `ValueError` |
| `remove_exist_dir` | `False` | `True` **deletes** `graph_save_dir` recursively when the object is created |

`fit(nbins=10, equal_freq=True, tree_binning_seed=None, chi2_config=None, precision=5, min_bin_prop=0.05, include_missing=True, fillna=None, spec_values=[], sv_min_bin_size=0.0, sv_small_policy="keep", sv_woe_smoothing="none", sv_smoothing_alpha=0.0)`

| Parameter | Default | Meaning |
|---|---|---|
| `nbins` | `10` | Maximum number of bins, capped at `max(5, 1 / min_bin_prop)`. In 0.8.2 the cap never exceeds 20, so a larger value has no effect even if you lower `min_bin_prop` |
| `equal_freq` | `True` | `True` for equal-frequency bins, `False` for equal-width bins |
| `spec_values` | `[]` | Values that get a bin of their own. See the next section |
| `include_missing` | `True` | `True` gives missing values a bin of their own on every binning path. `False` drops missing rows from the fit, and scoring then sends missing values to the lowest bin |
| `min_bin_prop` | `0.05` | Lowers the cap on the bin count (see `nbins`). It does not enforce a minimum size per bin: equal-width bins can be much smaller |
| `precision` | `5` | Decimals of the bin edges |
| `tree_binning_seed` | `None` | Any non-zero value switches to decision-tree bins and is used as the random seed. `None` or `0` keeps quantile bins |
| `chi2_config` | `None` | `(initial_bins, confidence)`, for example `(100, 0.95)`, switches to chi-square merging. A higher confidence gives fewer bins. See the warning below |
| `fillna` | `None` | Has no effect on the default binning path: missing values are filled with `-999999` while binning. Leave it unset |
| `sv_min_bin_size`, `sv_small_policy`, `sv_woe_smoothing`, `sv_smoothing_alpha` | `0.0`, `"keep"`, `"none"`, `0.0` | Governance of special-value bins. See [Low-Share Special-Value Governance](#low-share-special-value-governance-sv-bin-governance-080) |

!!! note "`chi2_config` with the default `missing_ref_value`"

    Up to 0.9.0, `fit(chi2_config=...)` raised `ValueError: Bin edges must be unique` with the default `missing_ref_value`
    and `include_missing=True`; it works now. Chi-square bins are labeled `[a, b)`, while all other bins are labeled
    `(a, b]`.

### Missing values and special values

By default (`include_missing=True`), `fit` and `transform` both replace missing values with `missing_ref_value` (or with
`fit(fillna=...)` while fitting) before they bin, and the missing values get a bin of their own, `(-inf, missing_ref_value]`,
on every binning path (equal-frequency, equal-width, decision tree and chi-square); the other bins are fitted on the real
values. A sentinel that is stored as a number, such as `-1`, is a real value to the binner and shares the bin of the
smallest values. To score it on its own, list it in `spec_values`: each listed value becomes a bin edge, so it gets a bin
of its own.

| `spec_values` | Result |
|---|---|
| `[]` (default) | Missing values get their own bin; `-1` falls in the lowest bin of the real values |
| `[-1]` | Missing values and `-1` get one bin each |

!!! note "Up to 0.9.0"

    The equal-frequency bins counted the filled missing values in their quantiles, so the missing rows shared the lowest
    bin with real values (unless they made up more than one bin's share), and with the default `missing_ref_value`, which
    the edge rounding turned into `-inf`, they shared it on every binning path. Refit to get the missing bin; an engine
    fitted earlier keeps scoring with its stored bins.

- The missing bin has `NaN` in `MIN` and `MAX`. Bins that stay empty are dropped, so `BIN_NUM` can skip numbers, and a
  variable without missing values has no missing bin.
- Keep `missing_ref_value` at its default unless you need another one. Before this was fixed, the fit always filled with
  `-999999` whatever `missing_ref_value` or `fit(fillna=...)` said, so a custom value such as `-99999` sent the missing
  rows to an unrelated bin when you scored. A value inside the range of the data (for example `0`) treats missing values
  as that value in the fit and in the transform.
- A value outside the fitted range takes the WOE of the nearest bin. A feature with few distinct values is binned at its
  own values, so its last bin ends at the largest training value; a larger value in new data used to get a `NaN` WOE (an
  LR model then failed, a tree model read it as missing). The log reports how many records were moved to the nearest bin.
- If `-999999` is in `spec_values` but a variable has no missing values in the fit sample, then missing values that show up
  later get a `NaN` WOE (the log says `Failed to Map WOE values for N Records`). Fit on data that contains missing values,
  or impute before scoring.
- A bin without bads (or without goods) gets a finite WOE in `WOE_Master`: `eps` (`1e-06`) is added to its two shares,
  `ln((bad_pct + eps) / (good_pct + eps))`, as `MonotoneWOEBinner` does. The WOE of every other bin is not smoothed. Such
  bins are common with a strong predictor or a rare target, also with equal-frequency bins. Up to 0.8.2
  `WOE_Master` returned `-inf` (`+inf`) for them, which made LR and XGBoost training fail. The helpers `calc_woe` and
  `calc_iv` and the Gains tables still return `-inf` (`+inf`) for a zero share.

### Methods and attributes

| Member | Description |
|---|---|
| `transform(data=None, varlist=None)` | Copy of `data` with one `<var><woe_suffix>` column per variable (defaults: `train_data`, all fitted variables). Raises `KeyError` for a variable without a mapping |
| `get_mapping_table()` | One DataFrame for all variables. Columns: `BIN_NUM`, `BIN_RANGE`, `MIN`, `MAX`, `N`, `AVG_SCORE`, `N_BAD`, `N_GOOD`, `AVG_BAD`, `AVG_GOOD`, `BAD_PCT_PER_BIN`, `GOOD_PCT_PER_BIN`, `LIFT`, `WOE`, `IV`, `VAR`. Raises `ValueError` before `fit` |
| `woe_dict` | `{variable: DataFrame}`, the tables behind the mapping table |
| `update_woe(varlist, **fit_kwargs)` | Re-bins only these variables with the `fit` arguments you pass, for example `woe.update_woe(["age"], nbins=5)`. Other variables keep their bins |
| `save_mapping_table(path)` | Writes the mapping table to CSV. The folder must exist |
| `load_mapping_table(csv_path_or_df)` | Restores `woe_dict` and `varlist` from a CSV path or a DataFrame |
| `plot_bivar_graph(data, group=None, dirname=None, varlist=None)` | See [Plots](#plots) |

`LIFT` is `AVG_BAD` divided by the unweighted mean of `AVG_BAD` over the bins, not by the overall bad rate.

### Applying a saved mapping table: `mapping_woe`

`transform` calls `mapping_woe` for you. Call it directly when you only have the table. It is not a top-level name, so
import it from `Modeling_Tool.WOE`.

```python
from Modeling_Tool import SMF_MISSING_BIN
from Modeling_Tool.WOE import mapping_woe

scored = mapping_woe(
    data=oot_df,
    varlist=features,
    woe_mapping_table=woe.get_mapping_table(),
    missing_ref_value=SMF_MISSING_BIN,      # the value `woe` used; the function's own default is -999999
)
print(scored[woe_features].equals(oot_woe[woe_features]))
```

The signature is `mapping_woe(data, varlist, woe_mapping_table, suffix="_woe", drop_bin_info=True, missing_ref_value=-999999)`.
If the table came from a `WOE_Master` with a custom `missing_ref_value`, pass the same value. With `drop_bin_info=False` the
result also holds `_BIN_NUM_` and `_BIN_RANGE_`, for the **last** variable of `varlist` only.

### Persisting the mapping table

```python
os.makedirs("output", exist_ok=True)
woe.save_mapping_table("output/woe_mapping.csv")

# In a scoring job no training data is needed
scorer = WOE_Master(train_data=pd.DataFrame(), varlist=[], dep="bad_flag")
scorer.load_mapping_table("output/woe_mapping.csv")
print(np.allclose(scorer.transform(oot_df)[woe_features], oot_woe[woe_features], equal_nan=True))

# Or read the table without an object
from Modeling_Tool import load_mapping_table

varlist, woe_dict = load_mapping_table("output/woe_mapping.csv")      # (list of variables, {variable: DataFrame})
```

### WOE and IV from percentages: `WOEIVCalculator`

`WOEIVCalculator(data, bad_pct_col, good_pct_col)` turns two share columns into WOE and IV. `calc_both()` computes the
logarithm once and returns `(woe, iv)` as two Series.

```python
from Modeling_Tool import WOEIVCalculator

bin_summary = pd.DataFrame({
    "BAD_PCT_PER_BIN":  [0.20, 0.30, 0.50],     # share of all bads in each bin
    "GOOD_PCT_PER_BIN": [0.40, 0.30, 0.30],     # share of all goods in each bin
})
woe_values, iv_values = WOEIVCalculator(
    bin_summary, bad_pct_col="BAD_PCT_PER_BIN", good_pct_col="GOOD_PCT_PER_BIN",
).calc_both()
print(woe_values.round(4).tolist(), iv_values.round(4).tolist())
```

### Plots

```python
from Modeling_Tool import plot_woe

mapping = woe.get_mapping_table()
os.makedirs("output/woe_plots", exist_ok=True)

# One variable: pass the rows of that variable only
plot_woe(mapping[mapping["VAR"] == "score_b"], to_show=False, save_dir="output/woe_plots", fig_name="score_b.png")

# Overall and by-group charts for several variables
woe.plot_bivar_graph(train_df, group="apply_month", dirname="by_month", varlist=["score_b", "age"])
print(sorted(os.listdir("output/woe_plots/by_month")))
```

`plot_woe(woe_df, var_rename=None, to_show=True, save_dir=None, fig_name="var.png")` needs an existing `save_dir`. It treats
every row of `woe_df` as one variable, so a table with several variables gives a mixed chart titled with the first variable.
`plot_bivar_graph` creates `<graph_save_dir>/<dirname>/` itself and writes `<var>.png`, plus `<var>_<group>.png` when you pass
`group`. `dirname` is required, and rows with a missing value in a variable are skipped in the by-group chart.

### Performance on wide tables

`fit` and `transform` loop over variables, because every variable has its own bins, but each step works on whole columns:
`pandas.cut` assigns the bins and a vectorized lookup (`numpy.take`) maps them to WOE. `transform` collects all WOE columns
and concatenates them once, which avoids fragmenting a wide frame.

### Other helpers

| Function | Import | Purpose |
|---|---|---|
| `get_overall_woe_table(woe, data, varlist=None)` | `from Modeling_Tool import get_overall_woe_table` | Recompute the bin table (`N`, `AVG_BAD`, `WOE`, `IV`, `LIFT`, ...) of any sample on the fitted bins. See [section 5](#5-monotonicity-check) |
| `get_group_woe_table(woe, data, group, varlist=None)` | `from Modeling_Tool.WOE import get_group_woe_table` | The same per group. Returns `{"summary", "pivot", "detail"}`. In `summary`, `KS_PER_BIN` repeats `TOP_LIFT` and is not a KS statistic |
| `get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt=None, var_dict=None)` | `from Modeling_Tool.WOE import get_woe_plot_report_new` | Place the images of `plot_bivar_graph` (`<var>.png`, `<var>_<grp_name>.png`) on an `ExcelMaster` worksheet |
| `calc_woe(data, bad_pct, good_pct)`, `calc_iv(...)` | `from Modeling_Tool.Core import calc_woe, calc_iv` | WOE or IV of a table of shares |

## 2. Greedy Monotone Binner: `MonotoneWOEBinner`

`MonotoneWOEBinner` starts from equal-frequency bins and merges neighbors until the WOE is monotone across the bins. Special
values (such as `-1`) and missing values get bins of their own, outside the monotone constraint. A categorical feature gets one
bin per category.

```python
from Modeling_Tool import MonotoneWOEBinner

binner = MonotoneWOEBinner(
    feature_cols=features,
    target_col="bad_flag",
    n_init_bins=20,
    min_bin_size=0.03,
    special_values=[-1, np.nan],        # np.nan gives missing values their own [Missing] bin
    cate_feats=["city_grade"],
)
binner.fit(train_df, chi2_binning=True, chi2_p=0.95)
binner.refine_cate(max_bins=3)

mono_train = binner.apply_woe(train_df)     # adds `<feature>_woe`, also for city_grade
bins = binner.get_final_bins()              # {feature: DataFrame}
edges = binner.get_bin_edges()              # {feature: [-inf, cut1, ..., inf]}

print(binner.iv_summary)
print(bins["score_b"][["bin_label", "n", "bad_rate", "woe", "iv", "is_special"]])
print(binner.get_direction_summary())
```

### Parameters

`MonotoneWOEBinner(feature_cols, target_col, n_init_bins=20, min_bin_size=0.03, min_n_bins=2, eps=1e-6, missing_woe=0.0, special_values=None, cate_feats=None, bin_label_decimals=None, ...)`.
The governance parameters (`min_bad_count` and the rest) are listed in [Binning Governance](#binning-governance-067) and
[Low-Share Special-Value Governance](#low-share-special-value-governance-sv-bin-governance-080).

| Parameter | Default | Meaning |
|---|---|---|
| `feature_cols` | required | Numeric features to bin |
| `target_col` | required | Binary target column (1 = bad) |
| `n_init_bins` | `20` | Number of equal-frequency bins to start from |
| `min_bin_size` | `0.03` | Minimum share of the rows per ordinary bin. It takes effect **only when `small_bin_policy` is set**. Without a policy, `fit` ignores it |
| `min_n_bins` | `2` | Merging stops at this many ordinary bins (special-value bins not counted); neither the greedy monotone step nor the chi-square merging goes below it (the greedy step used to end one bin below it). When the limit stops the merging, the WOE of the bins may not be monotone |
| `eps` | `1e-6` | Added to the shares so that `log` never sees zero |
| `missing_woe` | `0.0` | WOE for missing values that are **not** in `special_values`, for unseen categories, and for the placeholder bins of `unseen_special_policy="neutral"` |
| `special_values` | `None` | Values that get their own `[sv=<value>]` bin. `np.nan` adds a `[Missing]` bin. Applies to numeric features only |
| `sv_total_basis` | `"all"` | The bad and good totals that every bin's WOE is measured against: `"all"` (one base) or `"ordinary"` (legacy, the behavior of 0.8.2 and earlier). See [Which totals the WOE is measured against](#which-totals-the-woe-is-measured-against-sv_total_basis) |
| `cate_feats` | `None` | Categorical (discrete) features. Each value is a bin, with a `[Missing]` bin if the fit data has missing values. They are not cut into intervals |
| `bin_label_decimals` | `None` | Decimals of the numbers in bin labels. `None` keeps 8 significant digits |

### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(df, chi2_binning=False, chi2_p=0.99, chi2_init_size=1000, n_jobs=1)` | `self` | Greedy monotone binning. With `chi2_binning=True` it then merges neighbors whose chi-square p-value is at least `1 - chi2_p`, so **a higher `chi2_p` gives fewer bins**. `chi2_init_size` caps the stratified sample used for the test. `n_jobs > 1` uses that many processes, and `-1` uses all cores |
| `refine_chi2(df, features=None, chi2_p=0.99, chi2_init_size=1000, n_jobs=1)` | `self` | The chi-square step on its own, so you can retry other `chi2_p` values without refitting |
| `refine_dtree(df, features=None, max_bins=6, min_samples_leaf=0.05, monotone=True, n_jobs=1, max_depth=None)` | `self` | Re-cuts numeric features with a decision tree, then merges until monotone (if `monotone=True`) |
| `refine_cate(features=None, max_bins=5, min_bin_size=0.0, badrate_tol=None)` | `self` | Merges categories with similar bad rates until at most `max_bins` remain. `min_bin_size` forces small bins to merge, and `badrate_tol` stops merging when neighbors differ too much. It needs no data, because it uses the fitted counts. `[Missing]` is never merged |
| `apply_woe(data, suffix="_woe", inplace=False, unseen_category_policy="warn", varlist=None)` | DataFrame | Adds `<feature><suffix>` columns. `varlist` limits the features, which suits wide tables. `inplace=True` writes into `data` |
| `get_final_bins()` | `{feature: DataFrame}` | Final bin tables, with columns `bin_no`, `bin_label`, `n`, `bad`, `good`, `bad_rate`, `pct_n`, `lift`, `pct_bad`, `pct_good`, `woe`, `iv`, `cumiv`, `is_special`. Special-value bins follow the ordinary bins |
| `get_bin_edges()` | `{feature: [-inf, ..., inf]}` | Cut points of numeric features. Special-value bins and categorical features are not included |
| `get_direction_summary()` | DataFrame | Columns `feat`, `direction` (`increasing`, `decreasing`, `flat`, or `categorical`), `direction_basis`, `is_monotonic` |
| `iv_summary` (property) | DataFrame | Columns `feature`, `iv`, `n_bins`, `n_sv_bins`, `is_monotonic`, `is_categorical`, sorted by IV |
| `load_woe_bins(bins_dict)` | `self` | Loads bins instead of fitting. Accepts the frames of `get_final_bins()` ("Format A"), or dicts with `edges`, `woe_map`, `missing_woe`, and optionally `bin_df` (the pipeline's "Format B") |
| `export_woe_report(report_path)` | `None` | Excel report. See [Plots and Excel report](#plots-and-excel-report) |
| `plot_woe_graph(graph_path, group_name=None, _df_for_group=None, dpi=150, figsize=(9, 6), bar_mode="clustered")` | `None` | One PNG per feature |

`fit` and the `refine_*` methods return `self`, so they chain: `binner.fit(train_df).refine_dtree(train_df).refine_chi2(train_df, chi2_p=0.95)`.

### Missing values, special values, and categories

- A numeric value listed in `special_values` gets a bin labeled `[sv=<value>]`, and `np.nan` gets `[Missing]`. Each bin has its
  own WOE.
- A missing value that is not listed scores `missing_woe`.
- By default every bin, ordinary, special-value or `[Missing]`, is measured against the bad and good totals of **all** fit
  rows, so all bins are on one base. `sv_total_basis="ordinary"` restores the legacy two bases (see
  [Which totals the WOE is measured against](#which-totals-the-woe-is-measured-against-sv_total_basis)).
- `refine_cate` labels a merged bin with its member names joined by a vertical bar, such as `B | C`.
- Categories that were not seen at fit time score `missing_woe`. `apply_woe` emits a `RuntimeWarning` by default
  (`unseen_category_policy="warn"`). Use `"raise"` to fail instead, or `"silent"` to stay quiet.
  `binner._unseen_category_stats` records the unseen values of the latest call.
- A category is matched exactly first, and then by its `str()` form, which tolerates a dtype that drifted between fit and
  scoring. Rows that needed the fallback are counted in `binner._categorical_transform_stats[feature]["fallback_match_rows"]`
  and raise a warning.
- A feature that fails to fit is **skipped**. The traceback is printed, the other features are fitted, and the feature is
  missing from `get_final_bins()`. A string column that is not in `cate_feats` fails this way. Check `binner.iv_summary`
  after `fit`.

!!! note "Fit log"

    `fit` and the `refine_*` methods log one line per feature (`n_bins`, `IV`, whether the WOE is monotone) at INFO level.
    The first line reads `[MonotoneWOEBinner] Fitting N features ...` and the last one `[MonotoneWOEBinner] Fit finished
    (greedy): k/N features monotone`. Importing SMF calls `logging.basicConfig(level=logging.INFO)`, unless logging is
    already configured, so these lines show up. To silence the INFO lines, run
    `logging.getLogger().setLevel(logging.WARNING)` after importing SMF.

### Saving and reloading bins (Format A)

Every frame from `get_final_bins()` carries `DataFrame.attrs["smf_woe_format_a"]`. It holds the exact cut points, the
original (sparse) bin ids, the category members, and `missing_woe`, plus the special-value decisions when governance is on.
`load_woe_bins` uses this metadata only if its checksum and a row-identity check pass. Otherwise it falls back to parsing the
visible `bin_label` text, so edited or stale metadata is harmless.

```python
import pickle

with open("output/woe_bins.pkl", "wb") as f:
    pickle.dump(bins, f)                   # pickle keeps `attrs`: exact round trip
with open("output/woe_bins.pkl", "rb") as f:
    restored = MonotoneWOEBinner(feature_cols=[], target_col="bad_flag").load_woe_bins(pickle.load(f))

print(binner.apply_woe(test_df).equals(restored.apply_woe(test_df)))

# CSV and Excel drop `attrs`: only the visible labels remain
bins["score_b"].to_csv("output/score_b_bins.csv", index=False)
from_csv = MonotoneWOEBinner(feature_cols=[], target_col="bad_flag").load_woe_bins(
    {"score_b": pd.read_csv("output/score_b_bins.csv")}
)
orig_edges = np.array(binner.get_bin_edges()["score_b"][1:-1])
csv_edges = np.array(from_csv.get_bin_edges()["score_b"][1:-1])
print(np.abs(orig_edges - csv_edges).max())       # small but not zero: labels keep 8 significant digits
```

!!! warning "CSV and Excel are not an exact round trip"

    Without `attrs`, the label text is the only source of truth. Cut points beyond the printed digits, the original sparse bin
    ids, ambiguous category members, a non-default `missing_woe`, and the special-value decisions cannot be restored. To keep
    them, pass the frames directly or use a format that preserves `attrs`, such as pickle.

### Plots and Excel report

```python
binner.plot_woe_graph("output/woe_charts")                              # <feature>.png
binner.plot_woe_graph(
    "output/woe_charts_by_month", group_name="apply_month", _df_for_group=train_df,    # <feature>_by_apply_month.png
)
print(sorted(os.listdir("output/woe_charts"))[:2], sorted(os.listdir("output/woe_charts_by_month"))[:2])

binner.export_woe_report("output/woe_report.xlsx")
```

- The overall chart shows the bins as stacked good and bad shares, the WOE line, and a label with WOE, bad rate, and lift.
  Special-value bins are drawn to the right with a dashed outline.
- With `group_name`, `_df_for_group` is **required** (despite its leading underscore). It must hold the features, the target,
  and the group column. Without it, the call creates the folder, skips every feature, and only logs a message. Each group
  gets its own WOE line.
- `bar_mode` selects the bars of a by-group chart: `"clustered"` (default) draws side-by-side bars per group, each as a share
  of its group, `"pooled"` draws one set of bars for all rows, and `"small_multiples"` draws one panel per group. All modes
  write the same file name, so use one folder per mode. With `"clustered"` and a group, the chart is always `figsize=(16, 6)`
  at `dpi=200`. The other modes use the `figsize` and `dpi` you pass, and `"small_multiples"` treats `figsize` as the size of
  one panel (scaled by 0.62) in a grid of up to 3 columns.
- `export_woe_report` writes a workbook with two sheets. The first has a summary table and one bin table per feature, with
  special-value rows in purple. The second embeds the overall chart of every feature. It needs `xlsxwriter` and `Pillow`,
  which SMF installs. The sheets are named `WOE Bin Details` and `WOE Bin Charts`.

#### How the by-group IV and WOE lines are computed

- The IV in the legend (`pooled` and `clustered`) and in the panel titles (`small_multiples`) is the **within-group IV**: the
  same calculation as the overall IV, on the rows of that group.
    - Ordinary bins use the group's bads and goods that fall in ordinary bins as denominators.
    - Special-value and missing bins use all bads and goods of the group, and they reuse the decision made at fit time for
      each bin (`keep` uses the empirical value, smoothed with the fit-time parameters if smoothing was on, `neutral` counts as
      0, and `merged_into_missing` joins the group's `[Missing]` bin). The share is not judged again inside the group.
    - Bins that have only goods or only bads in the group, and were not smoothed, are left out of the group IV, so that a few
      one-class bins cannot inflate it.
- Treating the whole fit sample as one group gives the overall IV. Groups with fewer than 5 rows or with one class show
  `IV=0.000`. For categorical features, values unseen at fit time are left out of the denominators of ordinary bins.
- The group **WOE lines** use the bads and goods of all rows you pass as denominators. For bins with both classes, a group's
  line equals its within-group WOE plus the constant `ln(group's share of all bads / group's share of all goods)`. The shape
  matches the within-group WOE, and the vertical position shows how risky the group is compared with the whole.
- Bins loaded with `load_woe_bins` reuse the fit-time decisions when the metadata is intact. A reload from CSV, Excel, or
  Format B carries no decisions, and special-value bins are always counted at their empirical value.
- Before 0.8.1 the group IV used the whole sample as the denominator, so k groups of similar size each showed about 1/k of the
  within-group IV. Do not compare charts from those versions with the overall IV.

## 3. Unified Binning Engine: `as_woe_engine`

`WOE_Master` and `MonotoneWOEBinner` store their bins in different formats. `as_woe_engine()` wraps either one in an adapter
with one interface, which PSI, IV, and correlation screening reuse.

```python
from Modeling_Tool import as_woe_engine

engine = as_woe_engine(binner)                  # a fitted WOE_Master works too
woe_table = engine.get_woe_table(features)
engine_train = engine.transform(train_df, features)
print(engine.get_engine_name(), woe_table.shape, engine_train.shape)
```

| Adapter method | Returns |
|---|---|
| `transform(data, varlist=None, suffix="_woe")` | `data` with the WOE columns |
| `get_woe_table(varlist=None)` | One table with columns `VAR`, `BIN_NUM`, `BIN_RANGE`, `MIN`, `MAX`, `N`, `N_BAD`, `N_GOOD`, `AVG_BAD`, `WOE`, `IV`, `IS_SPECIAL`, `ENGINE` |
| `assign_bins(data, var)`, `assign_bins_frame(data, varlist, feature_block_size=64)` | Bin labels for PSI-style comparisons: the fitted WOE as text, or `__MISSING__`. Bins with equal WOE share a label |
| `get_bin_edges(varlist=None)` | `{feature: [-inf, ..., inf]}` for the monotone engine, and `{}` for `WOE_Master` |
| `get_engine_name()` | `"master"` or `"monotone"` |

`as_woe_engine(None)` returns `None`, an existing adapter is returned unchanged, and any other object raises `TypeError`. See
[WOE Binning Engine](woe_binning_engine.md) for more.

## 4. Linking with Feature Screening

Fit the binner once at training time, then reuse the same object for screening, monitoring, and modeling:

```python
from Modeling_Tool import PSICalculator, VarExtractionInsights, CorrelationFilter

psi_table = PSICalculator(buckets=10, binning_engine=binner).calculate(train_df, oot_df, features)

iv_report = VarExtractionInsights(
    train_df, "bad_flag", "./iv_plots/",
    woe_engine="monotone", woe_binner=binner,
).get_var_analysis_report(train_df, features, iv_cut=0.0)

keep_vars = CorrelationFilter(
    train_df, "bad_flag", corr_cutpoint=0.7,
    woe_engine="monotone", woe_binner=binner,
).remove_highly_correlated(features)

print(psi_table)
print(iv_report[["var", "iv", "n_bins"]])
print(keep_vars)
```

`get_var_analysis_report(data, varlist, dep=None, iv_cut=0.01)` drops variables whose IV is below `iv_cut`, so pass
`iv_cut=0.0` to see all of them. On a fitted engine, the report has one row per variable with the columns `var`, `n_all`,
`n`, `ks_in_gains`, `lift_in_gains`, `iv`, `n_bump`, `missing_rate`, `min`, `mean`, `max`, and `n_bins`.

## 5. Monotonicity Check

`is_monotonic(data, column, direction="auto", strict=False, handle_nan="drop")` returns `(is_monotone, direction)`, where
`direction` is `1` (increasing), `-1` (decreasing), or `0` (not monotone). Check the **ordinary bins in bin order**: the
missing and special-value bins are outside the monotone constraint.

```python
from Modeling_Tool import is_monotonic, get_overall_woe_table

special_values = [-1, -999999]
mapping = woe.get_mapping_table()
for var in features:
    table = mapping[mapping["VAR"] == var]
    ordinary = table[table["MIN"].notna() & ~table["MIN"].isin(special_values)]     # drop the missing and special bins
    mono, direction = is_monotonic(ordinary, "WOE")
    print(f"{var:12s} monotone={mono!s:5}  direction={direction}  bins={len(ordinary)}")

# The same check on another sample, using the fitted bins
oot_table = get_overall_woe_table(woe, oot_df, ["income"])
oot_table = oot_table.dropna(subset=["BIN_NUM"]).sort_values("BIN_NUM")        # missing values form a row with NaN BIN_NUM
print(is_monotonic(oot_table[~oot_table["MIN"].isin(special_values)], "WOE"))
```

`get_overall_woe_table` does not guarantee bin order, and it reports missing values as a row whose `BIN_NUM` is `NaN`, which is
why the example sorts and drops. `WOE_Master` does not force monotone bins, so a variable can report `False`. With
`MonotoneWOEBinner`, read the `is_monotonic` column of `binner.get_direction_summary()` instead.

## 6. Standalone WOE Transforms

The functional interface bins one variable at a time and returns the bins next to the data.

```python
from Modeling_Tool import woe_transform

single_df, single_map = woe_transform(train_df, var="income", dep="bad_flag", nbins=10, include_missing=True)
print(single_df.shape, single_map[["_bin_num_income", "_bin_range_income", "N", "WOE"]].head(3))
```

| Function | Returns |
|---|---|
| `woe_transform(train_df, var, dep, nbins, oot_df=None, ...)` | `(train_frame_with_<var>_woe, bin_table)`, or `(train_frame, oot_frame, bin_table)` when `oot_df` is given. The bin table keeps the columns `_bin_num_<var>` and `_bin_range_<var>`. With `ret_woe_table=False` you get the frames only |
| `woe_transformation(train_df, varlist, dep, oot_df=None, nbins=10, ...)` | `({"TRAIN": frame, "OOT": frame}, bin_table)`. The bin table has `VAR`, `BIN_NUM`, and `BIN_RANGE` columns. `"OOT"` appears only when `oot_df` is given |

Both functions default to `include_missing=False` and `fillna=-999999`, unlike `WOE_Master`. With `include_missing=False`,
`woe_transform` **drops** the rows with a missing value from the frame it returns. They accept `spec_values`,
`chi2_config`, `tree_binning_seed`, `precision`, `min_bin_prop`, `equal_freq`, `drop_bin_info`, and `ret_woe_table`, and
`woe_transform` also accepts the four `sv_*` parameters and `check_monotonicity` (it logs a warning when the WOE is not
monotone in the training sample).

!!! warning "`woe_transformation` switches to chi-square merging after the first variable"

    In 0.8.2, `woe_transformation` bins the first variable as you asked, and every later variable is silently merged with the
    chi-square method as well (`WOETransformer` stores a default `chi2_config` after its first call). The result depends on
    the order of `varlist`. For several variables, use `WOE_Master`, or call `woe_transform` once per variable.

## FAQ

??? question "When should I choose MonotoneWOEBinner?"

    When a variable enters a scorecard and needs a monotone, explainable shape, or when you have categorical features.
    `WOE_Master` is quicker for exploration and does not force monotone bins.

??? question "Why pass the binner at the screening stage?"

    PSI, IV, and KS should be computed on the same bins that finally go live. Otherwise the screening metrics and the model
    inputs can disagree.

??? question "`WOE_Master.fit` raises `TypeError: Expected numeric dtype, got object instead`"

    The variable is a string column. `WOE_Master` bins numeric columns only. Fit it with `MonotoneWOEBinner(...,
    cate_feats=[...])`, or map the categories to numbers first.

??? question "A WOE column contains NaN after `transform`"

    A value fell in a bin that is not in the mapping table. The usual cause is `-999999` in `spec_values` while the fit
    sample had no missing values for that variable (see [Missing values and special values](#missing-values-and-special-values)).
    The log line `Failed to Map WOE values for N Records` gives the count.

??? question "A bin has a WOE of `-inf`"

    The bin has no bads (or no goods). `WOE_Master` fits such a bin with a finite WOE (`eps` is added to its two shares),
    so the value comes from a table built by an earlier version, from `calc_woe` / `calc_iv`, or from a Gains table, which
    still return `-inf` (`+inf`) for a zero share. Refit with the current version, or use fewer bins, or fit with
    `MonotoneWOEBinner`.

??? question "`WOE_Master.fit(chi2_config=...)` raises `ValueError: Bin edges must be unique`"

    That happened up to 0.9.0 with the default `missing_ref_value` and `include_missing=True`. Upgrade, or pass
    `include_missing=False` or create the object with `missing_ref_value=-999999`.

??? question "`fit` printed a traceback, and a feature is missing from the results"

    `MonotoneWOEBinner` skips a feature that fails to fit and carries on with the rest. Read the printed error. The usual
    cause is a string column that is not listed in `cate_feats`.

??? question "I see a warning that special values never occur in the fit sample"

    You declared a numeric special value, such as `-1`, that no fit row has for some feature, and you set
    `unseen_special_policy="normal_bin"` (the default up to 0.8.2): that value is then scored as an ordinary number at
    scoring time. Remove the declaration for that feature, or use the default `unseen_special_policy="neutral"` (see
    [Declared but Unseen in the Fit Sample](#declared-but-unseen-in-the-fit-sample-unseen_special_policy-082)).

??? question "Why do I see more warnings since 0.8.2?"

    Importing SMF used to switch all warnings off. It no longer does. See the [FAQ](../faq.md).

## Binning Governance (0.6.7+)

`MonotoneWOEBinner` has three groups of governance parameters for bin size, direction, and missing values. They are off by
default (`None` or `"auto"`) and then behave as in earlier versions. The exception is `refine_min_n_bins_policy`, which
defaults to `"warn"` since 0.7.1.

| Parameter | Default | Values | Effect |
|---|---|---|---|
| `min_bad_count`, `min_good_count` | `None` | int | Minimum number of bads and goods per ordinary bin |
| `small_bin_policy` | `None` (off) | `"merge"`, `"warn"`, `"raise"` | A bin below `min_bad_count`, `min_good_count`, or `min_bin_size` is merged into the neighbor with the closer WOE, reported with a `UserWarning`, or raises `BinningPolicyViolation` |
| `monotone_direction` | `"auto"` | `"auto"`, `"increasing"`, `"decreasing"`, or `{feature: direction}` | Forces the WOE direction. `"increasing"` means the WOE rises with the feature value. A string applies to all numeric features |
| `reference_target` | `None` | a 0/1 column | Derives each feature's direction from this label: increasing if the feature's mean among bads is higher than among goods. Features named in a `monotone_direction` dict keep their forced direction |
| `direction_conflict_policy` | `None` (acts as `"warn"`) | `"warn"`, `"raise"`, `"keep"` | What to do when the fitted direction contradicts the expected one, or when the forced direction collapses a feature into one bin |
| `missing_bin_strategy` | `None` | `"empirical_special"`, `"fixed_woe"`, `"fail"` | How missing values are scored. See below |
| `refine_min_n_bins_policy` | `"warn"` | `"warn"`, `"enforce"`, `"raise"`, `None` | What `refine_dtree` does when it returns fewer than `min_n_bins` bins: keep the result and warn, keep the bins from before the refinement, raise, or do not check |

`missing_bin_strategy`:

- `None` picks `"empirical_special"` if `special_values` contains NaN, and `"fixed_woe"` otherwise.
- `"empirical_special"` requires NaN in `special_values`. Missing rows get a `[Missing]` bin with an empirical WOE.
- `"fixed_woe"` gives missing rows the constant `missing_woe`, and conflicts with NaN in `special_values` (`ValueError`).
- `"fail"` makes `fit` and `apply_woe` raise `ValueError` if a fitted feature has a missing value.

```python
binner = MonotoneWOEBinner(
    feature_cols=features,
    target_col="bad_flag",
    special_values=[-1, np.nan],
    min_bad_count=50, min_good_count=50,
    small_bin_policy="merge",                # merge, warn, or raise
    monotone_direction={"score_b": "decreasing", "utilization": "decreasing", "n_overdue": "increasing"},
    direction_conflict_policy="warn",        # warn, raise, or keep
)
binner.fit(train_df)
binner.refine_dtree(train_df, max_depth=3)
print(binner.get_direction_summary())       # feat, direction, direction_basis, is_monotonic
```

- `raise` throws `BinningPolicyViolation`, a `ValueError` subclass in `Modeling_Tool.WOE.WOE_Monotone_Binner`. It is the one
  error that stops `fit`. Other per-feature errors are logged and the feature is skipped.
- Categorical features also run `small_bin_policy` during the first `fit` (0.7.2). `merge` never goes below `min_n_bins`. If
  violating bins remain, `binner._small_bin_stats[feature]["remaining_violation"]` says so.
- The direction is resolved before fitting, so serial and parallel (`n_jobs`) fits agree.
- `reference_target` and a `monotone_direction` dict can be combined: the dict wins for the features it names.
- In `FeatureValidationPipeline`, `feature_screen`, and `CreditModelPipeline`, pass these parameters through
  `monotone_woe_params` (see [Pipeline-layer exposure](#pipeline-layer-exposure)).

## Which totals the WOE is measured against (`sv_total_basis`)

The WOE of a bin is `ln((bad in bin / total bad) / (good in bin / total good))`. The question is which "total" to use when the
feature has special values or missing values:

| `sv_total_basis` | Ordinary bins (and categories) | Special-value and `[Missing]` bins | Consequence |
|---|---|---|---|
| `"all"` (default) | totals of all rows | totals of all rows | One base, the textbook scorecard definition: the WOE of all bins is comparable, the shares add up to 1, and IV is a sum over one base |
| `"ordinary"` (legacy, the behavior of 0.8.2 and earlier) | totals of the ordinary rows | totals of all rows | Two bases. A special bin and an ordinary bin with the same bad rate get different WOE, the shares of the bins add up to more than 1, and the IV mixes the two bases |

The bin edges and the monotone merging are the same in both modes (they run on the ordinary rows). Only the WOE of the ordinary
bins moves, and by one constant, so their order and their gaps do not change. When the feature has no special or missing
values, both modes give identical tables.

In a test, 40% of the rows carry the code `-1` with a 28.6% bad rate. With `"ordinary"` that special bin gets a WOE of 0.645, below
ordinary bins whose bad rate is only 22.5% (WOE 0.96), and the feature's IV is 1.20. With `"all"` the order follows the bad
rate and the IV is 0.89. The IV of `"ordinary"` is overstated whenever special rows exist, so a screening threshold on IV
treats such features more leniently than it should.

```python
binner = MonotoneWOEBinner(
    feature_cols=["income"], target_col="bad_flag",
    special_values=[-1, np.nan],
    sv_total_basis="ordinary",  # only to reproduce a scorecard fitted before the default changed
)
```

`CreditModelPipeline`, `FeatureValidationPipeline` and `feature_screen` accept the key in `monotone_woe_params`. The setting is
applied at `fit` and again after `refine_chi2`, `refine_dtree` and `refine_cate`, and the by-group charts of `plot_woe_graph`
measure their in-group IV and per-group WOE lines on the same basis; bins loaded with `load_woe_bins` keep the WOE they
were saved with, and a binner pickled before the setting existed keeps `"ordinary"` when it is refitted. `"all"` is the default
because an LR model then sees special rows at the right place relative to the ordinary bins. Saved scorecards and artifacts keep
the WOE they were built with; refitting one with the new default moves the WOE of its ordinary bins (and its IV) when the
feature has special or missing values, so pass `sv_total_basis="ordinary"` to reproduce it exactly.

## Low-Share Special-Value Governance (SV Bin Governance, 0.8.0)

`small_bin_policy` and `min_bin_size` govern only **ordinary** interval bins. A special-value (SV) bin takes its empirical WOE
unconditionally and counts toward the total IV. When a sentinel is rare (for example `-1` in 0.05% of the rows),
`ln(bad share / good share)` is estimated from very few rows, so the WOE is noisy, the IV is overstated, and the PSI drifts
after go-live. Two independent switches (0.8.0) address this. They have the same names, meaning, and basis on both engines:
they are `MonotoneWOEBinner` constructor parameters and `WOE_Master.fit()` / `update_woe()` parameters.

| Parameter | Default | Values | Meaning |
|---|---|---|---|
| `sv_min_bin_size` | `0.0` | `[0.0, 1.0)` | Threshold on the SV bin's share of the **whole sample**. `0.0` turns the fallback off |
| `sv_small_policy` | `"keep"` | `"keep"`, `"neutral"`, `"merge_missing"` | How an SV bin below the threshold is handled |
| `sv_woe_smoothing` | `"none"` | `"none"`, `"laplace"` | Shrink the WOE of SV bins toward the overall bad rate |
| `sv_smoothing_alpha` | `0.0` | `>= 0.0` | Shrinkage strength. `0.0` turns it off |

The defaults give exactly the results of 0.7.2. An illegal value raises `ValueError` at `__init__` or `fit()`.

### Mode 1: low-share fallback

The share is `n_bin / N_total`, and the test is **strictly less than**: `share < sv_min_bin_size`. A bin exactly at the
threshold is kept.

- `keep` (default): empirical WOE, no change.
- `neutral`: the bin gets `woe = 0.0` and `iv = 0.0`, as if the value gave no evidence.
- `merge_missing`: the bad and good counts of the bin move into the `[Missing]` bin, whose WOE is recomputed from the
  empirical formula. The merged row keeps its label, but its `n` becomes 0, its stored WOE becomes the new `[Missing]` WOE,
  and its `iv` is `0`, so the total IV does not count it twice. A feature without a `[Missing]` bin falls back to `neutral`
  with a `UserWarning`.

`apply_woe` and `mapping_woe` need no change for `merge_missing`: the stored WOE of the merged row is already the `[Missing]`
WOE, so the fit and scoring paths stay consistent.

### Mode 2: WOE smoothing (`sv_woe_smoothing="laplace"`)

Smoothing touches **only SV bins**. It shrinks the bad rate inside the bin toward the overall bad rate, then converts back to
shares:

```
p = N_bad / (N_bad + N_good)                     # overall bad rate
n_bin = n_bad + n_good                           # rows in the bin
r = (n_bad + alpha * p) / (n_bin + alpha)        # bin bad rate shrunk toward p

pct_bad_s  = n_bin * r       / N_bad             # back to shares of ALL bads and goods
pct_good_s = n_bin * (1 - r) / N_good

woe = ln((pct_bad_s + eps) / (pct_good_s + eps))
iv  = (pct_bad_s - pct_good_s) * woe
```

- `alpha = 0` gives the empirical WOE exactly.
- `alpha -> infinity` gives `r -> p` and `woe -> 0`, monotonically.
- `alpha` competes with `n_bin`, so **the fewer rows a bin has, the stronger the shrinkage**.

Shrinking the bad rate, rather than adding pseudo-counts to the population shares, is what guarantees that the WOE tends to
0 as `alpha` grows. Both engines use this formula.

### Combining the switches

Each SV bin is decided in a fixed order:

```
share = n_sv / N_total
if sv_small_policy != "keep" and sv_min_bin_size > 0 and share < sv_min_bin_size:
    apply the Mode 1 fallback (neutral or merge_missing); no smoothing
elif sv_woe_smoothing == "laplace" and alpha > 0:
    smooth the WOE (Mode 2)
else:
    empirical WOE
```

Mode 1 wins: a bin below the threshold is never smoothed. Under `merge_missing` with `laplace`, the `[Missing]` bin that
receives the merge is recomputed from the **empirical** formula, because the merged bin should show the true bad rate. Other
SV bins that meet the threshold are still smoothed.

The `[Missing]` bin is a normal governed SV bin in both engines: smoothed if it meets the threshold, zeroed by `neutral` if it
does not. Its only special trait is that under `merge_missing` it can be a merge target but never a merge source. This is
independent of `missing_bin_strategy`, which governs only how missing values are scored, whereas the four `sv_*` parameters
govern **all** SV bins, including sentinels such as `-1`.

### Usage

```python
binner = MonotoneWOEBinner(
    feature_cols=features,
    target_col="bad_flag",
    special_values=[-1, np.nan],
    sv_min_bin_size=0.03,            # SV bins below 3% of the sample ...
    sv_small_policy="neutral",       # ... get WOE 0
    sv_woe_smoothing="laplace",      # the other SV bins are shrunk toward the overall bad rate
    sv_smoothing_alpha=50.0,
)
binner.fit(train_df)

# The decision for each special-value bin is stored in the metadata, one entry per special bin
bins = binner.get_final_bins()["income"]
policies = bins.attrs["smf_woe_format_a"]["sv_decisions"]["policies"]
special = bins[bins["is_special"]].assign(sv_policy_applied=policies)
print(special[["bin_label", "n", "woe", "iv", "sv_policy_applied"]])
```

The decision of each SV bin is one of `keep`, `neutral`, `neutral(fallback)`, `merged_into_missing`, `merge_target`, or (0.8.2)
`unseen_at_fit`. `get_final_bins()` has no `sv_policy_applied` column: read the decisions from `attrs` as above. They are
present only when governance is on or the table has `unseen_at_fit` placeholders.

For `merge_missing`, the sentinel `-1` (1.8% of the rows) is merged into the `[Missing]` bin of `income`:

```python
merged = MonotoneWOEBinner(
    feature_cols=["income"], target_col="bad_flag", special_values=[-1, np.nan],
    sv_min_bin_size=0.03, sv_small_policy="merge_missing",
).fit(train_df)

bins = merged.get_final_bins()["income"]
policies = bins.attrs["smf_woe_format_a"]["sv_decisions"]["policies"]
print(bins[bins["is_special"]].assign(sv_policy_applied=policies)[["bin_label", "n", "woe", "iv", "sv_policy_applied"]])
```

`WOE_Master` takes the same parameters in `fit()`:

```python
woe_sv = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe_sv.fit(
    nbins=10, equal_freq=True, spec_values=[-1, -999999],
    sv_min_bin_size=0.03, sv_small_policy="neutral",
    sv_woe_smoothing="laplace", sv_smoothing_alpha=50.0,
)
mapping_sv = woe_sv.get_mapping_table()
print(mapping_sv[(mapping_sv["VAR"] == "income") & (mapping_sv["MIN"].isna() | (mapping_sv["MIN"] == -1))][["BIN_RANGE", "N", "WOE", "IV"]])
```

The `WOE_Master` table has no `sv_policy_applied` column. A `neutral` bin shows `WOE == 0` and `IV == 0`, and a merged bin shows
`N == 0`.

### Declared but Unseen in the Fit Sample (`unseen_special_policy`, 0.8.2)

`special_values` is a **declaration**. If a feature has no `-1` row at all in the fit sample, the fitted table has no `[sv=-1]`
bin. `MonotoneWOEBinner(unseen_special_policy=...)` decides how that value is handled afterwards:

| Value | Bins table | Scoring (`apply_woe`, screening PSI and IV) | By-group charts and within-group IV |
|---|---|---|---|
| `"normal_bin"` (the default up to 0.8.2) | No bin | The value is binned as an ordinary number: `-1` falls in the lowest bin | Same as scoring: an ordinary number |
| `"neutral"` (default since 0.9.0) | A placeholder bin `[sv=-1]` is added: `n=0`, WOE `=missing_woe`, `iv=0`, decision `unseen_at_fit` | Scores `missing_woe` (0 by default, neutral) | The value's share inside the group is drawn separately and does not count toward the group IV |

```python
unseen = MonotoneWOEBinner(
    feature_cols=["age"], target_col="bad_flag",
    special_values=[-1, -999], unseen_special_policy="neutral",
)
unseen.fit(train_df)
print(unseen.get_final_bins()["age"][["bin_label", "n", "woe", "iv", "is_special"]])      # [sv=-999] has n=0
print(unseen.apply_woe(pd.DataFrame({"age": [30.0, -999.0]}))["age_woe"].tolist())       # -999 scores missing_woe
```

- The placeholder is added after all SV decisions, so it takes no part in the low-share fallback, merging, or smoothing, and
  the overall IV, the ordinary bin edges, and the WOE of other bins do not change.
- The placeholder is a visible row of the table. After `get_final_bins()` is exported, scoring stays neutral whether you
  reload through Format A or from CSV or Excel, and does not depend on the loader's constructor parameters.
- Only numeric special values get a placeholder: not NaN, not strings such as `"-1"`, and not categorical features. For a
  feature that declares NaN but had no missing values at fit time, missing values already score `missing_woe` in both modes.
- Whether a value "has a bin" is decided by the fitted table, numerically (`-1` and `-1.0` are the same value). Scoring,
  by-group charts, and the warning statistics use the same test, and it is unaffected if the loader declares `special_values`
  with another spelling. By-group charts split missing rows into their own bin whenever the table has a `[Missing]` bin, even
  if the loader did not declare NaN.
- A non-integer special value (such as `0.5`) matches only itself. In 0.8.1 and earlier the labels were truncated to an
  integer, so ordinary rows with the value `0` also got the WOE of `[sv=0.5]`. 0.8.2 fixes this, which changes scoring.
- Since 0.8.2, the monotone self-fit of `CreditModelPipeline` declares `-999999` only if that value occurs in the WOE fit
  sample (unless `special_values` is set in `monotone_woe_params`). Declaring it or not gives identical binning and
  scoring, so the only change is that the warning for a sentinel that does not exist, and the placeholder bins under
  `neutral`, are gone.

Both policies leave a trail, and the scored values do not depend on these records:

- `fit()`: `binner._unseen_special_at_fit` records `{feature: [values]}`. With `"normal_bin"`, `fit` issues one `UserWarning`
  that summarizes the whole fit, and writes a `logger.warning`.
- `apply_woe()`: `binner._unseen_special_stats` records, for the latest call, the values hit per feature, the row count, the
  share, and how they were handled (`normal_bin`, `neutral`, or `mixed`). It issues a `RuntimeWarning` and a
  `logger.warning`. With `unseen_category_policy="silent"` it does not warn but still records, and with `"raise"` it still
  only warns.
- `FeatureValidationPipeline` freezes these statistics after the transform of each split in
  `woe_artifacts["by_target"][target]["unseen_special_stats_by_split"]`, and keeps the batch summary in
  `woe_artifacts["unseen_special_stats_by_target"]`.
- Before 0.8.2, `import Modeling_Tool` switched warnings off for the whole process, so these warnings were invisible. See the
  [FAQ](../faq.md).

!!! warning "The default changed to `neutral` in 0.9.0"

    `"normal_bin"` let a sentinel (such as `-1` for "no record") take the WOE of a real value's bin, so 0.9.0 made `"neutral"`
    the default, as 0.8.2 announced. To reproduce the scoring of a binner fitted with 0.8.2 or earlier, pass
    `unseen_special_policy="normal_bin"` explicitly; a pickled binner keeps the setting it was created with (`"normal_bin"`
    for every binner from 0.8.2 or earlier). In the pipelines,
    when one feature holds the `-999999` sentinel it is declared for every monotone feature, so the features that never
    hold it get an empty `[sv=-999999]` row and score the sentinel as `missing_woe`.

### Pipeline-layer exposure

`CreditModelPipelineConfig` and `FeatureValidationPipelineConfig` both have two pass-through dictionaries with the `sv_*` keys
(and `unseen_special_policy` in the monotone one): `woe_params` feeds `WOE_Master.fit` (the `equal_freq` engine), and
`monotone_woe_params` feeds `MonotoneWOEBinner`.

```python
from Modeling_Tool import CreditModelPipelineConfig

cfg = CreditModelPipelineConfig(
    target_col="bad_flag",
    woe_engine="monotone",
    monotone_woe_params={
        "n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2,
        "special_values": [-999999],
        "sv_min_bin_size": 0.01,
        "sv_small_policy": "neutral",
    },
)
```

- A dictionary you pass replaces the default one. `MonotoneWOEBinner` fills the keys you leave out with its own defaults.
- When `special_values` is not in the dictionary, `CreditModelPipeline`, `FeatureValidationPipeline` and the screening engine
  of `feature_screen` all declare `-999999` as a special value if one of the fitted numeric features holds it (and nothing
  otherwise). Before, only `CreditModelPipeline` did, so a screening artifact handed to it binned the sentinel with the
  lowest real values while a self-fit gave it a bin of its own. Pass `special_values` (also `[]`) to decide yourself.
- In `woe_params`, the keys `woe_suffix` and `missing_ref_value` go to the `WOE_Master` constructor (the pipelines default
  `missing_ref_value` to `-999999`), and every other key goes to `fit()`.
- The `FeatureValidationPipeline` `config_snapshot` records both dictionaries as given, so the governance basis of a run can be
  reproduced.
- `CreditModelPipeline` hands every key of `monotone_woe_params` to the binner, and an unknown key raises `TypeError`.
  `FeatureValidationPipeline` and `feature_screen` instead keep only the keys on an allowlist and **silently drop the
  rest**, so a typo has no effect and no error. The allowlist holds every constructor parameter except `feature_cols`,
  `target_col`, and `cate_feats`: `n_init_bins`, `min_bin_size`, `min_n_bins`, `eps`, `missing_woe`, `special_values`,
  `bin_label_decimals`, `min_bad_count`, `min_good_count`, `small_bin_policy`, `monotone_direction`, `reference_target`,
  `direction_conflict_policy`, `missing_bin_strategy`, `refine_min_n_bins_policy`, the four `sv_*` keys, and
  `unseen_special_policy`. The keys `chi2_binning`, `chi2_p`, `chi2_init_size`, and `n_jobs` are passed to `fit()` instead.
