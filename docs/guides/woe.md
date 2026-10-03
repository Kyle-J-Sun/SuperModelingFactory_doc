# WOE Encoding

WOE (Weight of Evidence) maps categorical variables or binned continuous variables to linearly separable numeric values, and is the standard practice in scorecard modeling.

SuperModelingFactory provides a **master class + monotone binner + transformers + plotters + binning-engine adapter** in the [`WOE`](../api/woe.md) subpackage.

## 1. Master Class — `WOE_Master`

```python
from Modeling_Tool import SMF_MISSING_BIN, WOE_Master

woe = WOE_Master(
    train_data=train_df,
    varlist=features,
    dep="bad_flag",
    missing_ref_value=SMF_MISSING_BIN,
)
woe.fit(nbins=10, equal_freq=True)

train_woe = woe.transform(train_df)
test_woe = woe.transform(test_df)
oot_woe = woe.transform(oot_df)
```

### Vectorized Execution and Wide-Table Performance

`WOE_Master` computes row-level work on a vectorized path:

- Binning generates categorical codes through `pandas.cut`, then maps a whole column of bin labels at once with `numpy.take`.
- WOE mapping is a whole-column lookup, with no row-by-row `Series.apply()`.
- Multi-variable transforms first collect all `{feature}_woe` arrays and then `concat` them back to the original table in one go, avoiding DataFrame fragmentation from inserting wide-table columns one by one.
- WOE and IV share a single logarithm computation; `WOEIVCalculator.calc_both()` does not compute WOE and IV separately and redundantly.
- `WOE_Master.transform()` automatically reuses the instance's `missing_ref_value`, so training and inference use the same missing-value bin basis.

Different variables have different bin boundaries, so `fit()` / `transform()` still keep a **per-variable loop**; there are no row-by-row Python callbacks inside the loop. Vectorized here means whole-column operations, not forcing every variable to share one set of bins.

An existing mapping table can also be applied in batch directly. When calling `mapping_woe()` on its own, if training used a custom missing sentinel, pass the same value explicitly:

```python
from Modeling_Tool import mapping_woe

scored = mapping_woe(
    data=oot_df,
    varlist=features,
    woe_mapping_table=woe.get_mapping_table(),
    missing_ref_value=SMF_MISSING_BIN,
)
```

When you need both WOE and IV, prefer the one-shot interface:

```python
from Modeling_Tool import WOEIVCalculator

woe_values, iv_values = WOEIVCalculator(
    bin_summary,
    bad_pct_col="BAD_PCT_PER_BIN",
    good_pct_col="GOOD_PCT_PER_BIN",
).calc_both()
```

### Persisting the Mapping Table

```python
woe.save_mapping_table("./output/woe_mapping.csv")

from Modeling_Tool import load_mapping_table
varlist, woe_dict = load_mapping_table("./output/woe_mapping.csv")
```

## 2. Greedy Monotone Binner — `MonotoneWOEBinner`

If the scorecard needs a stronger monotonicity constraint, `MonotoneWOEBinner` is recommended.

```python
from Modeling_Tool.WOE.WOE_Monotone_Binner import MonotoneWOEBinner

binner = MonotoneWOEBinner(
    feature_cols=features,
    target_col="bad_flag",
    n_init_bins=20,
    min_bin_size=0.03,
    special_values=[-1, -100, -999999],
    cate_feats=["city_grade"],
)
binner.fit(train_df, chi2_binning=True, chi2_p=0.95)
binner.refine_cate(max_bins=5)

train_woe = binner.apply_woe(train_df)
bins = binner.get_final_bins()
edges = binner.get_bin_edges()
```

### Method List

| Method | Description |
|------|------|
| `fit(df, chi2_binning, chi2_p, n_jobs)` | Fit on training data |
| `refine_cate(max_bins)` | Cluster and merge categorical features by bad rate |
| `apply_woe(df, varlist=None)` | WOE transform; `varlist` can restrict it to specified variables, suited to wide-table chunking |
| `get_final_bins()` | Export the binning result (with WOE/IV) |
| `load_woe_bins(bins_dict)` | Load existing bins |
| `get_bin_edges()` | Get the list of bin boundaries |
| `export_woe_report(path)` | Export an Excel report |
| `plot_woe_graph(dir, group_name=)` | Output WOE plot PNGs |

### Format-A Bins Round Trip (0.7.2)

Every DataFrame returned by `get_final_bins()` carries `attrs["smf_woe_format_a"]`. The exact numeric boundaries, sparse bin IDs, category members, and `missing_woe` in it are used by `load_woe_bins()` only when the metadata digest and the row-identity check pass; passing the frame directly or round-tripping through pickle preserves this information, while corrupted or stale metadata safely falls back to parsing the visible labels.

For tables with low-share special-value governance enabled (see SV Bin Governance below), attrs also carries each special-value bin's `sv_policy_applied` decision and the smoothing parameters at fit time (`sv_decisions`). This part has its own validation digest and does not take part in the original metadata digest: old loaders validate the original fields as usual, and new loaders restore the decisions after loading, so the group IV in by-group plots matches fit time. Tables without SV governance have attrs identical to the old version; the exception is tables where `unseen_special_policy="neutral"` produced placeholder bins (see below), which carry `sv_policy_applied` and have their decisions written as usual. Loaders in 0.8.1 and earlier do not recognize `unseen_at_fit` and ignore this decision wholesale (group IV is computed as if there were no decision); scoring is unaffected.

!!! warning "CSV/Excel is not an exact round-trip carrier"
    CSV/Excel lose `DataFrame.attrs`. When reloading from these two kinds of files, `bin_label` (default `.8g` display precision) is the only source of truth, so exact cut points beyond the text, original sparse bin IDs, ambiguous category members, or a non-default `missing_woe` cannot be restored. When you need exact recovery, pass the DataFrame directly or use a format that preserves attrs, such as pickle.

Categorical transforms first do an exact match and then a supported `str()` dtype fallback. Only values that fail both enter `_unseen_category_stats`; values matched by the fallback are still counted in `_categorical_transform_stats[feature]["fallback_match_rows"]` and trigger the existing tripwire warning.

!!! note "Clustered by-group image specs"
    When `group_name` is non-empty and `bar_mode="clustered"`, the by-group WOE plot always uses
    `figsize=(16, 6)` and `dpi=200`, to fit the side-by-side bars, by-group WOE curves, and the legend on the right.
    Other modes still use the `figsize` and `dpi` passed in by the caller.

!!! note "Basis of the IV and WOE lines in by-group plots"
    The per-group IV in the by-group plot legend (`pooled` / `clustered`) and the subplot titles (`small_multiples`)
    is the **within-group IV**, on the same basis as the overall plot's IV, just with the sample replaced by that group:

    - Ordinary bins: the denominators are the bad/good of the group's rows that fall into ordinary bins;
    - Special-value / missing bins: the denominators are the bad/good of all the group's rows, and the governance decision made at fit time for each special-value bin
      (`sv_policy_applied`) is reused, with no re-judging of the share within the group — `keep` takes the empirical value (when fit had smoothing on, it is smoothed with the
      parameters from fit time), `neutral` counts as 0, and `merged_into_missing` is merged into the group's `[Missing]` bin;
    - Bins that have only good samples or only bad samples within the group and were not laplace-smoothed (ordinary bins, the merged `[Missing]`,
      special-value bins without smoothing) are excluded from the group IV, consistent with the `iv_guard` basis of the screening IV, to keep a few empty-class bins from being
      inflated by eps into an overstated IV; laplace-smoothed special-value bins have a finite WOE and are counted as usual even when single-class.

    When all unsmoothed bins in the fit sample have both classes, treating the whole fit sample as one group makes the group IV equal to the overall IV. Groups with fewer than 5 rows or only
    one class show `IV=0.000`; for categorical features, values that did not appear at fit time are excluded from the ordinary-bin denominators. Bins loaded through
    `get_final_bins()` → `load_woe_bins()` (with attrs preserved) reuse the fit-time decisions as well; reloading from CSV/Excel
    or Format B carries no decisions, and special-value bins are always counted with the empirical value of `keep`.

    The per-group **WOE lines** are still based on the full-sample bad/good. For bins with both classes, a group's line equals the within-group WOE plus the constant
    `ln(group's share of total bad / group's share of total good)`: the line shape matches the within-group one,
    and its vertical position reflects the group's bad-debt level relative to the overall.

    In 0.8.0 and earlier, each group's IV used the full sample as the denominator and excluded special-value bins, so k similarly sized groups each showed only
    about 1/k of the within-group IV; when fitted bin IDs were non-contiguous, each group's bars and lines were also misaligned. By-group plots produced by those versions
    should not be compared directly with the overall IV.

## 3. Unified Binning Engine — `as_woe_engine`

`WOE_Master` and `MonotoneWOEBinner` have different internal artifact formats. `as_woe_engine()` converts them to a unified interface for PSI, IV, and correlation screening to reuse.

```python
from Modeling_Tool import as_woe_engine

engine = as_woe_engine(binner)   # a WOE_Master can also be passed
woe_table = engine.get_woe_table(features)
train_woe = engine.transform(train_df, features)
```

See [WOE Binning Engine](woe_binning_engine.md) for more.

## 4. Linking with Feature Screening

Fit the binner once at training time, then reuse the same object for later screening, monitoring, and modeling:

```python
from Modeling_Tool import PSICalculator, VarExtractionInsights, CorrelationFilter

psi = PSICalculator(binning_engine=binner).calculate(train_df, oot_df, features)

iv_report = VarExtractionInsights(
    train_df, "bad_flag", "./iv_plots/",
    woe_engine="monotone", woe_binner=binner,
).get_var_analysis_report(train_df, features)

keep_vars = CorrelationFilter(
    train_df, "bad_flag", corr_cutpoint=0.7,
    woe_engine="monotone", woe_binner=binner,
).remove_highly_correlated(features)

train_woe = binner.apply_woe(train_df)
```

## 5. Monotonicity Check

```python
from Modeling_Tool import is_monotonic, get_overall_woe_table

for var in features:
    woe_table = get_overall_woe_table(woe, train_df, [var])
    mono, direction = is_monotonic(woe_table, "WOE", direction="auto")
    print(var, mono, direction)
```

## 6. Standalone WOE Transforms

```python
from Modeling_Tool import woe_transform, woe_transformation

single_df, single_map = woe_transform(train_df, var="age", dep="bad_flag", nbins=10)
batch_result = woe_transformation(train_df, varlist=features, dep="bad_flag", nbins=10)
```

## FAQ

??? question "When should I choose MonotoneWOEBinner?"

    When a variable will enter the scorecard and needs stronger interpretability and a monotonicity constraint, prefer `MonotoneWOEBinner`.

??? question "Why pass the binner at the screening stage?"

    Because PSI / IV / KS should be computed on the same binning that finally goes live; otherwise the screening metrics and the model inputs may be inconsistent.

## Binning Governance (0.6.7+, G08/G09/G17)

`MonotoneWOEBinner` adds three groups of governance parameters, all off by default (`None`/`"auto"`), with behavior byte-for-byte identical to the old version:

```python
binner = MonotoneWOEBinner(
    feature_cols=feats, target_col="y",
    # G08 small-bin governance: lower limits on bad/good counts + three-state policy
    min_bad_count=50, min_good_count=50, small_bin_policy="merge",  # merge/warn/raise
    # G09 direction governance: a fixed direction or one derived from a reference label; three-state conflict handling
    monotone_direction={"util_rate": "increasing"},   # or "increasing"/"decreasing"/"auto"
    reference_target="y",                              # mutually exclusive with monotone_direction
    direction_conflict_policy="raise",                 # warn/raise/keep
    # Missing-bin semantics: empirical_special / fixed_woe / fail
    missing_bin_strategy="fail",
    # G17 refine governance: warn/enforce/raise when the refined bin count falls below min_n_bins
    refine_min_n_bins_policy="enforce",
)
binner.fit(train)
binner.refine_dtree(train, max_depth=3)   # 0.6.7+: tree depth can be limited
binner.get_direction_summary()            # feat / direction / direction_basis / is_monotonic
```

Key points:

- `small_bin_policy="merge"` merges into the neighboring bin whose WOE is closer, with the merge trail recorded in the result's `merge_trace`; `raise` throws `BinningPolicyViolation` (which penetrates the per-feature fault tolerance and is not swallowed into the log). Since 0.7.2, categorical features also run `merge/warn/raise` during the initial `fit()`, and `merge` never goes below `min_n_bins`; if violating bins remain at the lower limit, you can audit them from `_small_bin_stats[feature]["remaining_violation"]`.
- The direction is resolved before `fit`: serial and parallel workers use the same `_expected_direction`, ruling out serial/parallel drift.
- These parameters can be passed through `monotone_woe_params` from FVP / CMP / feature_screen straight to the underlying binner.
- Since 0.7.1, `refine_min_n_bins_policy` defaults to `"warn"`; to turn this check off completely, pass `None` explicitly.

## Low-Share Special-Value Governance (SV Bin Governance, 0.8.0)

The `small_bin_policy` / `min_bin_size` of 0.6.7 govern only **ordinary interval bins**; special-value (hereafter SV) bins have always taken the empirical WOE **unconditionally** and counted toward the total IV. When an SV's share is extremely low (for example `-1` is only 0.05%), `ln(pct_bad/pct_good)` is estimated from very few samples, with huge variance, an overstated IV, and a PSI that easily drifts after going live.

0.8.0 introduces **two orthogonal** SV governance switches for this. The four parameters have **the same name, meaning, and basis on both engines**: for `MonotoneWOEBinner.__init__` they are constructor parameters, and for `WOE_Master.fit()` / `update_woe()` they are method parameters.

| Parameter | Type / default | Values | Semantics |
|------|-------------|------|------|
| `sv_min_bin_size` | `float = 0.0` | `[0.0, 1.0)` | Threshold on the SV bin's share of the **full sample**; `0.0` = off |
| `sv_small_policy` | `str = "keep"` | `keep` / `neutral` / `merge_missing` | How SV bins below the threshold are handled |
| `sv_woe_smoothing` | `str = "none"` | `none` / `laplace` | Whether to shrink the SV bin's WOE toward the global base rate |
| `sv_smoothing_alpha` | `float = 0.0` | `>= 0.0` | Smoothing strength α; `0.0` = off |

!!! note "Defaults are strictly equal to the old behavior"
    The default combination of the four parameters (`0.0` / `"keep"` / `"none"` / `0.0`) is **bit-for-bit identical** to 0.7.2, so upgrading to 0.8.0 does not change any existing output. Illegal values raise `ValueError` at the `__init__` / `fit()` entry, in the same style as the G08 `small_bin_policy`.

### Mode 1 — Low-Share Fallback (`sv_min_bin_size` + `sv_small_policy`)

The share is computed as `prop = n_bin / N_total` (**no eps added to the denominator**), and the test is **strictly less than**: `prop < sv_min_bin_size`; equality does **not** trigger it.

- `keep` (default): take the empirical WOE, zero behavior change.
- `neutral`: the sub-threshold SV bin gets `woe = 0.0` and `iv = 0.0`. The most stable option, equivalent to "this SV provides no evidence at all".
- `merge_missing`: the `bad` / `good` counts of the sub-threshold SV bin are **merged into the `[Missing]` bin**, and the `[Missing]` bin's WOE is then **recomputed** by the empirical formula; the stored `woe` of the merged row is **rewritten to the `[Missing]` bin's recomputed WOE**, and `iv` is set to `0` to avoid double-counting in the total IV. If the feature has **no** `[Missing]` bin → it degrades to `neutral` with a `warnings.warn(UserWarning)`.

!!! tip "`merge_missing` needs no change to transform"
    It uses a **rewrite-stored-WOE** scheme: the stored WOE of the merged SV row is written directly as the `[Missing]` WOE. So `apply_woe()` / `mapping_woe()` simply look up `bin_label → WOE` as usual and hit the correct value; **neither engine's transform path has any change**, and the fit→transform round trip is automatically consistent.

The governance result can be audited from the `sv_policy_applied` column of the result table, whose values are
`keep` / `neutral` / `neutral(fallback)` / `merged_into_missing` / `merge_target`,
and, from 0.8.2, `unseen_at_fit` (the placeholder bin for a declared special value that did not appear in the fit sample; see below).

### Mode 2 — SV WOE Smoothing (`sv_woe_smoothing="laplace"`)

Smoothing **applies only to SV bins**; ordinary interval bins are completely unaffected. It uses **bad-rate shrinkage**: first shrink the within-bin bad rate toward the global bad rate, then convert back to count shares.

```
p = N_bad / (N_bad + N_good)          # global bad rate
n_bin = n_bad + n_good                # sample size in the bin

r = (n_bad + alpha * p) / (n_bin + alpha)        # bad rate shrunk toward p; alpha competes with n_bin

pct_bad_smoothed  = n_bin * r       / N_bad      # converted back to shares; denominators are still global N_bad / N_good
pct_good_smoothed = n_bin * (1 - r) / N_good

woe = ln(pct_bad_smoothed / pct_good_smoothed)
iv  = (pct_bad_smoothed - pct_good_smoothed) * woe
```

Convergence properties (and the reason this form was chosen):

- `alpha = 0` → `r` degenerates to the empirical bad rate, **restoring bit-for-bit** the old empirical WOE (a regression guardrail).
- `alpha → ∞` → `r → p`, the ratio of the two shares tends to the global ratio, `woe → 0`, with **monotone** shrinkage.
- α competes with `n_bin` ⇒ **the fewer samples a bin has, the stronger the shrinkage**, exactly the property low-share SVs need.

!!! danger "Do not use the 'population-denominator pseudo-count' form"
    An early design put the pseudo-count on the population denominator (`(n_bad + alpha*p) / (N_bad + alpha)`). As `alpha → ∞`, that expression converges to `logit(p) = ln(p/(1-p)) ≠ 0`, and it is **not monotone** — increasing the smoothing strength can instead push |WOE| higher. Both engines implement the bad-rate shrinkage form above, and the formula above is the single authoritative definition.

### Orthogonal Combination Semantics (Fixed Execution Order)

The two switches can be enabled independently or stacked. At fit time, each SV bin is decided in a fixed order:

```
1. prop = n_sv / N_total
2. if sv_small_policy != "keep" and sv_min_bin_size > 0 and prop < sv_min_bin_size:
       take the Mode 1 fallback (neutral / merge_missing), and [do not] go through Mode 2
   else:
       if sv_woe_smoothing == "laplace" and alpha > 0 → take the Mode 2 smoothing
       otherwise → empirical WOE (old behavior)
```

That is, **Mode 1 takes priority**: a sub-threshold bin is **never smoothed**, since it has already been handled by the fallback; Mode 2 applies only to SV bins that "meet the share requirement and keep the empirical WOE".

When `merge_missing` + `laplace` are both enabled: the merge target `[Missing]` bin is recomputed by the **empirical** formula (**not** smoothed), because the merged bucket should reflect the true post-merge bad rate; other SV bins that meet the share requirement are still smoothed as usual.

### The `[Missing]` Bin Is Also a Governed SV Bin

In **both engines**, the `[Missing]` (NaN) bin is a normal governed SV bin: if its share meets the requirement it is smoothed as usual, and if it is below the threshold it is zeroed by `neutral` as usual. Its only special trait is that under `merge_missing` it **can be only a merge target, not a merge source**.

This is **orthogonal** to `missing_bin_strategy` (`empirical_special` / `fixed_woe` / `fail`): the latter governs only the semantics of the NaN missing bin, while the former governs **all** SV bins (including non-NaN sentinel values such as `-1`).

### Declared but Unseen in the Fit Sample (`unseen_special_policy`, 0.8.2)

`special_values` is a **declaration**: if a feature has not a single `-1` row in the training set, the fitted table has no `[sv=-1]` bin. `MonotoneWOEBinner(unseen_special_policy=...)` decides how such values are handled afterward:

| Value | Bins table | `apply_woe` scoring, screening PSI / IV | By-group plots and within-group IV |
|---|---|---|---|
| `"normal_bin"` (default, old behavior) | No bin is created | Binned as an ordinary number (`-1` falls into the lowest bin, e.g. "0 cards") | Consistent with scoring, binned as an ordinary number |
| `"neutral"` | A placeholder bin `[sv=-1]` is appended: n=0, WOE=`missing_woe`, IV=0, `sv_policy_applied="unseen_at_fit"` | Takes `missing_woe` (default 0, neutral) | The value's within-group share is drawn separately; it is not counted in the group IV |

- The placeholder bin is appended after all SV governance decisions, and takes no part in low-share fallback, merging, or smoothing; the overall IV, ordinary-bin boundaries, and WOE are all unaffected.
- The placeholder bin is a **visible row** in the bins table: after `get_final_bins()` is exported, scoring stays neutral whether reloaded through Format-A attrs or CSV/Excel, and does not depend on the loader's constructor parameters.
- It applies only to numeric special values: for a feature that declares NaN but had no missing values at fit time, missing values already score as `missing_woe` in both modes; declarations in string form (such as `"-1"`) get no placeholder bin; categorical features are not covered.
- Whether a value "has a bin" is decided by the **fitted table**, numerically (`-1` and `-1.0` count as the same value); scoring, by-group plots, and warning statistics use the same test, and it is unaffected if the loader declares `special_values` with a different spelling.
- A non-integer special value (such as `0.5`) matches only itself; in 0.8.1 and earlier, special-value bin labels were truncated to an integer key, so ordinary samples with the value `0` also got the WOE of `[sv=0.5]`, which 0.8.2 fixes as well.
- By-group plots also split special values by the fitted table (0.8.2): when the table has a `[Missing]` bin, missing rows form their own bin even if the loader did not declare NaN (consistent with scoring; 0.8.1 dropped these rows from the plots and the group IV); when two rows of the table parse to the same numeric value (for example Format B building `[sv=-1]` and `[sv=-1.0]` from `special_values=[-1, -1.0]`), each row is counted only in the first matching bin. The single exception is a value declared as `-inf` with no bin: scoring puts it in the lowest bin, while the plot (`pd.cut`) does not count those rows.
- The monotone self-fit of `CreditModelPipeline` used to always declare `-999999` by default; since 0.8.2 it declares it only if that value actually appears in the WOE fit sample (unchanged when `special_values` is given explicitly in `monotone_woe_params`). When it does not appear, declaring it or not gives identical binning and scoring, so the only change is that there is no longer a warning for a nonexistent sentinel, and no placeholder bin is added to every feature under `neutral`.

Both modes leave a trail, and scoring values are unaffected by these records:

- `fit()`: `binner._unseen_special_at_fit` records `{feature: [values]}`; under `"normal_bin"`, one `UserWarning` is issued for the whole fit as a summary, and `logger.warning` is also written.
- `apply_woe()`: `binner._unseen_special_stats` records, for the most recent call, the values hit by each feature (spelled as declared on this instance, or by the numeric value in the table when undeclared), the row count, the share, and how they were handled (`normal_bin` / `neutral` / `mixed`), and issues a `RuntimeWarning` + `logger.warning`; with `unseen_category_policy="silent"` it does not warn but still records the statistics, and with `"raise"` it still only warns and does not raise.
- `FeatureValidationPipeline` freezes these statistics after each split's transform: `woe_artifacts["by_target"][target]["unseen_special_stats_by_split"]`, with the batch/slim summary kept in `woe_artifacts["unseen_special_stats_by_target"]` (merged by the same rules as the categorical unseen statistics).
- In 0.8.1 and earlier, `import Modeling_Tool` suppressed warnings globally, so these `warnings.warn` calls were invisible in an ordinary session; since 0.8.2 they are no longer suppressed, see the [FAQ](../faq.md).

!!! warning "The default is planned to change to `neutral` in 0.9.0"
    `"normal_bin"` lets a sentinel value (such as `-1` meaning "no record") take the WOE of a real value's bin. 0.8.2 first lands it as an optional parameter, and the next minor version changes the default to `"neutral"`; if you need to keep the old scoring basis, pass `unseen_special_policy="normal_bin"` explicitly.

### Usage Example

```python
binner = MonotoneWOEBinner(
    feature_cols=feats, target_col="y",
    special_values=[-1, -999999, float("nan")],
    # SV bins with a share < 1% are neutralized directly
    sv_min_bin_size=0.01,
    sv_small_policy="neutral",       # keep / neutral / merge_missing
    # SV bins that meet the share requirement get bad-rate shrinkage
    sv_woe_smoothing="laplace",      # none / laplace
    sv_smoothing_alpha=50.0,
)
binner.fit(train)
binner.get_final_bins()["risk_score"]   # the sv_policy_applied column records the actual treatment of each SV bin
```

The `WOE_Master` side has the same names and meanings, only they sit on `fit()`:

```python
woe = WOE_Master(train_data=train, varlist=feats, dep="y")
woe.fit(
    nbins=10, equal_freq=True, spec_values=[-1, -999999],
    sv_min_bin_size=0.01, sv_small_policy="neutral",
    sv_woe_smoothing="laplace", sv_smoothing_alpha=50.0,
)
```

### Pipeline-Layer Exposure

**Both** pass-through dicts of `CreditModelPipelineConfig` and `FeatureValidationPipelineConfig` now carry the four `sv_*` default keys: `woe_params` (feeding `equal_freq` / `WOE_Master`) and `monotone_woe_params` (feeding `MonotoneWOEBinner`).

```python
from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig

cfg = CreditModelPipelineConfig(
    target_col="y",
    woe_engine="monotone",
    monotone_woe_params={
        "n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2,
        "special_values": [-999999],
        "sv_min_bin_size": 0.01,
        "sv_small_policy": "neutral",
    },
)
```

The FVP `config_snapshot` dumps these two dicts verbatim, so `sv_*` automatically enters the snapshot, which makes it easy to reproduce "which SV governance basis a model used".

!!! warning "Maintainer note: the FVP allowlist must be kept in sync"
    The monotone branch of `FeatureValidationPipeline` does **not** pass `**monotone_woe_params` directly; it first goes through
    the `_MONOTONE_INIT_KEYS` allowlist:

    ```python
    init_params = {k: v for k, v in params.items() if k in self._MONOTONE_INIT_KEYS}
    ```

    **Keys not on the allowlist are silently dropped, with no error and no warning** — the parameter looks like it was passed in, but the governance actually never took effect.
    The four `sv_*` keys have been added to `_MONOTONE_INIT_KEYS` (and also to
    `Feature_Screen._MONOTONE_INIT_KEYS`, otherwise the WOE fit reused at screening time would miss the governance, splitting the screening IV and the modeling WOE onto different bases).
    From now on, any new parameter added to `MonotoneWOEBinner.__init__` must be added to both allowlists.
    The `unseen_special_policy` added in 0.8.2 belongs only to the monotone engine: it has been added to both allowlists and to the
    `monotone_woe_params` default dict, and does not go into `woe_params`.

    `sv_*` are **constructor** parameters; **do not** add them to `_MONOTONE_FIT_KEYS` (`{chi2_binning, chi2_p, chi2_init_size, n_jobs}`),
    otherwise they would be passed down as `fit()` kwargs and raise `TypeError`. The same goes for the CM-side monotone fit-only `pop` list:
    keeping it free of `sv_*` is what lets them correctly reach the constructor.
