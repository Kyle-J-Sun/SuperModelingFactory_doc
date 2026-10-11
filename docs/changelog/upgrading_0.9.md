# Upgrading from 0.8.2 to 0.9.3

0.9.1 is the release that follows 0.8.2 on PyPI (0.9.0 was prepared but never published), then come 0.9.2 and 0.9.3.
They change some defaults and fix defects that change numbers. This page lists, in one place, what to pass to get the
0.8.2, 0.9.0, 0.9.1 or 0.9.2 behavior back where a switch exists, and which changes are bug fixes without one. The
details are in the [v0.9.0](v0.9.0.md), [v0.9.1](v0.9.1.md), [v0.9.2](v0.9.2.md) and [v0.9.3](v0.9.3.md) notes.
[Upgrading from 0.9.2 to 0.9.3](#upgrading-from-092-to-093) covers projects that rerun a screening.

## Changed Defaults

| Setting | 0.8.2 | 0.9.0 | 0.9.1 | 0.9.2 | 0.9.3 | Keep the old behavior with |
|---|---|---|---|---|---|---|
| `MonotoneWOEBinner(unseen_special_policy=...)` | `"normal_bin"` | `"neutral"` | `"neutral"` | `"neutral"` | `"neutral"` | `unseen_special_policy="normal_bin"` |
| `MonotoneWOEBinner(sv_total_basis=...)` | (did not exist; behaved as `"ordinary"`) | `"all"` | `"all"` | `"all"` | `"all"` | `sv_total_basis="ordinary"` |
| `CreditModelPipelineConfig(warm_start_score_scope=...)` | (did not exist; behaved as `"train"`) | `"full"` | `"full"` | `"full"` | `"full"` | `warm_start_score_scope="train"` |
| `MonotoneWOEBinner(min_bad_count=..., min_good_count=..., small_bin_policy=...)` | `None`, `None`, `None` | `None`, `None`, `None` | `1`, `1`, `"merge"` | `1`, `1`, `"merge"` | `1`, `1`, `"merge"` | `small_bin_policy=None` |
| Special-value bins below `min_bad_count` / `min_good_count` (class-pure with the defaults) | Not checked | Not checked | Reported: `UserWarning`, `sv_class_pure` column, `_sv_pure_stats`; numbers unchanged with the default `sv_small_policy="keep"`, neutralized or merged with the opt-in `"neutral"` / `"merge_missing"` | As in 0.9.1 | As in 0.9.1 | `min_bad_count=None, min_good_count=None` (also stops the merging of class-pure ordinary bins), or filter the warning |
| `ODPS_PROJECT`, `ODPS_ENDPOINT` for `ODPSRunner` | Optional, with built-in defaults | Optional, with built-in defaults | Required (`KeyError` if unset) | Required | Required | Set both variables |
| `screen_iv_convention` of `FeatureScreenConfig`, `FeatureValidationPipelineConfig`, `CreditModelPipelineConfig`, `VarExtractionInsights` and `CorrelationFilter`: the screening IV on WOE bins | (did not exist; behaved as `"raw"`) | (as `"raw"`) | (as `"raw"`) | `"guarded"`: class-pure bins are left out of the IV sum, which changes the IV, `iv_threshold`, the correlation decisions and the truncation ranking of features with such bins | As in 0.9.2 | `screen_iv_convention="raw"` (bit-for-bit the 0.9.1 IV tables and selections) |
| `iv_upper_threshold` (leakage limit) of `FeatureScreenConfig`, `screen_config_from_mapping`, `weighted_feature_screen`, the FVP `selection_params` and the CMP `feature_selection` | `None` (off) | `None` | `None` | `None` | `2.0`; on WOE bins under `"guarded"` it tests the larger of `iv_raw` and the floored IV, never the guarded `iv` | `iv_upper_threshold=None` (or your previous explicit value) |
| Leak warning: `UserWarning` and `iv_gap_flag` when `iv_raw / iv >= 2` or `iv_raw - iv >= 0.5` | (did not exist) | (did not exist) | (did not exist) | (did not exist) | On (`iv_gap_ratio_warn=2.0`, `iv_gap_abs_warn=0.5`); warns and flags only | Filter the warning, or set both thresholds to `inf` |
| `on_empty_stage="keep_all_warn"` (the default): a stage that would drop every feature | Keeps every feature | As in 0.8.2 | As in 0.8.2 | As in 0.8.2 | Never brings back a feature that `iv_upper_threshold` dropped; from the IV stage on keeps the `fallback_top_k` (5) features with the highest screening IV (`fallback_kept` in `dropped_detail`); raises `EmptyStageError` when only leak drops are left; the missing-rate and PSI stages still keep every feature | `on_empty_stage="keep_all"` (new value; for comparisons with old results only) |
| Leak-gate warning: a `UserWarning` per feature that `iv_upper_threshold` drops | (did not exist) | (did not exist) | (did not exist) | (did not exist) | On whenever the limit drops a feature ("possible target leakage; check the field's time point (PIT) and source before training") | Filter the warning, or switch the limit off |

The three `MonotoneWOEBinner` settings are also keys of `monotone_woe_params` in `CreditModelPipelineConfig` and
`FeatureValidationPipelineConfig` (and of `feature_screen`). A dict passed as `monotone_woe_params` replaces the default
dict, so start from the default when you add the legacy keys:

```python
from Modeling_Tool import CreditModelPipelineConfig

legacy_woe = dict(
    CreditModelPipelineConfig().monotone_woe_params,
    unseen_special_policy="normal_bin",
    sv_total_basis="ordinary",
    small_bin_policy=None,
)
config = CreditModelPipelineConfig(monotone_woe_params=legacy_woe, warm_start_score_scope="train")
print(config.monotone_woe_params["small_bin_policy"], config.warm_start_score_scope)
```

The screening IV of 0.9.1 comes back with one setting per object; it does not touch the binning:

```python
from Modeling_Tool import CreditModelPipelineConfig, FeatureScreenConfig, FeatureValidationPipelineConfig

screen = FeatureScreenConfig(iv_use_woe_bins=True, screen_iv_convention="raw")
fvp = FeatureValidationPipelineConfig(screen_iv_convention="raw")
cmp = CreditModelPipelineConfig(screen_iv_convention="raw")
print(screen.screen_iv_convention, fvp.screen_iv_convention, cmp.screen_iv_convention)
```

```python
from Modeling_Tool import MonotoneWOEBinner

binner = MonotoneWOEBinner(
    feature_cols=["score"],
    target_col="is_bad",
    unseen_special_policy="normal_bin",
    sv_total_basis="ordinary",
    small_bin_policy=None,
)
print(binner.small_bin_policy, binner.sv_total_basis)
```

## Saved Objects

- Scorecards, WOE tables and models built with an earlier version keep their numbers until they are refitted.
- A pickled `MonotoneWOEBinner` keeps the settings it was created with when it is loaded, applied or refitted. A binner
  from 0.8.2 or earlier keeps `unseen_special_policy="normal_bin"`, `sv_total_basis="ordinary"` and
  `small_bin_policy=None`; a binner from 0.9.0 keeps `small_bin_policy=None`.
- Bins loaded with `load_woe_bins` keep the WOE they were saved with.
- With the default `min_bin_size=0.03`, `small_bin_policy="merge"` also merges an ordinary bin under 3% of the rows; in
  the default flow such a bin only comes from tied values at the equal-frequency edges. `min_bin_size=0` restricts the
  merging to class-pure bins.
- The `.pyi` type stubs of 0.9.1 show the current signatures and defaults (they were stale up to 0.9.0).
- `screen_iv_convention` (0.9.2) changes the screening only: binners, WOE tables and `apply_woe` give the same values as
  in 0.9.1, so existing models score identically. When you build a new model, resolve every class-pure bin (flagged
  `sv_class_pure`, or listed in the screening's `iv_sv_pure_bins`) before training: its WOE is set by `eps=1e-6` and has
  no statistical meaning.

## Upgrading from 0.9.2 to 0.9.3

0.9.3 turns the leakage limit on (`iv_upper_threshold=2.0`) and adds a leak warning, so a rerun of the screening can drop
features that 0.9.2 kept: those whose floored IV or `iv_raw` is above 2.0. A dropped leak is never brought back by an
empty-stage fallback, and that fallback now keeps at most `fallback_top_k` (5) features from the IV stage on, so a stage
that every feature fails can keep fewer features than in 0.9.2, or stop with `EmptyStageError` when only leak drops are
left.

!!! note "Projects that are mid-flight or already frozen"

    If you rerun the screening of a project whose variables are already chosen, pass
    `screen_iv_convention="raw"` and `iv_upper_threshold=None` (or the explicit value you used before) to keep the same
    selected variables. Under `"raw"` the limit tests the same value as in 0.9.2, so a previous explicit value gives the
    previous drops. If a stage of that screening ever dropped every feature (a `<stage>_fallback` row in its summary),
    also pass `on_empty_stage="keep_all"`, the fallback of 0.9.2; it changes nothing otherwise, and it is meant only for
    reproducing old results. The warnings may still appear; they change nothing.

```python
from Modeling_Tool import CreditModelPipelineConfig, FeatureScreenConfig, FeatureValidationPipelineConfig

frozen_screen = FeatureScreenConfig(
    iv_use_woe_bins=True, screen_iv_convention="raw", iv_upper_threshold=None, on_empty_stage="keep_all",
)
frozen_fvp = FeatureValidationPipelineConfig(
    screen_iv_convention="raw", selection_params={"iv_upper_threshold": None, "on_empty_stage": "keep_all"},
)
frozen_cmp = CreditModelPipelineConfig(
    screen_iv_convention="raw",
    feature_selection=dict(
        CreditModelPipelineConfig().feature_selection, iv_upper_threshold=None, on_empty_stage="keep_all",
    ),
)
print(frozen_screen.on_empty_stage, frozen_fvp.selection_params, frozen_cmp.feature_selection["iv_upper_threshold"])
```

New projects should keep the defaults and review every feature that the warnings name or that `dropped_detail` lists
with `iv_above_upper` / `iv_above_upper_sv_pure` (check its time point and source), and every feature marked
`fallback_kept`. An `EmptyStageError` that names only leak drops means that nothing else is left to screen: fix the data
or the feature list rather than switching the fallback back.

## New Outputs (0.9.2)

These add information without changing a number of 0.9.1:

| Output | What it adds |
|---|---|
| `iv_table`, the `iv` table of the selection summaries, `ivks_summary`, the WOE-engine report of `VarExtractionInsights` | `iv` (under `screen_iv_convention`), `iv_raw` (every bin counted, the 0.9.1 value), `iv_sv_pure` and `n_sv_pure` (the IV and rows of the class-pure bins). `iv_table` keeps `iv_weighted`, equal to `iv` |
| `stage_tables["iv_sv_pure_bins"]`, `FeatureValidationPipelineResult.ivks_sv_pure_bins`, `feature_selection_summary["iv_sv_pure_bins"]`, `VarExtractionInsights.iv_sv_pure_bins` | The class-pure special-value bins, with `n < 30` marked `low_count`, as candidate rules for review |
| `dropped_detail` of the leakage gate (`iv_upper_threshold`) | The reason `"iv_above_upper_sv_pure"` when a feature exceeds the limit only because of its class-pure special-value bins; the gate drops the same features as in 0.9.1 |
| `iv_gap_flag` in `iv_table` and the `iv` tables of the selection summaries (0.9.3) | True when `iv_raw` is far above `iv` (the leak warning); `CreditModelPipeline` adds `dropped_detail` to `feature_selection_summary` |
| `dropped_detail` reason `fallback_kept` and `FeatureScreenConfig.fallback_top_k` (0.9.3) | The features that an empty-stage fallback kept; `feature_selection_summary["hard_dropped"]` lists the leak drops that a failed screening of `CreditModelPipeline` keeps out, and the FVP `config_snapshot` records `fallback_top_k` |

## Fixes Without a Legacy Switch

These changes correct results that were wrong; no setting brings the old numbers back.

| Area | What changes |
|---|---|
| Equal-frequency binning with `include_missing=True` (`WOE_Master`, `get_woe_table`, `WOETransformer`, the non-monotone `woe_engine`, unweighted Gains and cross-risk tables) | Missing values get their own bin instead of sharing the lowest bin with real values; the default fill value `-1.797e308` no longer overflows to `-inf` |
| `RejectInferencePipeline` | Rows without a label are no longer trained as goods; `ri_score_direction="high_good"` with a pre-score trained by the pipeline raises instead of inverting the reject labels; the random OOT is held out of the pre-score, so OOT labels no longer reach the RI models through the inferred reject labels; an external reference is rescored with the pipeline's pre-score |
| `ScoreComparisonPipeline`, `model_perf_compare` | Scores are compared on the same rows (`perf_common_rows=False` evaluates each score on its own rows, which reproduces 0.9.0 with one comparison score); numeric group columns are evaluated; the weighted Gains table no longer counts unlabelled rows as goods or ranks missing scores; a group with exactly `min_data_size` rows is kept |
| Everything listed in the v0.9.0 notes, sections 2, 4 and 5 | See the "Results that change" boxes there |
