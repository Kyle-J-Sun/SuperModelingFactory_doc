# Upgrading from 0.8.2 to 0.9.1

0.9.1 is the release that follows 0.8.2 on PyPI (0.9.0 was prepared but never published). It changes some defaults and
fixes defects that change numbers. This page lists, in one place, what to pass to get the 0.8.2 (or 0.9.0) behavior back
where a switch exists, and which changes are bug fixes without one. The details are in the [v0.9.0](v0.9.0.md) and
[v0.9.1](v0.9.1.md) notes.

## Changed Defaults

| Setting | 0.8.2 | 0.9.0 | 0.9.1 | Keep the old behavior with |
|---|---|---|---|---|
| `MonotoneWOEBinner(unseen_special_policy=...)` | `"normal_bin"` | `"neutral"` | `"neutral"` | `unseen_special_policy="normal_bin"` |
| `MonotoneWOEBinner(sv_total_basis=...)` | (did not exist; behaved as `"ordinary"`) | `"all"` | `"all"` | `sv_total_basis="ordinary"` |
| `CreditModelPipelineConfig(warm_start_score_scope=...)` | (did not exist; behaved as `"train"`) | `"full"` | `"full"` | `warm_start_score_scope="train"` |
| `MonotoneWOEBinner(min_bad_count=..., min_good_count=..., small_bin_policy=...)` | `None`, `None`, `None` | `None`, `None`, `None` | `1`, `1`, `"merge"` | `small_bin_policy=None` |
| `ODPS_PROJECT`, `ODPS_ENDPOINT` for `ODPSRunner` | Optional, with built-in defaults | Optional, with built-in defaults | Required (`KeyError` if unset) | Set both variables |

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

## Fixes Without a Legacy Switch

These changes correct results that were wrong; no setting brings the old numbers back.

| Area | What changes |
|---|---|
| Equal-frequency binning with `include_missing=True` (`WOE_Master`, `get_woe_table`, `WOETransformer`, the non-monotone `woe_engine`, unweighted Gains and cross-risk tables) | Missing values get their own bin instead of sharing the lowest bin with real values; the default fill value `-1.797e308` no longer overflows to `-inf` |
| `RejectInferencePipeline` | Rows without a label are no longer trained as goods; `ri_score_direction="high_good"` with a pre-score trained by the pipeline raises instead of inverting the reject labels; the random OOT is held out of the pre-score, so OOT labels no longer reach the RI models through the inferred reject labels; an external reference is rescored with the pipeline's pre-score |
| `ScoreComparisonPipeline`, `model_perf_compare` | Scores are compared on the same rows (`perf_common_rows=False` evaluates each score on its own rows, which reproduces 0.9.0 with one comparison score); numeric group columns are evaluated; the weighted Gains table no longer counts unlabelled rows as goods or ranks missing scores; a group with exactly `min_data_size` rows is kept |
| Everything listed in the v0.9.0 notes, sections 2, 4 and 5 | See the "Results that change" boxes there |
