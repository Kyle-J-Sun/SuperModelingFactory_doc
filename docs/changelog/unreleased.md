# Unreleased

Changes made after 0.9.0 that ship with the next release. The version number is not decided yet.

## 1. `RejectInferencePipeline`: Rows Without a Label Are No Longer Trained as Goods

An end-to-end audit of `RejectInferencePipeline` ran it on synthetic approved and rejected applications whose true label is known for every row. Two defects made the models learn labels that do not exist. Each has a regression test in `test_reject_inference_audit.py` that fails without the fix.

| Defect | Fix |
|---|---|
| Approved rows whose target is missing (not yet performed) were skipped by the pre-score but kept by the RI models and the `no_ri_benchmark` model, and the fit turned the missing target into 0: every such row was trained as a good. The random OOT and the validation sample could draw them too, and `ri_summary['prescore_AUC']` came out NaN without a warning. In the audit, 20% of the approved rows were unlabelled with a true bad rate of 10.5%, and 664 of them went into every model as goods | Model training, the validation sample and the random OOT use only rows with an observed target. The rows stay in `ri_datasets`; `ri_summary` counts them in `N_approved_unlabelled`, `ri_model_perf` counts the rows left out of each model in `train_unlabelled_n`, and a `UserWarning` names them. `prescore_AUC` is computed on the approved rows with an observed target |
| `hard_cutoff` gives a rejected row with a missing score a missing label on purpose (since 0.4.2, so that it is not invented as a good), and the same conversion in the fit turned it back into a good. In the audit, 300 such rejects with a true bad rate of 42% were trained as goods | These rejects stay in `ri_datasets['hard_cutoff']` with a missing label and are left out of the model; `ri_summary['N_rejected_unlabelled']` and `ri_model_perf['train_unlabelled_n']` count them |

With `train_ri_models=True`, a run in which no approved row has an observed target (possible with an external `ri_approved_data`) now raises `ValueError` instead of training every model on approved rows labelled good.

!!! note "Results that change"
    Only runs with missing targets change. When every approved row has a target and every reject gets a label, `ri_summary`, `ri_model_perf`, the OOT and the validation sample are identical to 0.9.0 (checked with the default random OOT, with `split_col` and with an external `oot_data`); the only difference is the new columns `N_approved_unlabelled`, `N_rejected_unlabelled` and `train_unlabelled_n`, which are 0.

## 2. `RejectInferencePipeline`: One Score Scale, and an OOT the Pre-score Never Saw

The same audit found three ways in which the pre-score disagreed with the rest of the pipeline. Each has regression tests in `test_reject_inference_audit.py` that fail without the fix.

| Defect | Fix |
|---|---|
| The pre-score the Pipeline trains is the probability of bad, but `ri_score_direction="high_good"` was accepted with it and read it as a good score: `hard_cutoff` and `fuzzy_augment` labelled the riskiest rejects good (hard-cutoff reject bad rate 0.23 instead of 0.99). The only hint was a `prescore_AUC` of 0.22 | `"high_good"` with a pre-score trained by the Pipeline (`train_prescore=True`, or `score_col` absent) raises `ValueError`, and `validate_pipeline_config` reports it. Your own high-good score with `train_prescore=False` works as before |
| An external `ri_approved_data` that carried its own `score_col` kept it when the Pipeline trained a new pre-score, so the rules were fitted on one score scale and applied to the rejects' other one (reject bad rate 0.09 for `hard_cutoff` and 0.21 for `parceling`, for a true 0.41) | When the Pipeline trains the pre-score, the reference is scored with it; an existing `score_col` is replaced with a `UserWarning`. With `train_prescore=False` the reference keeps its own scores |
| With neither `oot_data` nor OOT rows from `split_col`, the random OOT was drawn from approved rows the pre-score had been trained on. Their labels reached the RI models through the inferred reject labels, which inflated the RI models' OOT AUC against `no_ri_benchmark` and skewed `best_method`: on pure-noise features the `hard_cutoff` model scored an OOT AUC of 0.54 (0.51 with the OOT held out) | The random OOT is drawn before the pre-score is trained and is left out of its training data. It keeps the same rows as before. With `ri_approved_scope="output_subset"` it is drawn from all labelled approved rows and only the drawn rows inside the subset are used (`ValueError` if none falls inside) |

!!! note "Results that change"
    - A configuration with `ri_score_direction="high_good"` and a pre-score trained by the Pipeline now raises. Its results were inverted; use `"high_bad"`.
    - A run with an external `ri_approved_data` that carries its own `score_col`, while the Pipeline trains the pre-score, now infers the reject labels from the new pre-score.
    - With the default random OOT, the pre-score is trained on about `1 - oot_frac` of the labelled approved rows instead of all of them, so `prescore_model`, the pre-score column, the inferred labels, `ri_model_perf` and `best_method` change. Runs with `oot_data` or with OOT rows from `split_col`, and runs with `train_prescore=False` and your own scores, are not affected by this change (checked: identical results).

## 3. `ScoreComparisonPipeline`: Scores Compared on the Same Rows, Groups by Value, Weighted Gains Fixed

An end-to-end audit of `ScoreComparisonPipeline` ran it on synthetic scores whose true performance is known. Seven defects distorted the comparison or lost data without a word. Each has a regression test in `test_score_comparison_audit.py` that fails without the fix. Items 1, 2, 5, 6 and 7 are fixed in the shared evaluation code, so `Model_Evaluation_Tool.model_perf_compare` and the Gains tables behave the same way outside the pipeline.

| Defect | Fix |
|---|---|
| `model_perf_compare` evaluated each comparison score on the rows where *every* comparison score was above 0: one comparison score that was always 0 removed every other comparison score from `global_perf` and `group_perf`, and a score covering 10% of the rows cut the others to 800 of 8,000 rows | A score without any valid row is left out of the comparison with a `UserWarning` instead of emptying the others |
| The base score kept all its rows while the comparison scores used the shared ones, so `global_perf` compared AUCs measured on different populations. With a comparison score covering 60% of the rows both AUCs read about 0.71, while on the same rows the base score has 0.637 and the new score 0.713 | Every score, the base score included, is evaluated on the rows where all the scores are valid (`sync_data_size=True`, the default). The new column `N_OWN` gives each score's own number of valid rows. The pipeline's new `perf_common_rows=False` evaluates each score on its own rows instead |
| `multi_group_wrapper` selected each group with the query ``col == 'value'``, so a numeric group column (the default `apply_month` holding `202401`) gave an empty `group_perf` table | Groups are selected by value; missing values form no group |
| A group value containing a quote (`kid's app`) stopped the run with `SyntaxError` | Fixed by the same change |
| The weighted Gains table counted rows with a missing target as goods: the top bin's bad rate read 0.374 for a true 0.417, and the `AUC` came out NaN | Rows with a missing target count in `N` only; `PERF_CNT`, `N_BAD`, `N_GOOD`, `AVG_BAD`, `LIFT` and `AUC` use the labelled rows, as in the unweighted table |
| The weighted Gains table ranked rows with a missing score into the last bins whatever `include_missing` said (the top bin held only missing scores) | Missing scores never enter the bins: they are left out, or reported in a `Missing` row with `include_missing=True` |
| With `include_missing=True` and equal-frequency bins, the unweighted Gains table (`get_gains_table`, `GainsTableCalculator`, the pipeline's `gains`) and `cross_risk` filled missing scores with `fillna` and counted them in the quantiles: in the test the 800 missing rows shared the lowest bin with 800 real scores and moved every edge, unless they made up more than one bin's share. Equal-width bins already gave them a bin of their own | Missing scores (and scores already holding `fillna`) form their own bin `(-inf, fillna]`; the other bins are the equal-frequency bins of the real scores, the same as with `include_missing=False`. Grouped tables (`grp_name`) decide this once over all groups, so every group keeps the same bin numbers and labels. Complete scores keep their bins |

The group tables collect the notices about scores without valid rows into one `UserWarning` that names the affected group dimensions.

!!! note "Results that change"
    - `global_perf` and `group_perf` (and `model_perf_compare` with the default `sync_data_size=True`) evaluate the base score on the common rows: whenever a comparison score is missing, zero or negative where the base score is valid, the base score's `N`, `AUC`, `KS` and lift change. With full coverage nothing changes except the new `N_OWN` column. `sync_data_size=False` now means "each score on its own valid rows" for every score (comparison scores no longer need a valid base score).
    - Group tables of numeric group columns are filled, and their rows come in order of appearance instead of an arbitrary order.
    - Unweighted Gains and cross-risk tables built with `include_missing=True` (the default of `get_gains_table`, `GainsTableCalculator` and `Model_Evaluation_Tool`) change when a score has missing values: the missing rows get bin 0 and the equal-frequency edges of the real scores move. WOE binning follows the same rule since the change in section 4.
    - The weighted Gains table (pipeline `gains` with `weight_col`, `GainsTableCalculator` and `get_gains_table` with `weight_col`, and the weighted `PerformanceEvaluator` summary's `LIFT` and `IV`) changes only when the data has a missing target or a missing score.

## 4. Missing Values Get Their Own Bin on Every Binning Path

The equal-frequency branch of `quick_binning`, the binning behind `WOE_Master`, `get_woe_table`, `WOETransformer`, the `Binning` class and the evaluation tables, filled missing values with the fill value and counted them in its quantiles, so with `include_missing=True` the missing rows shared the lowest bin with real values. The other branches already gave them a bin of their own, but only for a fill value such as `-999999`: `WOE_Master`'s default `missing_ref_value` is `-1.797e308`, which Python's `round` and pandas' interval labels overflow to `-inf`, so the missing bin merged with the lowest bin on every path, and chi-square binning raised `ValueError: Bin edges must be unique`. Regression tests: `test_woe_missing_bin.py`.

| Defect | Fix |
|---|---|
| Equal-frequency bins counted the filled missing values in the quantiles: in the test the 800 missing rows shared the lowest bin with 800 real values, and every edge moved | The quantiles use the real values only and the fill value becomes a bin edge, as in the equal-width branch, when some rows are missing (or hold the fill value). Complete data keeps its bins |
| The default `missing_ref_value` of `WOE_Master` (`-1.797e308`) was rounded to `-inf`, so the missing bin vanished on the equal-width, tree and chi-square paths too, and chi-square binning raised | An edge that the rounding or pandas' interval labels would overflow into the same infinity as another edge is kept as it is (bins are numbered instead of labelled by pandas in that case); chi-square binning uses the exact cut points when pandas' labels merge two edges, and its special-value step uses the next representable value near the float limits. Edges at the float limits that merge with nothing are labelled `±inf` as before |

`transform` scores the missing rows with the WOE of their bin on every path; previously they took the WOE of the lowest bin.

!!! note "Results that change"
    - A WOE fit (`WOE_Master`, also the pipelines' equal-frequency `woe_engine` when it is not `"monotone"`, `get_woe_table`, `WOETransformer`) with `include_missing=True` on a feature with missing values gets a missing bin, so its bins, WOE values, IV and the transformed feature change. Engines fitted earlier keep scoring with their stored bins until they are refit. `MonotoneWOEBinner`, the default engine of the pipelines, is not affected.
    - With the default `missing_ref_value`, equal-width and tree WOE tables of complete data now number their bins from 1 and show `-1.7976931348623157e+308` as the lower end of the first range, as the tables built with `missing_ref_value=-999999` always did; the WOE values and the scores do not change.
    - `fit(chi2_config=...)` with the default `missing_ref_value` no longer raises.

## 5. `ScoreComparisonPipeline`: Weighted Gains Keep Their Columns, `min_data_size` Is a Minimum, Safe File Names

The remaining four defects of the audit in section 3. Each has a regression test in `test_score_comparison_audit.py` that fails without the fix.

| Defect | Fix |
|---|---|
| With `weight_col`, the `gains` table silently lost the `gains_add_func` / `custom_metric_cols` columns and each score's `Grand Summary` row: the weighted Gains table ignored `add_func` and `withSummary` (also in `get_gains_table`, `GainsTableCalculator.calculate` and `Model_Evaluation_Tool.get_gains_summary`) | The weighted table applies `add_func` to the rows of every row of the table (bins, `special:<value>` rows and the `Missing` row) and adds the `Grand Summary` row with `withSummary=True`; its counts and rates cover every row of the table. The pipeline's default `<col>_mean` columns are weighted means when `weight_col` is set. The other columns are unchanged |
| Single-column groups were kept only with *more* than `min_data_size` (or a spec's `min_size`) rows, crossed groups with at least that many: a group value with exactly 50 rows was missing from `group_perf['channel']` but present in the crossed table | A group value with exactly `min_data_size` rows is evaluated in both. `Model_Evaluation_Tool.multi_group_wrapper` itself keeps its documented "more than `min_subset_size`" rule |
| `cross_binning_numeric=True` or `False`, allowed by the annotation and shown in the docs, stopped the run with `TypeError: 'bool' object is not subscriptable` | `cross_risk` takes a single bool for both columns (and None for `[True, True]`); a list, tuple or array of another length than two raises `ValueError` instead of `IndexError` |
| A `group_specs` name containing `/` wrote `report/step2_by_chan/month.csv` into a subfolder, and a cross metric name with `/` did the same for `step4_` files | Characters that a file name cannot hold (`/`, `\`, `:`, `*`, `?`, quotes, angle brackets, the vertical bar and control characters) become `_` in the CSV file names, and a name that would repeat another one (ignoring case) gets a `_2` suffix. The keys of `group_perf` and `cross_results` are unchanged |

!!! note "Results that change"
    - Weighted `gains` (pipeline with `weight_col`, and `get_gains_table`, `GainsTableCalculator` and `get_gains_summary` with weights and `add_func` or `withSummary=True`) gain the custom columns and the `Grand Summary` row; the columns that were there do not change.
    - `group_perf` tables of single columns gain the group values that have exactly `min_data_size` rows (or the spec's `min_size`).
    - CSV files of names with unsafe characters, or of names that differ only in case, are written under new names.
