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

An end-to-end audit of `ScoreComparisonPipeline` ran it on synthetic scores whose true performance is known. Six defects distorted the comparison or lost data without a word. Each has a regression test in `test_score_comparison_audit.py` that fails without the fix. Items 1, 2, 5 and 6 are fixed in the shared evaluation code, so `Model_Evaluation_Tool.model_perf_compare` and the weighted Gains table behave the same way outside the pipeline.

| Defect | Fix |
|---|---|
| `model_perf_compare` evaluated each comparison score on the rows where *every* comparison score was above 0: one comparison score that was always 0 removed every other comparison score from `global_perf` and `group_perf`, and a score covering 10% of the rows cut the others to 800 of 8,000 rows | A score without any valid row is left out of the comparison with a `UserWarning` instead of emptying the others |
| The base score kept all its rows while the comparison scores used the shared ones, so `global_perf` compared AUCs measured on different populations. With a comparison score covering 60% of the rows both AUCs read about 0.71, while on the same rows the base score has 0.637 and the new score 0.713 | Every score, the base score included, is evaluated on the rows where all the scores are valid (`sync_data_size=True`, the default). The new column `N_OWN` gives each score's own number of valid rows. The pipeline's new `perf_common_rows=False` evaluates each score on its own rows instead |
| `multi_group_wrapper` selected each group with the query ``col == 'value'``, so a numeric group column (the default `apply_month` holding `202401`) gave an empty `group_perf` table | Groups are selected by value; missing values form no group |
| A group value containing a quote (`kid's app`) stopped the run with `SyntaxError` | Fixed by the same change |
| The weighted Gains table counted rows with a missing target as goods: the top bin's bad rate read 0.374 for a true 0.417, and the `AUC` came out NaN | Rows with a missing target count in `N` only; `PERF_CNT`, `N_BAD`, `N_GOOD`, `AVG_BAD`, `LIFT` and `AUC` use the labelled rows, as in the unweighted table |
| The weighted Gains table ranked rows with a missing score into the last bins whatever `include_missing` said (the top bin held only missing scores) | Missing scores never enter the bins: they are left out, or reported in a `Missing` row with `include_missing=True` |

The group tables collect the notices about scores without valid rows into one `UserWarning` that names the affected group dimensions.

!!! note "Results that change"
    - `global_perf` and `group_perf` (and `model_perf_compare` with the default `sync_data_size=True`) evaluate the base score on the common rows: whenever a comparison score is missing, zero or negative where the base score is valid, the base score's `N`, `AUC`, `KS` and lift change. With full coverage nothing changes except the new `N_OWN` column. `sync_data_size=False` now means "each score on its own valid rows" for every score (comparison scores no longer need a valid base score).
    - Group tables of numeric group columns are filled, and their rows come in order of appearance instead of an arbitrary order.
    - The weighted Gains table (pipeline `gains` with `weight_col`, `GainsTableCalculator` and `get_gains_table` with `weight_col`, and the weighted `PerformanceEvaluator` summary's `LIFT` and `IV`) changes only when the data has a missing target or a missing score.
