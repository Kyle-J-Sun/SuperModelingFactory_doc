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
