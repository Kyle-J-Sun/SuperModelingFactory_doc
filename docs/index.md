# SuperModelingFactory

SuperModelingFactory (SMF) is a Python toolkit for credit-risk scorecard development: sample design, WOE binning, feature
screening, model training, evaluation, explainability, online/offline consistency checks, and Excel reporting. You can call
each building block yourself, or hand a DataFrame to a one-click pipeline.

SMF installs three Python packages:

| Package | Role | What it provides |
|---|---|---|
| **[Modeling_Tool](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/Modeling_Tool)** | Modeling engine | Binning and WOE encoding, feature screening (PSI, IV, correlation), logistic regression and gradient-boosting models, evaluation, explainability, sample management, reject inference, UAT checks, one-click pipelines |
| **[ExcelMaster](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/ExcelMaster)** | Excel engine | A cursor-based writer for formatted workbooks: tables, images, charts, and conditional formatting |
| **[Report](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/Report)** | Report templates | Lay out modeling artifacts (performance CSVs, plot images, variable importances) as sheets of an Excel workbook |

New to SMF? [Install it](installation.md), then run the [Quickstart](quickstart.md) on synthetic data.

## What SMF Helps You Do

- Split a modeling sample, balance classes, infer labels for rejected applicants, and correct distribution shift.
- Bin variables, encode them with WOE, and screen features by PSI, IV/KS, and correlation.
- Train logistic regression, LightGBM, XGBoost, or CatBoost models, with optional sample weights
  (`weight_col` / `sample_weight`).
- Evaluate models with KS, AUC, lift, Gains tables, and champion/challenger comparisons.
- Explain models with SHAP, Owen value, PDP, ICE, ALE, and LIME.
- Check that online and offline scores and features agree before a model goes live (UAT).
- Write formatted Excel reports, or run the whole workflow with the one-click pipelines.

## Quick Overview

The four tabs continue from one another, so run them in order in one Python session. The first tab builds synthetic data;
the [Quickstart](quickstart.md) walks through the same flow with explanations.

=== "1. Data and split"

    ```python
    import numpy as np
    import pandas as pd
    from Modeling_Tool import SampleSplitter

    rng = np.random.default_rng(42)
    n = 6000
    data = pd.DataFrame({
        "age": rng.normal(35, 8, n).clip(18, 70),
        "income": rng.lognormal(10, 0.4, n),
        "score_b": rng.normal(600, 60, n),
        "utilization": rng.uniform(0, 1, n),
        "n_overdue": rng.poisson(0.3, n),
    })
    logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"] - 0.8 * data["utilization"]
    data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    features = ["age", "income", "score_b", "utilization", "n_overdue"]

    splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
    train_df, test_df = splitter.split_df(data, target="bad_flag")
    ```

=== "2. WOE encoding"

    ```python
    from Modeling_Tool import WOE_Master

    woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")  # numeric features
    woe.fit(nbins=10, equal_freq=True)
    train_woe = woe.transform(train_df)   # adds one `<feature>_woe` column per feature
    test_woe = woe.transform(test_df)
    woe_features = [f"{f}_woe" for f in features]
    ```

=== "3. Model training"

    ```python
    from Modeling_Tool import GradientBoostingModel

    model = GradientBoostingModel("lgb", {
        "n_estimators": 200, "learning_rate": 0.05,
        "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1,
    })
    model.fit(train_woe[woe_features], train_woe["bad_flag"],
              test_woe[woe_features], test_woe["bad_flag"])
    ```

=== "4. Evaluation and Excel report"

    ```python
    from Modeling_Tool import PerformanceEvaluator
    from ExcelMaster.ExcelMaster import ExcelMaster

    perf = (
        PerformanceEvaluator(tgt_name="bad_flag", model=model, feature_cols=woe_features)
        .add_dataset("train", train_woe)
        .add_dataset("test", test_woe)
        .evaluate(display=False)          # display=True needs IPython (notebooks)
    )
    print(perf[["index", "KS", "AUC"]])

    em = ExcelMaster("model_report.xlsx", verbose=False)    # `verbose` is required
    ws = em.add_worksheet("Performance")
    em.write_dataframe(ws, perf, title="Model Performance", titleformat="BLUE_H2")
    em.close_workbook()
    ```

## Documentation Map

<div class="grid cards" markdown>

- :material-package-variant: **[Installation](installation.md)**

    Requirements, optional extras, and how to verify the install.

- :material-rocket-launch: **[Quickstart](quickstart.md)**

    A complete scorecard workflow on synthetic data in about five minutes.

- :material-graph: **[Architecture](architecture.md)**

    Package layout, import rules, and where each class lives.

- :material-pipe: **[End-to-End Pipelines](pipeline.md)**

    The manual modeling workflow step by step, and the [seven one-click pipelines](pipeline_one_click.md).

- :material-book-open-variant: **[User Guides](guides/index.md)**

    One guide per task: samples, WOE, feature screening, models, evaluation, explainability, UAT, Excel reports, ODPS.

- :material-api: **[API Reference](api/index.md)**

    Signatures and docstrings of every public subpackage, and the list of top-level names.

- :material-frequently-asked-questions: **[FAQ](faq.md)**

    Import errors, warnings, ODPS credentials, and sample-weight keywords.

- :material-history: **[ChangeLog](changelog/index.md)**

    One page per release, with behavior changes called out.

</div>

## Who It Is For

- **Credit-risk modelers** developing application, behavior, and collection scorecards.
- **Model validators and auditors** who need UAT consistency checks, PSI monitoring, and variable interpretability.
- **Data scientists** who want reusable binning, WOE, backward elimination, and reject-inference code.
- **Modeling-platform developers** who build on SMF as the underlying library.

## Version and License

- **Version**: 0.9.0
- **Author**: Jingkai Sun
- **License**: [Business Source License 1.1](https://github.com/Kyle-J-Sun/SuperModelingFactory/blob/main/LICENSE).
  Personal study, academic research, internal evaluation, prototyping, and teaching are allowed. **Production use**, such
  as deploying SMF inside a credit-risk, lending, scoring, or other revenue-generating pipeline, requires a commercial
  license from the author. The license converts to Apache 2.0 on 2030-06-24.
