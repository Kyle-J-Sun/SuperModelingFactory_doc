# SuperModelingFactory

> **An end-to-end Python modeling toolchain for credit scorecard development**

SuperModelingFactory brings together three capabilities needed across the full credit-risk modeling workflow:

| Sub-project | Role | Core capabilities |
|--------|---------|---------|
| **[Modeling_Tool](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/Modeling_Tool)** | Modeling engine | Data binning, WOE encoding, feature analysis, model training and evaluation, sample management |
| **[ExcelMaster](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/ExcelMaster)** | Reporting engine | Programmatic Excel workbook generation with charts, conditional formatting, and cursor-based streaming writes |
| **[Report](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/Report)** | Report templates | Model performance reports, bulk WOE plot export, multi-model comparison reports |

---

## What It Helps You Do

!!! tip "Typical scenarios"

    - Start from a behavior-scoring sample and **produce a scorecard training sample in 5 minutes**
    - Use **WOE / IV / PSI** for feature screening and stability monitoring
    - Train **Logistic Regression / LightGBM / XGBoost / CatBoost** models and automatically produce Gains / ROC / KS reports
    - Support **sample-weighted** training and evaluation (`weight_col` / `sample_weight`, for balance weighting, oversampling correction, and similar scenarios)
    - Handle **reject inference** and **distribution shift**
    - Export formatted modeling reports in one click with **ExcelMaster**
    - Check online/offline score consistency with the **UAT module**

---

## Quick Overview

=== "Sample splitting"

    ```python
    from Modeling_Tool import SampleSplitter
    splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
    train_df, test_df = splitter.split_df(data, target="bad_flag")
    ```

=== "WOE encoding"

    ```python
    from Modeling_Tool import WOE_Master
    woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")  # numeric features
    woe.fit(nbins=10, equal_freq=True)
    train_woe = woe.transform(train_df)   # adds `<feature>_woe` columns
    test_woe  = woe.transform(test_df)
    ```

=== "Model training"

    ```python
    from Modeling_Tool import GradientBoostingModel
    woe_features = [f"{f}_woe" for f in features]
    model = GradientBoostingModel("lgb", {
        "n_estimators": 200, "learning_rate": 0.05,
        "early_stopping_rounds": 20, "eval_metric": "auc",
    })
    model.fit(train_woe[woe_features], train_woe["bad_flag"],
              test_woe[woe_features],  test_woe["bad_flag"])
    ```

=== "Excel report"

    ```python
    from ExcelMaster.ExcelMaster import ExcelMaster
    em = ExcelMaster("model_report.xlsx", verbose=False)
    ws = em.add_worksheet("Performance")
    em.write_dataframe(ws, perf, title="Model Performance", titleformat="BLUE_H2")
    em.insert_image(ws, "roc_curve.png", figScale=(0.8, 0.8))   # scale factors
    em.close_workbook()
    ```

---

## Documentation Map

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](quickstart.md)**

    Run your first scorecard training pipeline in 5 minutes.

- :material-package-variant: **[Installation](installation.md)**

    Core dependencies, optional dependencies, MaxCompute access.

- :material-graph: **[Architecture](architecture.md)**

    Module dependency graph, design principles, naming conventions.

- :material-pipe: **[End-to-End Pipelines](pipeline.md)**

    The complete modeling workflow, from sample splitting to Excel report.

- :material-book-open-variant: **[User Guides](guides/index.md)**

    Organized by scenario: sample / WOE / feature / model / evaluation / UAT / report.

- :material-api: **[API Reference](api/index.md)**

    Detailed descriptions of every public class, method, and function.

</div>

---

## Who It's For

- **Credit-risk modelers**: develop application (A-card), behavior (B-card), and collection (C-card) scorecards and anti-fraud models
- **Model validation / audit**: UAT consistency, PSI monitoring, variable interpretability
- **Data scientists**: reuse modules for binning / WOE / backward elimination / reject inference
- **Modeling platform developers**: build on SuperModelingFactory as the underlying library

---

## Version

- **Version**: 0.8.2
- **Author**: Jingkai Sun
- **License**: [Business Source License 1.1](https://github.com/Kyle-J-Sun/SuperModelingFactory/blob/main/LICENSE) (converts to Apache 2.0 after 2030-06-24; contact the author for a commercial-use license)

---

## Next Steps

👉 [Quickstart](quickstart.md) → [Installation](installation.md) → [Architecture](architecture.md)
