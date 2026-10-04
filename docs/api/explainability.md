# Modeling_Tool.Explainability

Model explainability layer. `ModelExplainer` is one entry point for SHAP attribution, Owen value grouped attribution,
PDP, ICE, ALE, and LIME. It accepts a `GradientBoostingModel`, an `LRMaster`, or a fitted estimator. User guide:
[Model Explainability](../guides/explainability.md).

!!! note "Optional dependencies"

    `shap` and `lime` load lazily, only when a SHAP, Owen, or LIME method runs, so `import Modeling_Tool` does not import
    them. PDP, ICE, and ALE need neither. Install both with:

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

## Model Explainer: `Model_Explainer`

::: Modeling_Tool.Explainability.Model_Explainer

## Coalition Structure for Owen Values: `Coalition_Structure`

::: Modeling_Tool.Explainability.Coalition_Structure
