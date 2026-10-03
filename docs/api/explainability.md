# Modeling_Tool.Explainability

Model explainability layer — a unified explainer built on [SHAP](https://shap.readthedocs.io/), supporting LightGBM / XGBoost / logistic regression, and any estimator with `predict_proba`.

!!! note "Optional dependency"
    This module depends on `shap` and uses **lazy loading** (it is imported only when an explanation is actually computed), so `import Modeling_Tool` does not pull in shap. Install it with:

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

## Model Explainer — `Model_Explainer`

::: Modeling_Tool.Explainability.Model_Explainer
