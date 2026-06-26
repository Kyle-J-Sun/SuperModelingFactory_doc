# Modeling_Tool.Explainability

模型解释层 — 基于 [SHAP](https://shap.readthedocs.io/) 的统一解释器，支持 LightGBM / XGBoost / 逻辑回归，以及任意带 `predict_proba` 的估计器。

!!! note "可选依赖"
    本模块依赖 `shap`，采用**懒加载**（只有真正计算解释时才导入），因此 `import Modeling_Tool` 不会拉起 shap。安装：

    ```bash
    pip install 'supermodelingfactory[explain]'
    ```

## 模型解释器 — `Model_Explainer`

::: Modeling_Tool.Explainability.Model_Explainer
