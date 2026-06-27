# 模型注册与版本管理

`save_model(...)` / `load_model(...)` 支持保存带 metadata 的 SMF model artifact，用于记录模型版本、特征列表、WOE 映射表路径、训练样本时间窗以及 AUC / KS 等上线所需信息。

这个功能不是数据库式模型注册中心；它先提供一个**本地标准 artifact schema**，方便后续扩展到目录式 registry、模型发布、回滚和审批流。

## 1. 保存带 metadata 的模型

```python
from Modeling_Tool import save_model

save_model(
    model=gbm._model.model,
    filename="./models/credit_risk_lgb_v20260627.pkl",
    model_name="credit_risk_lgb",
    model_version="2026-06-27-v1",
    feature_cols=woe_features,
    woe_mapping_path="./output/woe_mapping.csv",
    train_window={
        "start": "2025-01",
        "end": "2025-06",
        "oot": "2025-07",
    },
    metrics={
        "ins": {"auc": 0.742, "ks": 0.356},
        "oos": {"auc": 0.731, "ks": 0.341},
        "oot": {"auc": 0.718, "ks": 0.326},
    },
    metadata={
        "owner": "risk_modeling",
        "comment": "first GBM model after WOE refresh",
    },
)
```

默认 `include_metadata=True`，因此保存的是 SMF artifact envelope，而不是裸模型对象。

!!! tip "推荐字段"

    生产模型建议至少保存：`model_name`、`model_version`、`feature_cols`、`woe_mapping_path`、`train_window`、`metrics`。

## 2. 默认加载仍返回模型对象

为了兼容旧代码，`load_model(path)` 默认只返回模型本体：

```python
from Modeling_Tool import load_model

model = load_model("./models/credit_risk_lgb_v20260627.pkl")
proba = model.predict_proba(score_df[woe_features])[:, 1]
```

如果需要 metadata：

```python
model, metadata = load_model(
    "./models/credit_risk_lgb_v20260627.pkl",
    return_metadata=True,
)

print(metadata["model_version"])
print(metadata["metrics"]["oot"])
```

## 3. 只读取 metadata

```python
from Modeling_Tool import load_model_metadata

metadata = load_model_metadata("./models/credit_risk_lgb_v20260627.pkl")
print(metadata["feature_cols"])
```

旧格式裸模型文件没有 metadata，`load_model_metadata(...)` 会返回空 dict `{}`。

## 4. Artifact 结构

SMF artifact 的内部结构如下：

```python
{
    "__smf_model_artifact__": True,
    "artifact_version": "1.0",
    "model": model,
    "metadata": {
        "smf_version": "0.1.3",
        "artifact_version": "1.0",
        "model_name": "credit_risk_lgb",
        "model_version": "2026-06-27-v1",
        "model_class": "LGBMClassifier",
        "model_module": "lightgbm.sklearn",
        "feature_cols": ["age_woe", "income_woe", "history_woe"],
        "woe_mapping_path": "./output/woe_mapping.csv",
        "train_window": {"start": "2025-01", "end": "2025-06", "oot": "2025-07"},
        "metrics": {
            "ins": {"auc": 0.742, "ks": 0.356},
            "oos": {"auc": 0.731, "ks": 0.341},
            "oot": {"auc": 0.718, "ks": 0.326},
        },
        "created_at": "2026-06-27T11:35:00+00:00",
        "python_version": "3.12.0",
        "platform": "...",
    },
}
```

`metadata=...` 中传入的自定义字段会覆盖或补充自动生成字段。

## 5. 兼容旧模型

旧文件如果是直接 `joblib.dump(model, path)` 保存的裸模型：

```python
model = load_model("./models/legacy_model.pkl")
model, metadata = load_model("./models/legacy_model.pkl", return_metadata=True)
assert metadata == {}
```

如果确实需要继续保存裸模型：

```python
save_model(model, "./models/raw_model.pkl", include_metadata=False)
```

!!! warning "joblib.load 与 load_model 的区别"

    新 artifact 文件用 `joblib.load(path)` 直接读取时会得到一个 dict envelope；推荐生产代码统一使用 `load_model(path)`，它会自动识别新旧格式并默认返回模型对象。

## 6. 建议的模型目录结构

```text
models/
└── credit_risk_lgb/
    ├── 2026-06-27-v1.pkl
    ├── 2026-06-27-v1_woe_mapping.csv
    └── README.md
```

当前版本只负责 artifact 标准化；如果需要更完整的模型注册中心，可在此 schema 基础上继续扩展 `ModelRegistry`、模型卡、审批状态和回滚指针。
