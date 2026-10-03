# Model Registry and Versioning

`save_model(...)` / `load_model(...)` can save an SMF model artifact together with metadata, recording the information needed for deployment: model version, feature list, WOE mapping table path, training-sample time window, and AUC / KS metrics.

This is not a database-style model registry; it first provides a **standard local artifact schema**, to make later extension toward a directory-based registry, model release, rollback, and approval workflows straightforward.

## 1. Save a Model with Metadata

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

`include_metadata=True` is the default, so what gets saved is the SMF artifact envelope, not a bare model object.

!!! tip "Recommended fields"

    For production models, save at least: `model_name`, `model_version`, `feature_cols`, `woe_mapping_path`, `train_window`, `metrics`.

## 2. Loading Still Returns the Model Object by Default

For compatibility with old code, `load_model(path)` returns only the model itself by default:

```python
from Modeling_Tool import load_model

model = load_model("./models/credit_risk_lgb_v20260627.pkl")
proba = model.predict_proba(score_df[woe_features])[:, 1]
```

If you need the metadata:

```python
model, metadata = load_model(
    "./models/credit_risk_lgb_v20260627.pkl",
    return_metadata=True,
)

print(metadata["model_version"])
print(metadata["metrics"]["oot"])
```

## 3. Read Only the Metadata

```python
from Modeling_Tool import load_model_metadata

metadata = load_model_metadata("./models/credit_risk_lgb_v20260627.pkl")
print(metadata["feature_cols"])
```

Old-format bare model files have no metadata, so `load_model_metadata(...)` returns an empty dict `{}`.

## 4. Artifact Structure

The internal structure of an SMF artifact is as follows:

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

Custom fields passed through `metadata=...` override or supplement the automatically generated fields.

## 5. Compatibility with Old Models

If an old file is a bare model saved directly with `joblib.dump(model, path)`:

```python
model = load_model("./models/legacy_model.pkl")
model, metadata = load_model("./models/legacy_model.pkl", return_metadata=True)
assert metadata == {}
```

If you really need to keep saving bare models:

```python
save_model(model, "./models/raw_model.pkl", include_metadata=False)
```

!!! warning "Difference between joblib.load and load_model"

    Reading a new artifact file directly with `joblib.load(path)` gives you a dict envelope; production code should consistently use `load_model(path)`, which recognizes both the new and old formats automatically and returns the model object by default.

## 6. Suggested Model Directory Layout

```text
models/
└── credit_risk_lgb/
    ├── 2026-06-27-v1.pkl
    ├── 2026-06-27-v1_woe_mapping.csv
    └── README.md
```

The current version only standardizes the artifact; if you need a fuller model registry, you can extend this schema with a `ModelRegistry`, model cards, approval status, and rollback pointers.
