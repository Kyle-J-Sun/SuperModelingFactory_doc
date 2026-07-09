# Pipeline GUI Schema

SMF 0.5.4+ provides a dependency-free schema layer for building a Pipeline configuration GUI.
The GUI itself should live outside the modeling package, but it can treat SMF as the source of truth for:

- the 7 public high-level Pipeline cards
- each Pipeline's Config dataclass fields
- field labels, descriptions, widgets, options, ranges, groups and dependencies
- run signature metadata
- result attributes
- YAML-safe config export and import
- lightweight config-only validation

## Public API

```python
from Modeling_Tool import (
    FieldMeta,
    PIPELINE_REGISTRY,
    get_pipeline_registry,
    get_pipeline_registry_schema,
    extract_pipeline_schema,
    extract_config_schema,
    extract_schema,
    config_to_dict,
    config_from_dict,
    config_to_yaml,
    config_from_yaml,
    validate_pipeline_config,
    generate_pipeline_code,
)
```

The same names are also exported from `Modeling_Tool.Pipeline`.

## Registered Pipelines

`get_pipeline_registry_schema()` returns a JSON/YAML-friendly registry without class objects:

```python
registry = get_pipeline_registry_schema()

for key, item in registry.items():
    print(key, item["display_name"], item["run_method"])
```

The registry currently contains:

| key | Pipeline | Config | `run_requires_data` |
|---|---|---|---|
| `credit_model` | `CreditModelPipeline` | `CreditModelPipelineConfig` | `True` |
| `feature_validation` | `FeatureValidationPipeline` | `FeatureValidationPipelineConfig` | `True` |
| `reject_inference` | `RejectInferencePipeline` | `RejectInferencePipelineConfig` | `True` |
| `score_comparison` | `ScoreComparisonPipeline` | `ScoreComparisonPipelineConfig` | `True` |
| `score_consistency_uat` | `ScoreConsistencyUATPipeline` | `ScoreConsistencyUATPipelineConfig` | `False` |
| `sample_analysis` | `SampleAnalysisPipeline` | `SampleAnalysisPipelineConfig` | `True` |
| `mock_sample` | `MockSamplePipeline` | `MockSamplePipelineConfig` | `False` |

`run_requires_data=False` means the generated code should not use `data=your_dataframe`.
For UAT, users may still call `run(offline_data=..., online_data=...)` manually in code mode.

## Extract One Schema

```python
schema = extract_pipeline_schema("feature_validation")

print(schema["display_name"])
print(schema["config_class_name"])
print(schema["run_method"])

for field in schema["fields"]:
    print(field["name"], field["label"], field["widget"], field["default"])
```

Each field item includes:

| Field | Meaning |
|---|---|
| `name` | Config dataclass field name. |
| `type` | Resolved type hint as text. |
| `default` | JSON/YAML-safe default value when possible. |
| `label` | Human-readable label. |
| `description` | Business or technical explanation. |
| `widget` | Suggested UI widget: `text`, `number`, `select`, `multiselect`, `toggle`, `slider`, `textarea`, `json`, `hidden`. |
| `options` | Allowed values for select/multiselect fields. |
| `min_val` / `max_val` / `step` | Numeric control hints. |
| `required` | Whether the GUI should mark the field as required. |
| `group` | Suggested form section. |
| `depends_on` | Conditional display rule. |
| `yaml_serializable` | Whether this field is safe for YAML export by default. |
| `gui_editable` | Whether the GUI should show it as an editable field. |
| `advanced` / `expert_only` | Suggested progressive disclosure flags. |
| `nested_fields` | Sub-form hints for dict-like fields such as `feature_selection`, `woe_params`, `corr_params`. |

## Extract All Schemas

```python
all_schemas = extract_pipeline_schema()
credit_schema = all_schemas["pipelines"]["credit_model"]
```

`extract_schema("credit_model")` is a compatibility helper that returns only the `fields` list.

## YAML Export

```python
from Modeling_Tool.Pipeline import MockSamplePipelineConfig
from Modeling_Tool import config_to_yaml

cfg = MockSamplePipelineConfig(
    n_samples=80000,
    applied_sample=1,
)

yaml_text = config_to_yaml(cfg)
print(yaml_text)
```

Output shape:

```yaml
pipeline: mock_sample
pipeline_class: MockSamplePipeline
config_class: MockSamplePipelineConfig
smf_version: 0.5.5
config:
  n_samples: 80000
  applied_sample: 1
```

`config_to_yaml()` requires `PyYAML` at runtime. SMF does not require Streamlit or any frontend dependency.

## YAML Import

```python
from Modeling_Tool import config_from_yaml

cfg = config_from_yaml(None, yaml_text)
```

When the first argument is `None`, SMF resolves the config class from the YAML `pipeline` or `config_class`.
You can also pass an explicit key:

```python
cfg = config_from_yaml("mock_sample", yaml_text)
```

## Dict Export

```python
from Modeling_Tool import config_to_dict

payload = config_to_dict(cfg)
```

By default, non-serializable fields are skipped. This is important for GUI/YAML safety.

Examples of skipped fields:

- `screening_artifact`
- `feature_validation_result`
- `extra_eval_datasets`
- `oot_data`
- `ri_approved_data`
- `ri_approved_func`
- `gains_add_func`
- `sqlrunner`
- `offline_data`
- `online_data`
- `psi_reference_data`

These fields remain usable in normal Python code, but should not be edited in a generic GUI form.

## Config Validation

```python
from Modeling_Tool import validate_pipeline_config

messages = validate_pipeline_config(
    "credit_model",
    {
        "target_col": "badflag",
        "warm_start_enabled": True,
        "warm_start_score_col": None,
    },
)

for msg in messages:
    print(msg)
```

The validator is intentionally lightweight. It catches common form mistakes before code generation, while each Pipeline's `run()` method remains the authoritative runtime validation.

## Code Generation

```python
from Modeling_Tool import generate_pipeline_code

code = generate_pipeline_code(
    "score_comparison",
    {
        "target_col": "badflag",
        "score_cols": ["score_old", "score_new"],
        "base_score": "score_old",
    },
)

print(code)
```

For `MockSamplePipeline` and `ScoreConsistencyUATPipeline`, generated code uses `result = pipeline.run()`.
For the other five pipelines, generated code uses `result = pipeline.run(data=your_dataframe)`.

## GUI Design Guidance

- Use `get_pipeline_registry_schema()` to render Pipeline cards.
- Use `extract_pipeline_schema(key)["fields"]` to render forms.
- Hide fields with `gui_editable=False` unless the GUI has an explicit code-only advanced mode.
- Do not put non-serializable fields in YAML exports.
- Render dict fields with `nested_fields` as sub-forms when possible; otherwise use a JSON textarea.
- Treat `FeatureScreeningArtifact` handoff as a Python-code advanced workflow, not a YAML field.
- Keep Streamlit or web UI dependencies outside the SMF main package.
