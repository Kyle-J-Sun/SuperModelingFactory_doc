# Pipeline GUI Schema

SMF describes its seven high-level Pipelines as plain Python data, so that a GUI can render a configuration form without
importing any frontend package. The schema layer, available since 0.5.4, gives you for every Pipeline:

- a registry entry: name, description, how to call `run`, and which result attributes to show;
- one entry per field of its `*Config` dataclass: type, default, suggested widget, options, numeric range, form section,
  dependency, and flags for what to hide or collapse;
- YAML-safe export and import of a config, a lightweight config-only validator, and a code generator.

The GUI itself should live outside the modeling package. SMF is the source of truth for what can be configured; it does not
depend on Streamlit or any other frontend.

!!! warning "Display text is in Chinese in 0.8.2"

    The human-readable strings are Chinese: `display_name`, `description`, `use_case`, and `audience` of the registry
    entries, `label`, `description`, and `group` of the fields, and most messages from `validate_pipeline_config`. Field
    names, option values, types, and defaults are language-neutral. If your GUI needs another language, key your own
    translations on the registry `key` and the field `name`.

    Curated text is also incomplete. A field without a curated label gets its title-cased name as `label` (16 of the 260
    fields in 0.8.2, for example `Score Cols`), and for most fields (201 of 260) `description` just repeats `label`. Do not
    rely on `description` as help text: the [Top-Level Pipelines](../pipeline_one_click.md) page describes the parameters of
    each Pipeline.

All blocks on this page run top to bottom in one Python session. The YAML helpers need PyYAML (`pip install pyyaml`), which
SMF does not install.

## Public API

```python
from Modeling_Tool import (
    FieldMeta,
    PipelineRegistryEntry,
    PIPELINE_REGISTRY,
    get_pipeline_registry,
    get_pipeline_registry_schema,
    get_config_field_meta,
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

| Name | Signature | Returns |
|---|---|---|
| `PIPELINE_REGISTRY` | | `dict` of registry key to `PipelineRegistryEntry` |
| `get_pipeline_registry_schema` | `()` | `dict` of key to registry item, free of class objects |
| `get_pipeline_registry` | `(include_classes=True)` | The same items; by default each also holds the `pipeline_class`, `config_class`, and `result_class` objects |
| `extract_pipeline_schema` | `(pipeline_key=None)` | One Pipeline's schema; with `None`, `{"pipelines": {key: schema}}` for all |
| `extract_config_schema` | `(config_class_or_key)` | One Pipeline's schema (what `extract_pipeline_schema(key)` returns) |
| `extract_schema` | `(config_class_or_key)` | Only the `fields` list of that schema |
| `get_config_field_meta` | `(config_class_or_key)` | `dict` of field name to `FieldMeta` |
| `config_to_dict` | `(config, *, include_non_serializable=False, exclude_none=False)` | Plain `dict` |
| `config_from_dict` | `(config_class_or_key, payload, *, strict=True)` | A Config instance |
| `config_to_yaml` | `(config, *, pipeline_key=None, include_non_serializable=False)` | YAML text |
| `config_from_yaml` | `(config_class_or_key, yaml_text, *, strict=True)` | A Config instance |
| `validate_pipeline_config` | `(pipeline_key, values)` | `list[str]` of messages |
| `generate_pipeline_code` | `(pipeline_key, values)` | Python source as text |

Wherever a function takes a Pipeline reference, you can pass the registry key (`"credit_model"`), the name of the Pipeline,
Config, or Result class as text (`"CreditModelPipelineConfig"`), one of those classes, or a Config instance. An unknown
reference raises `KeyError`.

## Registered Pipelines

`get_pipeline_registry_schema()` returns a JSON/YAML-friendly registry:

```python
registry = get_pipeline_registry_schema()

for key, item in registry.items():
    print(key, item["pipeline_class_name"], item["run_method"], item["run_requires_data"])
```

| Key | Pipeline | Config | `run_requires_data` |
|---|---|---|---|
| `credit_model` | `CreditModelPipeline` | `CreditModelPipelineConfig` | `True` |
| `feature_validation` | `FeatureValidationPipeline` | `FeatureValidationPipelineConfig` | `True` |
| `reject_inference` | `RejectInferencePipeline` | `RejectInferencePipelineConfig` | `True` |
| `score_comparison` | `ScoreComparisonPipeline` | `ScoreComparisonPipelineConfig` | `True` |
| `score_consistency_uat` | `ScoreConsistencyUATPipeline` | `ScoreConsistencyUATPipelineConfig` | `False` |
| `sample_analysis` | `SampleAnalysisPipeline` | `SampleAnalysisPipelineConfig` | `True` |
| `mock_sample` | `MockSamplePipeline` | `MockSamplePipelineConfig` | `False` |

Each registry item has these keys:

| Key | Meaning |
|---|---|
| `key` | The registry key |
| `display_name`, `description`, `use_case` | Card text; `audience` is a list of reader roles |
| `pipeline_class_name`, `config_class_name`, `result_class_name` | Class names |
| `module_path`, `import_path` | Where the classes are defined, and the package that exports them (`Modeling_Tool.Pipeline`); generated code imports from `import_path` |
| `run_requires_data` | `False` when `run()` takes no DataFrame |
| `run_method` | The call to show on the card. For `score_consistency_uat` it lists both forms, `run()` and `run(offline_data=..., online_data=...)`, joined by the Chinese word for "or" |
| `result_attrs` | Headline attributes of the result object. The result dataclasses have more fields than these |

`run_requires_data=False` means the generated code should not pass `data=your_dataframe`. `ScoreConsistencyUATPipeline.run`
still accepts `offline_data` and `online_data` when you call it from code.

## Extract One Schema

```python
schema = extract_pipeline_schema("feature_validation")

print(schema["config_class_name"])
print(schema["run_method"])

for field in schema["fields"][:5]:
    print(field["name"], field["widget"], field["default"])
```

A schema holds the registry item keys plus `fields`, a list in the order of the Config dataclass. Group the editable
fields into form sections like this:

```python
sections = {}
for field in schema["fields"]:
    if field["gui_editable"]:
        sections.setdefault(field["group"], []).append(field["name"])

for group, names in sections.items():
    print(group, len(names), names[:3])
```

Each field item has these keys:

| Key | Meaning |
|---|---|
| `name` | Config dataclass field name |
| `type` | The type hint as text (see the note below the table). Choose a control from `widget`, not from this text |
| `default` | The default as JSON/YAML-safe data (tuples and ranges become lists), or its `repr()` text when it cannot be converted |
| `has_default` | Whether the dataclass declares a default. It is `True` for every field in 0.8.2, so every Config can be built with no arguments |
| `label`, `description` | Display text (Chinese in 0.8.2) |
| `widget` | `text`, `number`, `select`, `multiselect`, `toggle`, `slider`, `textarea`, `json`, or `hidden` |
| `options` | Allowed values for `select` and `multiselect`, for example `["equal_freq", "monotone"]` for `woe_engine`; otherwise `None` |
| `min_val`, `max_val`, `step` | Numeric control hints, mostly for sliders; otherwise `None` |
| `required` | The form should ask for this field. It comes from a fixed list per Pipeline (`credit_model`: `target_col`) and does not mean "has no default" |
| `group` | Name of the suggested form section (Chinese in 0.8.2). Sections are inferred from the field name, so a few assignments are rough: `main_model_score_col` of `score_consistency_uat` lands in the model-training section |
| `depends_on` | Display condition `{field_name: value}`, or `None`. Only four fields have one: `warm_start_score_col` (`{"warm_start_enabled": True}`) and the three `monotone_refine_*_params` fields of `feature_validation` |
| `since_version` | Reserved: `None` for every field in 0.8.2 |
| `yaml_serializable` | `False` for the code-only fields listed under [Dict Export](#dict-export) |
| `gui_editable` | `False` for the same fields: do not render them |
| `advanced` | Hint to collapse the field. It is inferred from the name: names containing `model` and the `lr_`, `warm_start`, `backward`, `optuna`, and `explain` families, every `*_params` dictionary, and CSV batching. It can flag a field that is also `required` (`main_model_score_col`), and you should not hide that one |
| `expert_only` | Hint for fields most users should leave alone: `extra_eval_datasets`, `batch_corr_pair_chunk_size`, `pairwise_cross_agg_dict` |
| `placeholder` | Example input text for a few fields such as `feature_cols`; otherwise `None` |
| `is_dict_subkey`, `parent_field` | Set only on nested entries (below) |
| `nested_fields` | Sub-form hints for dictionary fields (below) |

`type` is the type hint as Python prints it, with the `typing.` prefix removed: plain classes appear as `<class 'bool'>`,
other hints as `str | None`, `list[str]`, or `Literal['skip', 'raise']`.

### Nested fields

`nested_fields` describes the keys of these dictionary fields: `split_config`, `feature_selection`, `woe_params`,
`monotone_woe_params`, `corr_params`, `psi_params`, `ivks_params`, `distribution_params`, `ri_method_params`, and
`submodel_pairs`. A nested entry has the same keys as a field item, except `name`, `type`, `default`, and `has_default`: the
dictionary key is stored in `label` (for `submodel_pairs`, the pattern `offline_col = online_col`), `parent_field` names the
owning field, and `is_dict_subkey` is `True`.

```python
credit = extract_pipeline_schema("credit_model")
split_config = next(f for f in credit["fields"] if f["name"] == "split_config")

print(split_config["default"])
for nested in split_config["nested_fields"]:
    print(nested["parent_field"], nested["label"], nested["widget"], nested["min_val"], nested["max_val"])
```

The nested hints are a starting point, not a full list of the accepted keys. For example, those of `monotone_woe_params` do
not include `eps`, `missing_woe`, `special_values`, the `sv_*` options, or `unseen_special_policy` of `MonotoneWOEBinner`,
and `corr_params.base_metric` offers `lift` although `CorrelationFilter` supports only `iv` and `ks` (`lift` raises
`KeyError`). Offer a JSON text area as a fallback.

## Extract All Schemas

```python
all_schemas = extract_pipeline_schema()
credit_schema = all_schemas["pipelines"]["credit_model"]

print(list(all_schemas["pipelines"]))
print(len(credit_schema["fields"]))
```

`extract_schema("credit_model")` is a compatibility helper that returns only the `fields` list.

### Field metadata objects

`get_config_field_meta` returns the same information as `FieldMeta` objects keyed by field name, without the `name`, `type`,
`default`, and `has_default` keys. `FieldMeta.to_dict()` converts one back to a dictionary.

```python
meta = get_config_field_meta("credit_model")

print(type(meta["target_col"]).__name__, meta["target_col"].widget, meta["target_col"].required)
print(sorted(meta["target_col"].to_dict()))
```

## YAML Export

```python
from Modeling_Tool.Pipeline import MockSamplePipelineConfig

cfg = MockSamplePipelineConfig(
    n_samples=80000,
    applied_sample=1,
)

yaml_text = config_to_yaml(cfg)
print(yaml_text)
```

The output (`smf_version` is the installed SMF version):

```yaml
pipeline: mock_sample
pipeline_class: MockSamplePipeline
config_class: MockSamplePipelineConfig
smf_version: 0.8.2
config:
  n_samples: 80000
  applied_sample: 1
  approve_rate: 0.25
  num_online_scores: 5
  y_flag_candidates:
  - 15
  - 30
  - 45
  num_features: 20
  min_num_feature_business_type: 5
  random_state: 42
  observation_timestamp: null
  application_months: 18
  write_csv: false
  output_path: output/mock_sample/mock_sample.csv
```

`config_to_yaml` writes **every** serializable field, defaults included, so the file is a complete snapshot of the config.
Keep `include_non_serializable` at its default for YAML: the option writes raw objects that `yaml.safe_dump` cannot always
represent. `config_to_yaml` needs PyYAML at call time and raises `ImportError` without it.

## YAML Import

```python
from Modeling_Tool import config_from_yaml

cfg = config_from_yaml(None, yaml_text)
```

When the first argument is `None`, SMF resolves the Config class from the `pipeline` or `config_class` entry of the YAML.
You can also pass a key:

```python
cfg = config_from_yaml("mock_sample", yaml_text)
```

By default (`strict=True`), a key that is not a field of the Config raises `KeyError`. With `strict=False`, unknown keys are
dropped. The round trip is exact except that tuples and ranges are written as lists, so `range(3000, 3020)`, the default of
`SampleAnalysisPipelineConfig.random_seeds`, comes back as a list.

## Dict Export

```python
from Modeling_Tool import config_to_dict

payload = config_to_dict(cfg)
print(payload["n_samples"], payload["applied_sample"])
```

By default, fields that cannot be serialized are skipped. This keeps GUI and YAML payloads safe. A field is skipped when the
schema marks it `yaml_serializable=False`, or when its value cannot be converted. The code-only fields are:

| Pipeline | Code-only fields |
|---|---|
| `credit_model` | `screening_artifact`, `feature_validation_result`, `extra_eval_datasets` |
| `feature_validation` | `psi_reference_data` |
| `reject_inference` | `oot_data`, `ri_approved_data`, `ri_approved_func` |
| `score_comparison` | `gains_add_func` |
| `score_consistency_uat` | `sqlrunner`, `offline_data`, `online_data` |

These fields remain usable in Python code, but a generic GUI form should not edit them. Two options change the result:
`exclude_none=True` drops fields whose value is `None`, and `include_non_serializable=True` keeps every field, with the
`repr()` text for values that cannot be converted.

`config_from_dict(config_class_or_key, payload, *, strict=True)` is the inverse. It accepts the plain dictionary or the whole
YAML payload (the fields are read from its `config` entry), and `strict` behaves as for YAML.

```python
from Modeling_Tool import config_from_dict

same_cfg = config_from_dict("mock_sample", payload)
print(same_cfg == cfg)
```

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

This prints one message (in Chinese in 0.8.2): `warm_start_score_col` is required when `warm_start_enabled` is on. The
validator returns a list of strings, errors first, then warnings, each prefixed with `WARNING: `. An empty list means it
found nothing. `values` can be a dictionary or a Config instance. A key you leave out counts as unset: a required key such as
`target_col` is reported as empty, and the others fall back to the Config default. Unknown keys are not reported (use
`config_from_dict` for that).

```python
print(validate_pipeline_config("mock_sample", {"n_samples": 500}))     # one WARNING: message
```

The validator is intentionally lightweight. It catches common form mistakes before code generation, while each Pipeline's
`run()` method remains the authoritative runtime validation. The checks are:

| Pipeline | Error | Warning |
|---|---|---|
| `credit_model` | `target_col` empty; `warm_start_enabled` without `warm_start_score_col`; `optuna_n_trials` below 1; `lr_search_params` keys other than `objective`, `primary_set`, `gap_ref_sets`, `metric`, `refit`, `verbose` | `optuna_n_trials` below 5 |
| `feature_validation` | `feature_batch_size` not positive; `enable_batch` without `feature_batch_size` or `feature_batches`; `batch_corr_mode="block_pairwise"` with `corr_params["method"]="kendall"` | `enable_batch=False` while batch settings are present |
| `reject_inference` | `approved_col` or `target_col` empty; no `ri_methods`; `train_prescore=False` without `score_col`; both `ri_approved_frac` and `ri_approved_n` set | |
| `score_comparison` | `target_col` empty; invalid `group_specs`; a `cross_metrics` entry that is not a `[column, aggregation]` pair; `pairwise_cross_agg_dict` that is not a dictionary | Neither `score_cols` nor `base_score` (the Pipeline then detects scores itself) |
| `score_consistency_uat` | `main_model_score_col` empty; `numeric_coercion_mode` not `safe`, `aggressive`, or `off`; `comparison_block_size` not positive | |
| `sample_analysis` | `target_cols` or `time_col` empty; `materialize_split=True` without `id_col` | |
| `mock_sample` | `n_samples` below 1; `applied_sample` not 0 or 1; `min_num_feature_business_type` above `min(num_features, 10)` | `n_samples` below 1000 |

The messages for `lr_search_params`, `group_specs`, `cross_metrics`, and `pairwise_cross_agg_dict` are in English; all others
are Chinese in 0.8.2.

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

The generated code:

```text
from Modeling_Tool.Pipeline import ScoreComparisonPipeline, ScoreComparisonPipelineConfig

cfg = ScoreComparisonPipelineConfig(
    target_col='badflag',
    score_cols=['score_old', 'score_new'],
    base_score='score_old',
)

pipeline = ScoreComparisonPipeline(cfg)
result = pipeline.run(data=your_dataframe)
```

For `MockSamplePipeline` and `ScoreConsistencyUATPipeline`, the generated code ends with `result = pipeline.run()`. For the
other five Pipelines it ends with `result = pipeline.run(data=your_dataframe)`. Values are written with `repr()`, so they
must be Python literals, and field names are not checked: a misspelled name only fails when you run the generated code.

## GUI Design Guidance

- Render the Pipeline cards from `get_pipeline_registry_schema()`.
- Render each form from `extract_pipeline_schema(key)["fields"]`, in section order by `group`. Collapse `advanced` fields
  unless they are also `required`, and ask for `required` ones.
- Hide fields with `gui_editable=False` unless the GUI has an explicit code-only advanced mode.
- Do not put non-serializable fields in YAML exports.
- Render dictionary fields with `nested_fields` as sub-forms when possible; otherwise, and for keys the hints do not list,
  use a JSON text area.
- Run `validate_pipeline_config` on every change, and treat its result as a form hint: the Pipeline's `run()` does the real
  validation.
- Show messages and labels through your own translation table, keyed on the registry `key` and the field `name`.
- Treat the `FeatureScreeningArtifact` hand-off as a Python-code workflow, not a YAML field.
- Keep Streamlit or web UI dependencies outside the SMF main package.
