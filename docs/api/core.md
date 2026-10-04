# Modeling_Tool.Core

Foundation layer: binning, model registry, sample-weight helpers, the ODPS client, the parallel apply engine, ProcCompare,
encryption, JSON conversion, slope computation, and general utilities. The other subpackages build on it. At import time it
imports none of them (see the [import rules](../architecture.md#rules)).

## Sample Weights: `sample_weight_utils`

Resolves `weight_col` / `sample_weight` (and the `wgt_col` / `wgt` aliases) for uniform use by the Model and Eval layers.

::: Modeling_Tool.Core.sample_weight_utils

## Binning: `Binning_Tool`

::: Modeling_Tool.Core.Binning_Tool

## Model Registry: `Model_Registry_Tool`

`save_model`, `load_model`, and `load_model_metadata` are also importable from `Modeling_Tool`.

::: Modeling_Tool.Core.Model_Registry_Tool

## ODPS Client: `ODPS_Tool`

Needs the `odps` extra, and the `ALIBABA_CLOUD_ACCESS_KEY_ID` and `ALIBABA_CLOUD_ACCESS_KEY_SECRET` environment variables
when you construct `ODPSRunner`. See [ODPS Data Extraction](../guides/odps.md).

::: Modeling_Tool.Core.ODPS_Tool

## ODPS Concurrency: `Parallel_ODPS_Manager`

::: Modeling_Tool.Core.Parallel_ODPS_Manager

## Parallel Apply Engine: `Parallel_Engine`

::: Modeling_Tool.Core.Parallel_Engine

## Dataset Comparison: `Proc_Compare`

::: Modeling_Tool.Core.Proc_Compare

## Slope Computation: `Slope_Tool`

::: Modeling_Tool.Core.Slope_Tool

## General Utilities: `utils`

::: Modeling_Tool.Core.utils

## Encryption: `XOR_Encryptor`

::: Modeling_Tool.Core.XOR_Encryptor

## Extended DataFrame: `kDataFrame`

::: Modeling_Tool.Core.kDataFrame

## DataFrame and JSON Conversion: `Json_Data_Converter`

::: Modeling_Tool.Core.Json_Data_Converter
