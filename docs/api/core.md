# Modeling_Tool.Core

Infrastructure layer — binning, ODPS, utility functions, encryption, JSON, slope computation.

It has no cross-package dependencies and is depended on by all other subpackages.

## Sample Weights — `sample_weight_utils`

Resolves `weight_col` / `sample_weight` (and the `wgt` / `wgt_col` aliases) for uniform use by the Model / Eval layers.

::: Modeling_Tool.Core.sample_weight_utils

## Binning Tool — `Binning_Tool`

::: Modeling_Tool.Core.Binning_Tool

## ODPS Tool — `ODPS_Tool`

::: Modeling_Tool.Core.ODPS_Tool

## ODPS Concurrency Management — `Parallel_ODPS_Manager`

::: Modeling_Tool.Core.Parallel_ODPS_Manager

## Data Consistency Comparison — `Proc_Compare`

::: Modeling_Tool.Core.Proc_Compare

## Slope Computation — `Slope_Tool`

::: Modeling_Tool.Core.Slope_Tool

## General Utilities — `utils`

::: Modeling_Tool.Core.utils

## Encryption — `XOR_Encryptor`

::: Modeling_Tool.Core.XOR_Encryptor

## Extended DataFrame — `kDataFrame`

::: Modeling_Tool.Core.kDataFrame

## CDC JSON Conversion — `Json_Data_Converter`

::: Modeling_Tool.Core.Json_Data_Converter
