# Modeling_Tool.Core

基础设施层 —— 分箱、ODPS、工具函数、加密、JSON、斜率计算。

无跨包依赖，被其他所有子包依赖。

## 样本权重 — `sample_weight_utils`

解析 `weight_col` / `sample_weight`（及 `wgt` / `wgt_col` 别名），供 Model / Eval 层统一调用。

::: Modeling_Tool.Core.sample_weight_utils

## 分箱工具 — `Binning_Tool`

::: Modeling_Tool.Core.Binning_Tool

## ODPS 工具 — `ODPS_Tool`

::: Modeling_Tool.Core.ODPS_Tool

## 斜率计算 — `Slope_Tool`

::: Modeling_Tool.Core.Slope_Tool

## 通用工具 — `utils`

::: Modeling_Tool.Core.utils

## 加密 — `XOR_Encryptor`

::: Modeling_Tool.Core.XOR_Encryptor

## 扩展 DataFrame — `kDataFrame`

::: Modeling_Tool.Core.kDataFrame

## CDC JSON 转换 — `Json_Data_Converter`

::: Modeling_Tool.Core.Json_Data_Converter
