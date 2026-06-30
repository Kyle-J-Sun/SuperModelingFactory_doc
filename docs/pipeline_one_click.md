# 一键建模流水线

`Modeling_Tool.Pipeline` 提供三条高层业务流水线，把常见的端到端脚本封装成可复用、可配置、可返回结构化结果的 API。

这三条 Pipeline **不生成模拟数据**。调用方需要先准备真实业务 DataFrame，然后通过 `Config` 控制要跑哪些步骤、哪些模型、哪些维度和哪些输出。

```python
from Modeling_Tool.Pipeline import (
    RejectInferencePipeline,
    RejectInferencePipelineConfig,
    CreditModelPipeline,
    CreditModelPipelineConfig,
    ScoreComparisonPipeline,
    ScoreComparisonPipelineConfig,
)
```

也可以从主包顶层导入：

```python
from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig
```

## 1. 拒绝推断流水线

`RejectInferencePipeline` 用于已审批通过样本有标签、被拒绝样本无标签的场景。它可以自动训练 pre-score，也可以直接使用你已经准备好的拒绝推断分数。

### 最小示例

```python
from Modeling_Tool import RejectInferencePipeline, RejectInferencePipelineConfig

cfg = RejectInferencePipelineConfig(
    output_dir="output/reject_inference",
    approved_col="approved",
    target_col="badflag",
    score_col="prescore_prob",
    feature_cols=[
        "income", "age", "score_b", "mob_on_book",
        "overdue_days_max", "util_rate", "loan_amount",
    ],
    train_prescore=True,
    ri_methods=["simple_augment", "hard_cutoff", "fuzzy_augment", "parceling"],
    train_ri_models=True,
)

result = RejectInferencePipeline(cfg).run(application_df)

result.ri_summary
result.ri_datasets["parceling"]
result.ri_model_perf
result.best_method
```

### 输入数据要求

| 列 | 是否必需 | 说明 |
|---|---:|---|
| `approved_col` | 是 | 审批通过标识，默认 `approved`；通过样本为 `1`，拒绝样本为 `0`。 |
| `target_col` | 是 | 表现标签，默认 `badflag`；通过样本需要有标签，拒绝样本可为空。 |
| `score_col` | 视配置 | 拒绝推断分数。若 `train_prescore=True`，Pipeline 会生成该列；若 `False`，输入数据必须已有该列。 |
| `feature_cols` | 是 | 训练 pre-score 和 RI 后模型使用的特征列。 |

### `RejectInferencePipelineConfig` 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `output_dir` | `"output/reject_inference"` | 输出根目录。数据集、CSV 报告和 Excel 报告会写到该目录下。 |
| `approved_col` | `"approved"` | 审批通过标识列。 |
| `target_col` | `"badflag"` | 目标标签列。 |
| `score_col` | `"prescore_prob"` | RI 使用的预评分列。 |
| `feature_cols` | `None` | 特征列列表。建议显式传入；不传时会从数值列中排除标签、审批标识、分数字段后推断。 |
| `random_state` | `42` | 随机种子，用于 pre-score 切分和 OOT 抽样。 |
| `write_outputs` | `True` | 是否写 CSV、图片等中间产物。 |
| `write_excel` | `True` | 是否输出 Excel 报告。 |
| `train_prescore` | `True` | 是否在通过样本上训练 pre-score 模型并给全量样本打分。 |
| `prescore_model_type` | `"lgb"` | pre-score 模型类型，传给 `GradientBoostingModel`，常用 `"lgb"`、`"xgb"`、`"cat"`。 |
| `prescore_params` | `{}` | pre-score 模型参数，会覆盖默认 LightGBM 参数。 |
| `prescore_test_size` | `0.3` | 在通过样本内部切分 pre-score validation 的比例。 |
| `ri_methods` | `["simple_augment", "hard_cutoff", "fuzzy_augment", "parceling"]` | 要尝试的拒绝推断方法。支持别名：`simple`、`hard`、`fuzzy`、`parcel`。 |
| `ri_method_params` | `{}` | 各 RI 方法的独立参数。 |
| `train_ri_models` | `True` | 是否基于每个 RI 增强数据集训练后续模型并比较 OOT 表现。 |
| `ri_model_type` | `"lgb"` | RI 后建模使用的模型类型。 |
| `ri_model_params` | `{}` | RI 后模型参数，会覆盖默认 LightGBM 参数。 |
| `oot_data` | `None` | 外部指定 OOT 数据。传入后不再从 approved 样本随机抽 OOT。 |
| `oot_frac` | `0.2` | 未传 `oot_data` 时，从 approved 样本中随机抽取 OOT 的比例。 |
| `perf_pct_bins` | `10` | `PerformanceEvaluator` 的分箱数量。 |
| `min_bin_prop` | `0.03` | 性能评估最小分箱占比。 |

### `ri_method_params` 写法

```python
cfg = RejectInferencePipelineConfig(
    ri_methods=["hard_cutoff", "parceling"],
    ri_method_params={
        "hard_cutoff": {"cutoff": 0.42},
        "parceling": {"n_parcels": 8},
        "simple_augment": {"bad_rate": 0.18},
        "fuzzy_augment": {"weight_factor": 1.5},
    },
)
```

| 方法 | 可配置参数 | 说明 |
|---|---|---|
| `simple_augment` | `bad_rate` | 不传时自动使用通过样本平均坏账率。 |
| `hard_cutoff` | `cutoff` | 不传时使用通过样本坏样本 pre-score 的 25 分位数。 |
| `fuzzy_augment` | `weight_factor` | 控制 fuzzy augmentation 权重强度。 |
| `parceling` | `n_parcels` | 按 score 分档的档数。 |

### 结果对象

`RejectInferencePipeline.run()` 返回 `RejectInferencePipelineResult`。

| 字段 | 说明 |
|---|---|
| `approved_data` | 通过样本数据，包含 pre-score。 |
| `rejected_data` | 拒绝样本数据，包含 pre-score。 |
| `ri_datasets` | `{method: DataFrame}`，每种 RI 方法生成的增强训练集。 |
| `ri_summary` | 每种 RI 方法的数据规模、坏账率、是否有权重列等汇总。 |
| `ri_model_perf` | 每种 RI 增强数据训练模型后的 train / validation / OOT 表现。 |
| `best_method` | 按 OOT AUC 排名的最佳 RI 方法。 |
| `prescore_model` | pre-score 模型 wrapper；若 `train_prescore=False` 则为空。 |
| `ri_models` | `{method: model}`，每种 RI 方法训练出的模型。 |
| `report_path` | Excel 报告路径；若 `write_excel=False` 则为空。 |

## 2. 信用建模流水线

`CreditModelPipeline` 封装完整信用建模主线：样本切分、PSI/IV/相关性筛选、WOE、模型训练、backward、Optuna、评估、解释和 Excel 报告。

### 最小示例

```python
from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig

cfg = CreditModelPipelineConfig(
    output_dir="output",
    target_col="badflag",
    feature_cols=[
        "income", "age", "score_b", "mob_on_book",
        "overdue_days_max", "util_rate", "loan_amount",
    ],
    oot_col="oot_flag",
    woe_engine="equal_freq",
    train_models=["lr", "lgb", "xgb", "cat"],
    backward_enabled=True,
    optuna_models=["lgb", "xgb", "cat"],
    explain_models=["lr", "lgb", "cat"],
    owen_enabled=True,
)

result = CreditModelPipeline(cfg).run(modeling_df)

result.selected_features
result.models["lgb"]
result.perf_results["lgb"]
```

### 输入数据要求

| 场景 | 必需列 | 说明 |
|---|---|---|
| 已切好样本 | `sample_col` | 默认 `sample_ind`，取值建议为 `ins`、`oos`、`oot`。 |
| 只标记 OOT | `oot_col` | 默认 `oot_flag`，`0` 为 INS+OOS，非 `0` 为 OOT；Pipeline 再随机切 INS/OOS。 |
| 未标记样本 | 无 | 会从全量数据随机切 INS/OOS，并用 OOS 作为 OOT 的兜底。 |

### `CreditModelPipelineConfig` 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `output_dir` | `"output"` | 输出根目录。 |
| `target_col` | `"badflag"` | 目标标签列。 |
| `feature_cols` | `None` | 原始入模特征列。建议显式传入；不传时从数值列推断。 |
| `sample_col` | `"sample_ind"` | 已切分样本标识列。若存在并包含 `ins/oos/oot`，优先使用。 |
| `oot_col` | `"oot_flag"` | OOT 标识列。仅在没有有效 `sample_col` 时使用。 |
| `random_state` | `42` | 随机种子。 |
| `write_outputs` | `True` | 是否输出 CSV、图表等中间文件。 |
| `write_excel` | `True` | 是否输出 Excel 报告。 |
| `split_config` | `{"test_size": 0.3, "stratify": True}` | INS/OOS 切分配置。 |
| `feature_selection` | 见下表 | PSI、IV、相关性筛选开关和阈值。 |
| `woe_engine` | `"equal_freq"` | WOE 引擎。支持 `"equal_freq"` 和 `"monotone"`。 |
| `woe_params` | `{"nbins": 10, "equal_freq": True, "min_bin_prop": 0.05}` | `WOE_Master.fit()` 参数和通用 WOE 配置。 |
| `monotone_woe_params` | `{"n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2}` | `MonotoneWOEBinner` 参数。 |
| `train_models` | `["lr", "lgb", "xgb", "cat"]` | 要训练的模型列表。 |
| `model_params` | `{}` | 每个模型的参数字典。 |
| `backward_enabled` | `True` | 是否运行 backward variable elimination。 |
| `backward_model` | `"lgb"` | backward 使用的代理模型。 |
| `backward_params` | `{}` | backward 初始化和运行参数。 |
| `use_backward_features` | `True` | 是否用 backward 选出的特征重新训练模型。 |
| `optuna_models` | `["lgb", "xgb", "cat"]` | 要跑 Optuna 搜索的模型。传 `[]` 可关闭。 |
| `optuna_n_trials` | `5` | 每个模型 Optuna trial 数。 |
| `optuna_params` | `{}` | Optuna 搜索空间和通用参数覆盖。 |
| `explain_models` | `["lr", "lgb", "cat"]` | 要跑 SHAP/LIME/PDP/ALE/ICE 主解释路径的模型。传 `[]` 可关闭。 |
| `explain_params` | `{"sample_n": 500, "background_n": 200}` | 解释样本量、背景样本量和 Owen 参数。 |
| `owen_enabled` | `True` | 是否计算 Owen value。 |
| `business_prior_groups` | `None` | Owen value 的业务先验分组。 |
| `perf_pct_bins` | `10` | 性能评估分箱数量。 |
| `perf_min_bin_prop` | `0.03` | 性能评估最小分箱占比。 |

### `split_config`

```python
split_config={
    "test_size": 0.3,
    "stratify": True,
    "random_state": 2026,
}
```

| 键 | 默认值 | 说明 |
|---|---|---|
| `test_size` | `0.3` | INS/OOS 切分中 OOS 占比。 |
| `stratify` | `True` | 是否按目标变量分层抽样。 |
| `random_state` | 使用顶层 `random_state` | 切分随机种子。 |

### `feature_selection`

```python
feature_selection={
    "psi_enabled": True,
    "psi_threshold": 0.2,
    "iv_enabled": True,
    "iv_threshold": 0.02,
    "corr_enabled": True,
    "corr_threshold": 0.75,
}
```

| 键 | 默认值 | 说明 |
|---|---|---|
| `psi_enabled` | `True` | 是否运行 PSI 稳定性筛选。 |
| `psi_threshold` | `0.2` | PSI 小于该阈值的变量保留。 |
| `psi_buckets` | `10` | PSI 分箱数。 |
| `iv_enabled` | `True` | 是否运行 IV 筛选。 |
| `iv_threshold` | `0.02` | IV 大于等于该阈值的变量保留。 |
| `iv_nbins` | `10` | IV 分析分箱数。 |
| `iv_equal_freq` | `True` | IV 分析是否等频分箱。 |
| `iv_min_bin_prop` | `0.05` | IV 分析最小分箱占比。 |
| `corr_enabled` | `True` | 是否运行高相关剔除。 |
| `corr_threshold` | `0.75` | 相关性阈值。 |
| `corr_max_iterations` | `10` | 相关性剔除最大迭代次数。 |

### WOE 参数

`woe_engine="equal_freq"` 时使用 `WOE_Master`：

```python
woe_params={
    "nbins": 10,
    "equal_freq": True,
    "min_bin_prop": 0.05,
    "woe_suffix": "_woe",
    "missing_ref_value": -999999,
}
```

`woe_engine="monotone"` 时使用 `MonotoneWOEBinner`：

```python
monotone_woe_params={
    "n_init_bins": 20,
    "min_bin_size": 0.03,
    "min_n_bins": 2,
    "special_values": [-999999],
    "chi2_binning": False,
}
```

### 模型参数

`train_models` 控制训练哪些模型，`model_params` 按模型名覆盖默认参数。

```python
model_params={
    "lr": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "standardize": False,
    },
    "lgb": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "early_stopping_rounds": 50,
        "eval_metric": "auc",
    },
    "xgb": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "auc",
    },
    "cat": {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "AUC",
    },
}
```

### Backward 参数

```python
backward_params={
    "init": {
        "model_type": "lgbm",
    },
    "run": {
        "n_rounds": 3,
        "cum_importance_threshold": 0.99,
        "min_vars": 5,
        "ret_perf": True,
    },
}
```

| 键 | 说明 |
|---|---|
| `init` | 覆盖 `BackwardVariableEliminator` 初始化参数。 |
| `run` | 覆盖 backward 运行参数。 |

### Optuna 参数

```python
optuna_models=["lgb", "xgb"]
optuna_n_trials=20
optuna_params={
    "search_spaces": {
        "lgb": {
            "num_leaves": {"type": "int", "low": 16, "high": 64},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        },
    },
    "common": {
        "objective": "oot_gap_penalized",
        "primary_set": "oos",
        "gap_ref_sets": ["oot"],
        "metric": "auc",
        "refit": True,
    },
}
```

### Explain / Owen 参数

```python
explain_params={
    "sample_n": 500,
    "background_n": 200,
    "owen_threshold": 0.35,
    "owen_method": "complete",
    "owen_corr_method": "spearman",
    "owen_model_output": "probability",
}

business_prior_groups={
    "repayment_capacity": ["income_woe", "employment_months_woe", "loan_amount_woe"],
    "credit_behavior": ["score_b_woe", "overdue_days_max_woe", "mob_on_book_woe"],
}
```

| 键 | 默认值 | 说明 |
|---|---|---|
| `sample_n` | `500` | 解释计算使用的 OOS 样本数。 |
| `background_n` | `200` | SHAP/Owen 背景样本数。 |
| `owen_threshold` | `0.35` | coalition 自动聚类距离阈值。 |
| `owen_method` | `"complete"` | 层次聚类方法。 |
| `owen_corr_method` | `"spearman"` | 相关性方法。 |
| `owen_min_group_size` | `1` | 自动分组最小组大小。 |
| `owen_intra_dist` | `0.01` | 同组特征在 linkage 中的距离。 |
| `owen_inter_dist` | `0.99` | 跨组特征在 linkage 中的距离。 |
| `owen_model_output` | `"probability"` | Owen value 解释的模型输出空间。 |

### 结果对象

`CreditModelPipeline.run()` 返回 `CreditModelPipelineResult`。

| 字段 | 说明 |
|---|---|
| `splits` | `{"ins": df, "oos": df, "oot": df}`。 |
| `feature_selection_summary` | PSI、IV、相关性筛选结果和最终变量列表。 |
| `woe_artifacts` | WOE 引擎、WOE 后数据、WOE 特征名、WOE 表。 |
| `models` | `{model_name: (wrapper, raw_model, feature_cols)}`。 |
| `selected_features` | 最终用于训练和评估的特征。 |
| `backward_summary` | backward 汇总表；未启用时为空。 |
| `optuna_results` | `{model_name: search_result_df}`。 |
| `perf_results` | `{model_name: perf_df}`。 |
| `explain_outputs` | SHAP/Owen 等解释输出。 |
| `report_path` | Excel 报告路径；若 `write_excel=False` 则为空。 |

## 3. 模型分对比流水线

`ScoreComparisonPipeline` 用于多模型分、多版本分数或 champion/challenger 模型分对比。它封装全局 AUC/KS、分维度 AUC/KS、Gains、自定义指标、cross risk 和 pairwise score cross risk。

### 最小示例

```python
from Modeling_Tool import ScoreComparisonPipeline, ScoreComparisonPipelineConfig

cfg = ScoreComparisonPipelineConfig(
    output_dir="output/score_comparison",
    target_col="badflag",
    score_cols=["score_A", "score_B", "score_C"],
    base_score="score_A",
    comp_scores=["score_B", "score_C"],
    weight_col="_w_ones",
    time_dims=["apply_month", "vintage"],
    population_dims=["channel", "product_type"],
    custom_metric_cols=["credit_limit", "age", "apr"],
    cross_vars=["rating"],
    cross_binning_numeric=[True, False],
)

result = ScoreComparisonPipeline(cfg).run(score_df)

result.global_perf
result.group_perf["channel_x_apply_month"]
result.gains
result.cross_results
result.pairwise_cross
```

### 输入数据要求

| 列 | 是否必需 | 说明 |
|---|---:|---|
| `target_col` | 是 | 目标标签列。 |
| `score_cols` / `base_score` / `comp_scores` | 是 | 模型分列。 |
| `weight_col` | 否 | 样本权重列。 |
| `time_dims` | 否 | 时间维度列，如 `apply_month`、`week`、`vintage`。 |
| `population_dims` | 否 | 人群维度列，如 `channel`、`product_type`、`city_tier`。 |
| `cross_vars` | 否 | 交叉风险中的第二维变量，如 `rating`。 |
| `flow_id` | 否 | 若不存在，Pipeline 会自动生成。 |

### `ScoreComparisonPipelineConfig` 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `output_dir` | `"output/score_comparison"` | 输出根目录。 |
| `target_col` | `"badflag"` | 目标标签列。 |
| `score_cols` | `None` | 全部分数列。若传入，会结合 `base_score` 自动推断比较分。 |
| `base_score` | `None` | 基准分数列。未传时使用 `score_cols[0]`。 |
| `comp_scores` | `None` | 待比较分数列。未传时使用 `score_cols` 中除 `base_score` 以外的列。 |
| `weight_col` | `None` | 样本权重列。 |
| `random_state` | `42` | 随机种子，保留给后续需要抽样的扩展逻辑。 |
| `write_outputs` | `True` | 是否输出 CSV。 |
| `write_excel` | `True` | 是否输出 Excel 报告。 |
| `nbins` | `10` | Gains 和 cross risk 分箱数。 |
| `min_bin_prop` | `0.02` | 最小分箱占比。 |
| `equal_freq` | `True` | 是否等频分箱。 |
| `min_data_size` | `50` | 全局和分组评估的最小样本量。 |
| `precision` | `5` | 数值精度。 |
| `include_missing` | `False` | 是否在分箱中包含缺失值。 |
| `fillna` | `-999999` | 缺失填充值。 |
| `positive_score_only` | `True` | 是否按正向分数处理。 |
| `time_dims` | `["apply_month"]` | 时间维度列表。 |
| `population_dims` | `["channel"]` | 人群维度列表。 |
| `segment_dims` | `None` | `population_dims` 的别名。传入后会覆盖 `population_dims`。 |
| `include_time_population_cross` | `True` | 是否自动跑人群 x 时间交叉维度。 |
| `group_min_size` | `None` | 分组评估最小样本量；不传时使用 `min_data_size`。 |
| `group_specs` | `None` | 高级自定义分组配置。传入后会覆盖 `time_dims/population_dims` 自动生成逻辑。 |
| `gains_add_func` | `None` | 自定义 Gains 分箱附加指标函数。 |
| `custom_metric_cols` | `["credit_limit", "age", "apr"]` | 默认自定义业务指标列，会自动计算均值。 |
| `gains_display_metric_list` | 标准 Gains 指标列表 | `add_func=None` 时用于控制 Gains 展示列。 |
| `cross_vars` | `["rating"]` | cross risk 的第二维变量列表。 |
| `cross_metrics` | `{}` | cross risk 指标配置；不传时使用 bad rate 和 `custom_metric_cols` 均值。 |
| `cross_binning_numeric` | `[True, False]` | cross risk 两个维度是否数值分箱。支持 bool 或二元列表。 |
| `pairwise_cross_enabled` | `True` | 是否计算 compare score x base score 的 pairwise cross risk。 |
| `pairwise_cross_agg_dict` | `None` | pairwise cross risk 的聚合配置。 |

### 时间维度与人群维度

最常见的用法是直接传 `time_dims` 和 `population_dims`：

```python
cfg = ScoreComparisonPipelineConfig(
    time_dims=["apply_month", "vintage"],
    population_dims=["channel", "product_type", "city_tier"],
    include_time_population_cross=True,
    group_min_size=100,
)
```

Pipeline 会自动生成：

| 输出 key | 维度 |
|---|---|
| `apply_month` | 单时间维度 |
| `vintage` | 单时间维度 |
| `channel` | 单人群维度 |
| `product_type` | 单人群维度 |
| `city_tier` | 单人群维度 |
| `channel_x_apply_month` | 人群 x 时间 |
| `channel_x_vintage` | 人群 x 时间 |
| `product_type_x_apply_month` | 人群 x 时间 |
| `city_tier_x_vintage` | 人群 x 时间 |

结果在：

```python
result.group_perf["channel_x_apply_month"]
```

如果你只想跑单维，不跑交叉：

```python
cfg = ScoreComparisonPipelineConfig(
    time_dims=["apply_month"],
    population_dims=["channel", "product_type"],
    include_time_population_cross=False,
)
```

### 高级分组：`group_specs`

如果维度组合不是简单的人群 x 时间，可以直接传 `group_specs`。一旦传入 `group_specs`，Pipeline 不再自动使用 `time_dims/population_dims`。

```python
cfg = ScoreComparisonPipelineConfig(
    group_specs=[
        {"name": "month", "columns": ["apply_month"], "min_size": 100},
        {"name": "channel_x_month", "columns": ["channel", "apply_month"], "min_size": 100},
        {"name": "channel_x_product_x_month", "columns": ["channel", "product_type", "apply_month"], "min_size": 80},
    ],
)
```

| 键 | 说明 |
|---|---|
| `name` | 输出 key 和落盘文件名的一部分。 |
| `columns` | 依次 group by 的列。一个列表示单维，多个列表示链式交叉维度。 |
| `min_size` | 该分组下最小样本量。 |

### Gains 自定义指标

不传 `gains_add_func` 时，Pipeline 会对 `custom_metric_cols` 自动输出均值：

```python
cfg = ScoreComparisonPipelineConfig(
    custom_metric_cols=["credit_limit", "age", "apr"],
)
```

需要自定义更多指标时传函数：

```python
import pandas as pd

def add_business_metrics(sub_df):
    return pd.Series({
        "credit_limit_mean": sub_df["credit_limit"].mean(),
        "apr_p75": sub_df["apr"].quantile(0.75),
        "approval_amount_sum": sub_df["approval_amount"].sum(),
    })

cfg = ScoreComparisonPipelineConfig(
    gains_add_func=add_business_metrics,
)
```

### Cross risk 指标

默认 cross risk 指标为：

- `bad_rate`: `target_col` 的均值
- `custom_metric_cols` 中每个字段的均值

自定义写法：

```python
cfg = ScoreComparisonPipelineConfig(
    cross_vars=["rating", "policy_group"],
    cross_metrics={
        "bad_rate": ("badflag", "mean"),
        "credit_limit": ("credit_limit", "mean"),
        "cnt": ("flow_id", "count"),
    },
    cross_binning_numeric=[True, False],
)
```

`cross_binning_numeric` 用于控制 `[score, cross_var]` 两个维度是否数值分箱：

| 值 | 含义 |
|---|---|
| `[True, False]` | 分数列分箱，评级/人群列按类别原值展示。 |
| `[True, True]` | 两个变量都数值分箱。 |
| `False` | 两个变量都按类别处理。 |

### Pairwise cross risk

默认会计算 `comp_scores` 与 `base_score` 的 pairwise cross risk。

关闭：

```python
cfg = ScoreComparisonPipelineConfig(pairwise_cross_enabled=False)
```

自定义聚合：

```python
cfg = ScoreComparisonPipelineConfig(
    pairwise_cross_agg_dict={
        "badflag": ["count", lambda x: round(x.sum() / x.count(), 4)],
        "credit_limit": ["count", lambda x: round(x.mean(), 2)],
        "apr": ["count", lambda x: round(x.mean(), 4)],
    },
)
```

### 结果对象

`ScoreComparisonPipeline.run()` 返回 `ScoreComparisonPipelineResult`。

| 字段 | 说明 |
|---|---|
| `global_perf` | 所有 score 的全局 AUC/KS 表。 |
| `group_perf` | `{group_name: DataFrame}`，时间、人群和交叉维度评估表。 |
| `gains` | 全局 Gains 表，可包含自定义业务指标。 |
| `cross_results` | `{score__cross_var__metric: DataFrame}`。 |
| `pairwise_cross` | compare score x base score 的 pairwise cross risk。 |
| `report_path` | Excel 报告路径；若 `write_excel=False` 则为空。 |

## 推荐配置模板

### 快速信用建模

```python
cfg = CreditModelPipelineConfig(
    target_col="badflag",
    feature_cols=features,
    oot_col="oot_flag",
    train_models=["lr", "lgb"],
    backward_enabled=False,
    optuna_models=[],
    explain_models=[],
    owen_enabled=False,
)
```

### 完整信用建模

```python
cfg = CreditModelPipelineConfig(
    target_col="badflag",
    feature_cols=features,
    oot_col="oot_flag",
    woe_engine="monotone",
    train_models=["lr", "lgb", "xgb", "cat"],
    backward_enabled=True,
    backward_model="lgb",
    use_backward_features=True,
    optuna_models=["lgb", "xgb", "cat"],
    optuna_n_trials=20,
    explain_models=["lr", "lgb", "cat"],
    owen_enabled=True,
)
```

### 模型分多维对比

```python
cfg = ScoreComparisonPipelineConfig(
    target_col="badflag",
    score_cols=["score_A", "score_B", "score_C", "score_D"],
    base_score="score_A",
    comp_scores=["score_B", "score_C", "score_D"],
    weight_col="sample_weight",
    time_dims=["apply_month", "vintage"],
    population_dims=["channel", "product_type", "city_tier"],
    custom_metric_cols=["credit_limit", "age", "apr"],
    cross_vars=["rating"],
    cross_binning_numeric=[True, False],
)
```

## 常见关闭项

| 目标 | 配置 |
|---|---|
| 不落盘 CSV / 图片 | `write_outputs=False` |
| 不输出 Excel | `write_excel=False` |
| 不训练 RI 后模型 | `train_ri_models=False` |
| 不跑 backward | `backward_enabled=False` |
| 不跑 Optuna | `optuna_models=[]` |
| 不跑解释 | `explain_models=[]` |
| 不跑 Owen | `owen_enabled=False` |
| 不跑人群 x 时间交叉 | `include_time_population_cross=False` |
| 不跑 pairwise cross risk | `pairwise_cross_enabled=False` |

