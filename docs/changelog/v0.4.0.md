# v0.4.0 变更笔记 —— MEDIUM hygiene batch #2

发布日期：2026-07-06

本次批量修复六个 MEDIUM 级别的 hygiene bug，覆盖 5 个 pipeline 模块。其中 **M5** 是**行为变更**（会影响原本依赖隐式默认的调用），因此版本升为 0.4.0 而不是 0.3.19。其它五项都是纯粹的鲁棒性/性能改进，向后兼容。

## Bug 一览

| ID | 模块 | 类型 | 影响面 |
|---|---|---|---|
| M1 | `CreditModelPipeline._run_optuna` | 行为修正 | 空 `search_spaces={}` 现在被尊重 |
| M5 | `ScoreComparisonPipelineConfig.cross_vars` | **默认值变更** | `["rating"]` → `[]` |
| M6 | `ScoreConsistencyUATPipeline._load_dataframes` | 行为修正 | object 列 numeric 强转改为 opt-in |
| M7 | `FeatureValidationPipeline._split_frame` | 观测性 | bare `except: pass` → warning log |
| M8 | `FeatureValidationPipeline._high_corr_pairs` | 性能 | O(n²) Python 循环 → `np.triu_indices` 向量化 |
| M9 | `SampleAnalysisPipeline` | 新 API | 新增 `dry_run` 字段 + `estimate_split_count()` |

## M1: `optuna_params["search_spaces"]={}` 现在被尊重

**问题**：`search_spaces = cfg.optuna_params.get("search_spaces") or self._default_search_spaces()`。空 dict `{}` 触发 `or` 短路，回退到内置默认搜索空间。用户显式传空表意为"我不想搜"却拿到默认搜索。

**修复**：显式 `is None` 判断。

```python
# 0.4.0 后
user_search_spaces = cfg.optuna_params.get("search_spaces")
search_spaces = self._default_search_spaces() if user_search_spaces is None else user_search_spaces
```

**迁移**：无。**要触发 fallback**，请不要设置 `search_spaces` 键（或明确传 `None`）。

## M5: `cross_vars` 默认从 `["rating"]` 改为 `[]` （BREAKING）

**问题**：`ScoreComparisonPipelineConfig.cross_vars` 默认 `["rating"]`。多数场景没有 `rating` 列 —— 直接 `KeyError: Missing required columns: ['rating']` 于 `_validate_input`。想跑标准 score 对比但没有 rating 列的用户必须显式覆盖成 `[]`。

**修复**：
- 默认改成 `[]`；不指定 cross-var 时不再产出交叉表。
- 当 `cross_vars` 中存在但输入数据不含时，改为 `_logger.warning(...)` + skip；不再抛错。
- `_validate_input` 不再把 cross_vars 计入 required 列。

**迁移**：
- 依赖 `rating` 交叉：**显式**写 `cross_vars=["rating"]`。
- 没有交叉需求：不用改，之前你八成是靠手动传 `cross_vars=[]` 绕过的，现在这就是默认。

## M6: object 列 → numeric 强转改为 opt-in

**问题**：`_load_dataframes` 遍历 `df_compare` 所有 object 列，`pd.to_numeric(..., errors="coerce")` 后只要 `notna().any()` 就整列覆盖。例如 rating 列 `["A", "B", "1", "2"]`，A/B 被静默转成 NaN。

**修复**：新增两个 config 字段：

- `numeric_coercion_mode: str = "safe"`：三种模式
  - `"safe"`（默认，0.4.0 新语义）：仅当**≥ `numeric_coercion_min_ratio`**（默认 99%）的非空值可解析为数值时才转换；否则保留 object，且当**部分**值可解析时发出 warning 提示用户显式选择。
  - `"aggressive"`：等价于 0.3.x 行为（任何 parseable 就转），但改为**每次 coerce 都 warn**丢失的字符串个数。
  - `"off"`：完全跳过 coerce。
- `numeric_coercion_min_ratio: float = 0.99`：只在 safe 模式使用。

**迁移**：如果依赖旧行为，请设置 `numeric_coercion_mode="aggressive"`。

## M7: `_split_frame` bare-except 改为 warning log

**问题**：`FeatureValidationPipeline._split_frame` 用 `try: SampleSplitter(...).split_df(...) except Exception: pass` 静默 fallback 到朴素随机切分。stratify 出问题时没有任何输出。

**修复**：`except Exception as exc` + `_logger.warning(...)`，含 `target/test_size/random_state/error` 详情。

**迁移**：无。若之前调试痛苦，现在有日志。

## M8: `_high_corr_pairs` 向量化

**问题**：3000+ 特征时，原实现 `for i, var1 in enumerate(cols): for var2 in cols[i+1:]: corr_matrix.loc[var1, var2]` 约 450 万次 `.loc[]` 标量查找，明显是瓶颈。

**修复**：`np.triu_indices(n, k=1)` 一次拿到上三角坐标 → `np.abs > threshold` mask 一次筛选 → 用集合 membership 生成 pair_type。**数值输出与旧循环版本严格一致**，回归测试 `TestM8HighCorrPairsVectorized::test_matches_loop_reference` 验证。

**迁移**：无。等相同结果，更快。

## M9: `SampleAnalysisPipeline.dry_run` + `estimate_split_count()`

**问题**：`SampleAnalysisPipelineConfig` 默认 `4 targets × 4 windows × 3 ratios × 20 seeds = 960` 次 `SampleSplitter` 调用，用户没意识到就跑起来了。

**修复**：
- 新字段 `dry_run: bool = False`。开启后 `run()` 短路：跳过所有耗时计算，只返回 `split_candidate_summary=DataFrame([info])`（info 见下），其它 DataFrame 为空。
- 新公共方法 `estimate_split_count(data=None) -> dict`：返回估计的 splits 上限；若传 data，则额外统计 `available_oot_periods` 和 `windows_usable`。

```python
pipe = SampleAnalysisPipeline(cfg)
pipe.estimate_split_count(df)
# {'n_targets': 4, 'n_oot_windows': 4, 'n_ins_oos_ratios': 3,
#  'n_random_seeds': 20, 'estimated_max_splits': 960,
#  'available_oot_periods': 12, 'windows_usable': 4}
```

**迁移**：无。默认关闭。想 preview：显式 `dry_run=True` 或调用 `estimate_split_count()`。

## 版本

- **Version**: 0.4.0
