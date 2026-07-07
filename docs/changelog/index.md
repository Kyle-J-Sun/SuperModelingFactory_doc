# ChangeLog

SuperModelingFactory 各版本迭代总览。所有 `0.3.x` 发布在成型的发版流程之前,合并在 [v0.3.x 汇总](v0.3.md) 里;从 `0.4.0` 开始每个版本单独一页,包含完整的修复清单、行为变更说明与向后兼容影响。

## 发版策略

- **Patch 版本 (`0.x.Y`)**:纯硬化释放,零 API 移除,健康输入零数值差异。以字母编号:HIGH (H/N)、MEDIUM (M/N)、LOW (L/N)。
- **Minor 版本 (`0.X.0`)**:包含至少一处 **行为变更**(默认值翻转、返回值语义调整、必填字段变化)。任何 breaking default 必须先在上一 patch 通过 opt-in 参数落地,再在下一 minor 翻默认(参考 0.3.19 → 0.4.0 的 `cross_vars` 模式,以及计划中的 0.4.2 `missing_policy` 翻默认)。
- **Major 版本 (`X.0.0`)**:重大架构调整。目前无计划。

## 版本索引

<div class="grid cards" markdown>
- :material-tag: **[v0.5.4](v0.5.4.md)** — Pipeline GUI schema metadata

- :material-tag: **[v0.5.3](v0.5.3.md)** — Cross-module hardening

    0.6.0 spec bug batch shipped on the 0.5.x line: UAT aggressive coercion, prediction NaN stats, all-zero weight guards, ODPS partition atomic swap, RI NaN-score policy, evaluator duplicate protection, unified evaluate return type, safer AUC failures, and NaN handling for chi2/VIF.

- :material-tag: **[v0.5.1](v0.5.1.md)** — Feature / WOE hygiene

    Weighted IV zero-mass guards, PSI one-sided bucket smoothing, safer WOE missing sentinel, clearer WOE empty-dict errors, and per-variable failure diagnostics.

- :material-tag: **[v0.5.0](v0.5.0.md)** — 2026-07-07

    MEDIUM batch —— **PSI `missing_policy` 默认翻转（breaking）** + 6 个上游公开 API 暴露 kwarg，`MonotoneWOEBinner.apply_woe` 新增 `unseen_category_policy` 与 `_unseen_category_stats`

- :material-tag: **[v0.4.2](v0.4.2.md)** — 2026-07-06

    HIGH hotfix batch —— ODPS 原子上传、`split_df` `exclude_cols` 语义修正、`HardCutoffInferrer` NaN 守卫、`Weighted_Screen` 缺失分箱、`PSI_Tool.missing_policy` 新参数

- :material-tag: **[v0.4.1](v0.4.1.md)** — 2026-07-06

    LOW hygiene batch —— 拒绝推断默认 cutoff NaN 处理、`predict_positive` shape/长度/finite 校验、`feature_validation` 混合 Interval+NaN groupby 稳定性

- :material-tag: **[v0.4.0](v0.4.0.md)** — 2026-07-06

    MEDIUM hygiene batch #2 —— `cross_vars` 默认变更(**breaking**)、object 列 numeric 强转 opt-in、相关性计算向量化、样本分析 dry-run 等六项

- :material-history: **[v0.3.x 汇总](v0.3.md)** — 2026-06-30 之前

    0.3.4 → 0.3.18,涵盖 WOE 分箱引擎、统一 `feature_screen`、加权 screen、`FeatureScreeningArtifact`、Pipeline 硬化批次 H1-H4、hygiene 批次 M2-M4

</div>

## 关联信息

- [SMF 主仓 Releases](https://github.com/Kyle-J-Sun/SuperModelingFactory/releases)
- [完整 tag 列表](https://github.com/Kyle-J-Sun/SuperModelingFactory/tags)
- API 破坏性变更、行为翻转均在对应版本页明确标注,搜索关键词 "**breaking change**" 或 "**行为变更**"。
