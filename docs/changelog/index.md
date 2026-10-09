# ChangeLog

An overview of every SuperModelingFactory release. All `0.3.x` releases predate the established release process and are consolidated in the [v0.3.x summary](v0.3.md); starting with `0.4.0`, each version has its own page with the full list of fixes, behavior-change notes, and backward-compatibility impact.

## Release Policy

- **Patch versions (`0.x.Y`)**: pure hardening releases with zero API removals and zero numerical differences on healthy inputs. Batches are labeled by severity: HIGH (H/N), MEDIUM (M/N), LOW (L/N).
- **Minor versions (`0.X.0`)**: include at least one **behavior change** (a flipped default, a change in return-value semantics, or a changed required field). Any breaking default must first land behind an opt-in parameter in the previous patch release, and only then be flipped in the next minor (see the `cross_vars` pattern from 0.3.19 → 0.4.0, and the planned flip of the 0.4.2 `missing_policy` default).
- **Major versions (`X.0.0`)**: major architectural changes. None are currently planned.

## Version Index

<div class="grid cards" markdown>
- :material-tag: **[v0.9.0](v0.9.0.md)** - Three defaults flip (`unseen_special_policy='neutral'`, `sv_total_basis='all'`, `warm_start_score_scope='full'`); every text is English and every public docstring documents its parameters; the defects of the documentation verification and of the end-to-end audit of both pipelines are fixed (batch-mode selection over all batches, pooled tables without the stand-in OOT, consistent weighted and unweighted IV, evaluation and output-directory fixes)

- :material-tag: **[v0.8.2](v0.8.2.md)** - Seven defect batches: `unseen_special_policy` for declared-but-unseen special values, warnings visible at import again, SMF's own warning noise and the bugs behind it, single-class evaluation, bool features in binning/WOE/FVP, and no more process-wide pandas option changes or writes into the caller's DataFrame

- :material-tag: **[v0.8.1](v0.8.1.md)** - By-group WOE chart IV fixed to in-group IV (special bins, fitted SV decisions, unsmoothed single-class cells excluded), SV decisions in Format-A attrs, sparse bin-id alignment, dict-wrapped `load_woe_bins`, and joblib ≥ 1.6 compatibility for `ParallelApplyEngine`

- :material-tag: **[v0.8.0](v0.8.0.md)** - SV bin governance: `sv_min_bin_size` / `sv_small_policy` low-share fallback and `sv_woe_smoothing` / `sv_smoothing_alpha` bad-rate shrinkage on both WOE engines, defaults bit-for-bit identical to 0.7.2

- :material-tag: **[v0.7.2](v0.7.2.md)** - 0.7.1 boundary hardening: categorical small-bin fit governance, verified Format-A round-trips, frozen FVP transform audits, weighted VIF, and constant-weight corr parity

- :material-tag: **[v0.7.1](v0.7.1.md)** - First 0.7.x release (0.7.0 burned pre-release): default-governance flip, categorical VIF support, and the unweighted-corr categorical crash fix with mixed WOE matrix basis

- :material-tag: **[v0.6.9](v0.6.9.md)** - Categorical transform guards (missing-drift + str-fallback tripwires, coverage stats) and CMP self-fit param split

- :material-tag: **[v0.6.8](v0.6.8.md)** - Weighted-path audit closure: hard-leak IV upper band on all paths, weighted G03/G04 evidence, floored G05/G06 ranking

- :material-tag: **[v0.6.7](v0.6.7.md)** - Selection & binning governance: WOE ordering, IV/VIF/group gates, LR p-value elimination, verifiable split materialization

- :material-tag: **[v0.6.6](v0.6.6.md)** - Evaluation governance: multi-label, special bins, direction, eval weights

- :material-tag: **[v0.6.5](v0.6.5.md)** - OOT governance: candidate mode, split gates, no silent OOT synthesis

- :material-tag: **[v0.6.4](v0.6.4.md)** - ODPS proc means, weighted evaluation figures, and wider grouped WOE charts

- :material-tag: **[v0.6.3](v0.6.3.md)** - Close-out of equal_freq WOE-plot fix: `if group:` guard hoisted in `get_bivar_graph`

- :material-tag: **[v0.6.2](v0.6.2.md)** - Fix equal_freq WOE plots silently producing zero PNGs

- :material-tag: **[v0.6.1](v0.6.1.md)** - Independent FeatureValidation plot output control

- :material-tag: **[v0.6.0](v0.6.0.md)** - Vectorized wide-table analysis and bounded-memory execution

- :material-tag: **[v0.5.9](v0.5.9.md)** - Custom evaluation splits and FVP handoff metadata

- :material-tag: **[v0.5.8](v0.5.8.md)** — ODPS pull-probe diagnostics

- :material-tag: **[v0.5.7](v0.5.7.md)** — Categorical screening and RI lr NaN handling

- :material-tag: **[v0.5.6](v0.5.6.md)** — Pipeline and data IO correctness

- :material-tag: **[v0.5.5](v0.5.5.md)** — Parallel ODPS row-number pull

- :material-tag: **[v0.5.4](v0.5.4.md)** — Pipeline GUI schema metadata

- :material-tag: **[v0.5.3](v0.5.3.md)** — Cross-module hardening

    0.6.0 spec bug batch shipped on the 0.5.x line: UAT aggressive coercion, prediction NaN stats, all-zero weight guards, ODPS partition atomic swap, RI NaN-score policy, evaluator duplicate protection, unified evaluate return type, safer AUC failures, and NaN handling for chi2/VIF.

- :material-tag: **[v0.5.1](v0.5.1.md)** — Feature / WOE hygiene

    Weighted IV zero-mass guards, PSI one-sided bucket smoothing, safer WOE missing sentinel, clearer WOE empty-dict errors, and per-variable failure diagnostics.

- :material-tag: **[v0.5.0](v0.5.0.md)** — 2026-07-07

    MEDIUM batch — **PSI `missing_policy` default flipped (breaking)**, 6 upstream public APIs now expose the kwarg, and `MonotoneWOEBinner.apply_woe` gains `unseen_category_policy` and `_unseen_category_stats`

- :material-tag: **[v0.4.2](v0.4.2.md)** — 2026-07-06

    HIGH hotfix batch — atomic ODPS upload, `split_df` `exclude_cols` semantics fix, `HardCutoffInferrer` NaN guard, `Weighted_Screen` missing bin, new `PSI_Tool.missing_policy` parameter

- :material-tag: **[v0.4.1](v0.4.1.md)** — 2026-07-06

    LOW hygiene batch — NaN handling for the default reject-inference cutoff, `predict_positive` shape/length/finite validation, and `feature_validation` stability for mixed Interval+NaN groupby

- :material-tag: **[v0.4.0](v0.4.0.md)** — 2026-07-06

    MEDIUM hygiene batch #2 — `cross_vars` default change (**breaking**), opt-in numeric coercion for object columns, vectorized correlation computation, sample-analysis dry-run, and two more items

- :material-history: **[v0.3.x Summary](v0.3.md)** — before 2026-06-30

    0.3.4 → 0.3.18, covering the WOE binning engine, unified `feature_screen`, weighted screen, `FeatureScreeningArtifact`, Pipeline hardening batches H1-H4, and hygiene batches M2-M4

</div>

## Related Links

- [SMF main repo Releases](https://github.com/Kyle-J-Sun/SuperModelingFactory/releases)
- [Full tag list](https://github.com/Kyle-J-Sun/SuperModelingFactory/tags)
- API breaking changes and behavior flips are called out explicitly on the corresponding version page; search for "**breaking change**" or "**behavior change**".
