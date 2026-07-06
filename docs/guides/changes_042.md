# SMF 0.4.2 — HIGH-severity hotfix

Patch release. Five independent HIGH-severity fixes across the ODPS, sample,
reject-inference, and feature-screening layers. All five turn a silent
data-corruption or numeric-inflation path into either an explicit error,
a warning, or an opt-in corrected computation. **No numeric changes on
healthy inputs; no signature breaks; the default output of
`_calculate_single_psi` is deliberately unchanged.**

## Summary

| Fix | Symbol | Severity | Impact |
|-----|--------|----------|--------|
| N11 | `Core.ODPS_Tool.ODPSRunner.upload_df` | HIGH | Atomic table replacement via temp + rename swap |
| N16 | `Sample.Sample_Split.SampleSplitter.split_df` | HIGH | `exclude_cols` columns are now preserved on output frames |
| N17 | `Sample.Reject_Infer.HardCutoffInferrer.infer` | HIGH | NaN-scored rejects no longer silently labelled `0` |
| N23 | `Feature.Weighted_Screen._weighted_bin_distribution` | HIGH | Removed forced `__MISSING__` injection + clip on non-observed bins |
| N29 | `Feature.PSI_Tool._calculate_single_psi` | HIGH | New `missing_policy` parameter to capture missing-rate drift |

## Fixes

### N11 — `ODPSRunner.upload_df` was non-atomic (Core/ODPS_Tool.py)

**Before:** The write path unconditionally ran

```python
self.o.delete_table(table_name, if_exists=True)
t = self.o.create_table(table_name, table_schema)
with t.open_writer(...) as writer:
    writer.write(df.values.tolist())
```

Any exception between `delete_table` and the completed `writer.write` —
network drop, quota rejection, schema error surfacing at the writer, or a
process kill — left the caller with a **dropped-and-empty target table**.
Downstream jobs then read the empty table as if it were a legitimate update.

**After (default `atomic=True`):**

1. Write the data into a temp table `<table>__tmp_<ts>`.
2. If the target table exists, rename it aside to `<table>__old_<ts>`
   (via `ALTER TABLE ... RENAME TO ...`).
3. Rename the temp table to the target name.
4. Drop the moved-aside old copy (best-effort, warn on failure since the
   swap already succeeded).

If step 1 fails, the temp table is dropped and the exception is re-raised —
**the live target table is never touched**. If step 2 fails, the temp table
is dropped and the original is left in place. If step 3 fails after a
successful step 2, the code attempts to rename `<table>__old_<ts>` back to
the target name so the caller is not left with a missing table; if that
restore itself fails, an ``ERROR``-level log names the surviving `__old_`
table for manual recovery.

Pass `atomic=False` explicitly to restore the pre-0.4.2 behaviour when the
caller needs the old object-identity semantics.

### N16 — `SampleSplitter.split_df` silently dropped `exclude_cols` (Sample/Sample_Split.py)

**Before:** `exclude_cols` in the docstring reads as "columns to exclude
from split," but the implementation

```python
feature_cols = [c for c in df.columns if c not in exclude_cols + [target]]
X = df[feature_cols]
```

physically dropped those columns from the split's feature matrix. After
concatenating the target back on, the returned `train_df` / `test_df` were
missing every column in `exclude_cols`. Common casualties: `user_id`,
`apply_date`, weight columns, ID columns needed for downstream joins.

**After:** The split runs on the DataFrame's index; both output frames
carry the full column set (features, target, and any `exclude_cols`).
`exclude_cols` is now the "carry-through, do not use as split key" hint
the name implies.

```python
train_idx, test_idx = train_test_split(
    df.index,
    test_size=test_size,
    random_state=self.random_state,
    stratify=strat,
)
train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()
```

**Downstream:** All internal callers of `split_df` in `Pipeline/` code
never passed `exclude_cols`, so no pipeline behaviour changes. External
callers that were unknowingly relying on the drop side-effect will now
see the previously-lost columns in their outputs — usually the desired
behaviour, since dropping them was the bug.

### N17 — `HardCutoffInferrer.infer` silently labelled NaN-scored rejects as good (Sample/Reject_Infer.py)

**Before:** The scoring logic

```python
if self.score_direction == "high_bad":
    df_rejected_copy[self.target_col] = (df_rejected_copy[score_col] >= self.cutoff).astype(int)
else:
    df_rejected_copy[self.target_col] = (df_rejected_copy[score_col] <= self.cutoff).astype(int)
```

used the raw comparison result. In pandas, `NaN >= cutoff` and
`NaN <= cutoff` both evaluate to `False`, which `.astype(int)` then cast to
`0`. Every NaN-scored reject was silently labelled "good" — biasing the
inferred training set with as many fabricated goods as there were NaN
scores.

**After:**

- If **every** reject row has a NaN score → `ValueError` with the missing
  count and an actionable hint pointing at the prescore step.
- If **some** reject rows have NaN scores → `RuntimeWarning` naming the
  count and share; those rows carry `target = NaN` in the returned frame
  (not `0`). Callers that intersect on the target column will drop them
  explicitly instead of training on silently-fabricated labels.
- Rejects with finite scores are labelled exactly as before.

### N23 — `_weighted_bin_distribution` inflated PSI on asymmetric-missing inputs (Feature/Weighted_Screen.py)

**Before:** The helper returned

```python
return dist.reindex(dist.index.union([_MISSING_BIN]), fill_value=0.0).clip(lower=content)
```

Two things happened here that should not have. First, `_MISSING_BIN` was
unconditionally union'd into the index of the returned distribution, even
when no NaN rows were observed. Second, every bin (observed or not) was
clipped up to `content` (a small positive floor, e.g. 1e-5).

When only one side of a PSI comparison had missing values, this
manufactured a positive mass on the missing bin of the side that observed
zero missings — which then propagated through `_psi_from_distributions`
as a large spurious PSI component, even though `_psi_from_distributions`
itself already runs `.reindex(all_bins, fill_value=content).clip(lower=content)`.
The result was a double-clipped, one-sided-inflated PSI for any feature
with a real missing-rate asymmetry.

**After:** `_weighted_bin_distribution` returns only the bins actually
observed in `bins`. Alignment and the zero-safety floor are done exactly
once, symmetrically on both sides, inside `_psi_from_distributions`.
The `content` parameter is retained on the helper's signature for
backward compatibility but is no longer used inside it.

### N29 — `_calculate_single_psi` was blind to missing-rate drift (Feature/PSI_Tool.py)

**Before:** The function opened with

```python
expected_clean = expected_series.dropna()
actual_clean = actual_series.dropna()
```

and normalized every subsequent count by the NaN-dropped denominators.
The single most common early signal of a data-pipeline break — the
missing rate on a feature suddenly changing from 5% to 30% — was
completely invisible in the returned PSI.

**After:** New `missing_policy` parameter:

- `"drop"` (default in 0.4.2) — pre-0.4.2 behaviour, exact numeric
  backward compatibility. The default remains here for one release so
  callers who compare PSI numbers to prior runs do not see a numeric
  shift. **The next minor release will flip this to `"include"`.**
- `"include"` — NaN rows are held out of the binning fit (so breakpoints
  remain finite) but re-attached as a dedicated `"__MISSING__"` bin on
  both sides before the PSI sum. Recommended for any production drift
  monitor. Denominators are the *original* row counts, so the missing
  bin's fraction is directly comparable across sides.
- `"warn_and_drop"` — same numeric answer as `"drop"` but emits a
  `RuntimeWarning` naming both NaN counts so the caller sees what was
  dropped.

Empirical impact on a real 5% → 30% missing-rate shift:

- `"drop"` PSI ≈ 0.007 (silently blind)
- `"include"` PSI ≈ 0.53 (drift correctly flagged)

## Durable lessons

- **`delete_table` before `create_table` is a non-atomic replacement.** Any
  write path that starts with "drop the target" is a footgun the moment
  the write step can fail. Temp table + rename is the correct primitive
  in a MaxCompute/Hive-family engine because both engines expose
  `ALTER TABLE ... RENAME` as a first-class operation.
- **A parameter name that suggests "carry through" must not physically
  drop the column.** The pattern `X = df[feature_cols]` looks harmless
  but silently amputates the split output. When the semantics are
  "select for computation, preserve for output," split on the index and
  reindex the full frame.
- **`NaN >= cutoff` returns `False`, and `.astype(int)` casts that to
  `0`.** Any code that compares a numeric column against a threshold and
  immediately casts to int needs an explicit NaN handling policy up
  front — otherwise NaN rows silently become "the falsy side" of the
  comparison, whatever that means for the caller's semantics.
- **A distribution helper that returns "the observed distribution plus
  a floor-clipped synthetic entry" is asymmetric by construction.** The
  floor is a property of the *comparison*, not of the individual
  distribution — apply it exactly once, at the comparison site, on both
  sides symmetrically.
- **`dropna()` at the top of a drift monitor is invisible bias.** Any
  metric whose purpose is to detect changes in the population must at
  least measure the missing-rate change — a `missing_policy` opt-in is
  the minimum, but the flip to opt-out (i.e. default-include) is where
  it belongs long-term.

## Migration

None required for the default path. To opt in to the corrected numeric
behaviour where the default was kept for compatibility:

```python
# PSI calculation with missing-rate drift captured (recommended for prod)
_calculate_single_psi(expected, actual, missing_policy="include")

# Or via the class API
PSICalculator(...).calculate(..., missing_policy="include")
```

Behaviour flips coming in the next minor release:

- `_calculate_single_psi(missing_policy=...)` default will change from
  `"drop"` to `"include"`. Pin `missing_policy="drop"` explicitly if you
  need the pre-0.4.2 numeric answer beyond that release.
