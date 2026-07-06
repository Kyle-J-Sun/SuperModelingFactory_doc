# SMF 0.4.1 — LOW-severity defensive hardening

Patch release. Three defensive fixes, no interface changes, no behavior
changes on healthy inputs. All three make previously-silent failure modes
loud, so callers hear about them instead of shipping bad numbers.

## Fixes

### L1 — `RejectInferencePipeline._default_hard_cutoff` (reject_inference.py)

**Before:** When both bad-sample scores and the overall score column were
entirely NaN, `_default_hard_cutoff` returned `float(NaN)`. That NaN then
propagated into every `score < cutoff` comparison downstream, producing
a completely empty inferred rejected set with no diagnostic.

**After:**

- If the fallback median is not finite, raise `ValueError` with an
  actionable message pointing to `ri_method_params['hard_cutoff']['cutoff']`.
- If bad samples have no valid scores but the overall median is finite,
  emit a `RuntimeWarning` and continue with the median as cutoff.
- Healthy path (bad samples with valid scores) is unchanged.

### L2 — `predict_positive` shape/length/finiteness validation (_common.py)

**Before:** Model output shapes were partially handled: 2-D `(N, 2)` took
column 1, everything else was reshape-flattened. Silent failures included
row-count mismatches (model dropped rows), unexpected 2-D shapes with
3+ columns, and all-NaN predictions.

**After:**

- Validate `ndim ∈ {1, 2}` and 2-D column count `∈ {1, 2}`. Raise `ValueError`
  otherwise.
- Assert output length matches input row count. Raise `ValueError` on
  mismatch with a message identifying the model class.
- Warn (RuntimeWarning) when any prediction is NaN or Inf, reporting the
  count and model class.

### L3 — `_ivks_from_binner` mixed `Interval + NaN` groupby (feature_validation.py)

**Before:** The IV/KS-from-binner report path called
`groupby("bin", dropna=False)` on a mixed `Interval + NaN` key column.
Under pandas <2.0 the default `sort=True` on that mixed dtype raises
`TypeError: '<' not supported between instances of 'Interval' and 'float'`.
Under pandas ≥2.0 it works but relies on implementation detail.

**After:**

- Cast bins to `object` dtype and replace NaN with the explicit sentinel
  `"__MISSING__"` before grouping.
- Set `sort=False` to make the code order-independent.
- Behavior on healthy pandas ≥2.0 is numerically identical (same bin
  counts, same IV, same KS) — only the sort order of the output rows may
  differ, and downstream code re-sorts by `bad_rate` anyway.

## Non-changes

- No public API surface change.
- No new config field.
- No default value change.
- All 0.4.0 regression tests continue to pass (49 old + 10 new = 59 green).

## Upgrade

```bash
pip install --upgrade supermodelingfactory==0.4.1
```

No code changes required for existing pipelines. Two new callable
behaviors to be aware of:

- If you were relying on `_default_hard_cutoff` returning NaN in
  degenerate cases (unlikely — nobody was), that now raises.
- If your model wrapper drops rows silently inside `predict_proba`, you
  will now see an explicit `ValueError` at scoring time instead of a
  shape-mismatched score column later.

## Test coverage

`test_pipeline_low_0401.py` (10 tests, 3 classes):

- `TestL1DefaultHardCutoffNaNPropagation` — 3 tests (all-NaN raises,
  bad-only-NaN warns, healthy path silent).
- `TestL2PredictPositiveShapeValidation` — 5 tests (1-D pass-through,
  2-D positive-column extraction, length mismatch raises, unexpected
  shape raises, NaN predictions warn).
- `TestL3IVKSFromBinnerMixedBinTypes` — 2 tests (end-to-end
  mixed-Interval+NaN via monkeypatched adapter; direct sentinel
  behavior).
