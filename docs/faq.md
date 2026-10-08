# FAQ

This page collects frequently asked questions and solutions encountered while using SuperModelingFactory.

---

## Environment and Dependencies

### Why does `import Modeling_Tool` fail with `_ARRAY_API not found`?

**Symptom**

In a K8s (or other containerized) environment, running `from Modeling_Tool.Core import *` or `from config import *` produces the following error chain:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.x as it may crash. ...
AttributeError: _ARRAY_API not found
...
NameError: name 'exit' is not defined
```

**Root cause**

The environment has NumPy 2.x installed (for example 2.2.6), but dependencies such as `matplotlib` and `lightgbm` were compiled against the NumPy 1.x ABI.
The ABI is incompatible with the current NumPy major version, which crashes the whole import chain:

```
NumPy 2.x installed
  └─► matplotlib (compiled against NumPy 1.x) → _ARRAY_API not found
        └─► lightgbm/compat.py fails to import matplotlib
                    └─► GBM_Tool.py fails to import lightgbm
                          └─► initialization of the remaining Model submodules is aborted
```

**Solutions**

=== "Option 1: Downgrade NumPy (recommended)"

    ```bash
    pip install "numpy<2" --force-reinstall
    ```

    **Restart the Jupyter Kernel** afterward. This is the simplest and most reliable fix.

=== "Option 2: Upgrade the dependencies to NumPy 2.x-compatible versions"

    If your environment policy does not allow downgrading NumPy, upgrade `matplotlib` and `lightgbm` to versions compatible with NumPy 2.x:

    ```bash
    pip install "matplotlib>=3.9" "lightgbm>=4.3" --upgrade
    ```

    The main repository has switched to **pure-Python source distribution** (see [PR #24](https://github.com/Kyle-J-Sun/SuperModelingFactory/pull/24)),
    and no longer depends on precompiled `.so`/`.pyd` extensions; just install `numpy` / `lightgbm` according to the matrix constraints.

=== "Option 3: Import on demand, bypassing the Model module"

    If your current task (such as an inference workflow) does not need GBM/LR training, import only the necessary modules in `config.py`:

    ```python
    # Import only Core / Sample / Eval, skipping the Model module that triggers lightgbm
    from Modeling_Tool.Core import *
    from Modeling_Tool.Sample import *
    from Modeling_Tool.Eval import *
    # Do not run `from Modeling_Tool import *` or `from Modeling_Tool.Model import *`
    ```

**Affected versions**

| Package | Incompatible versions | Fixed versions |
|---|---|---|
| NumPy | ≥ 2.0 | < 2.0 (downgrade) |
| matplotlib | < 3.9 | ≥ 3.9 |
| lightgbm | < 4.3 | ≥ 4.3 |

---

### Why does the import fail with `AttributeError: module 'numpy' has no attribute 'float'`?

**Symptom**

NumPy has already been downgraded to 1.23.x, but running `from Modeling_Tool.Core import *` (or any import that triggers the top-level `Modeling_Tool`) still fails with:

```
File .../dask/array/numpy_compat.py, line 13
    if (not np.allclose(np.divide(.4, 1, casting="unsafe"),
                        np.divide(.4, 1, casting="unsafe", dtype=np.float)) ...
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`.
The aliases was originally deprecated in NumPy 1.20;
```

**Root cause**

`GBM_Tool.py` used to run `import lightgbm as lgb` at module top level, and `lightgbm/compat.py` tries an optional import of `dask.array`.
An **old dask in the environment (older than 2022, roughly `<2021.11`)** uses `np.float` in `dask/array/numpy_compat.py`.
That alias was deprecated in NumPy 1.20 and removed entirely in NumPy 1.24+.
Even with NumPy 1.23.x, this usage triggers an `AttributeError`, breaking every import that depends on `Modeling_Tool`.

The error chain:

```
from Modeling_Tool.Core import *
  └─► Modeling_Tool/__init__.py (old) → from .Model import ...
        └─► GBM_Tool.py → import lightgbm as lgb  (module level)
              └─► lightgbm/compat.py → from dask.array import ...
                    └─► dask/array/numpy_compat.py → np.float
                          └─► AttributeError: module 'numpy' has no attribute 'float'
```

**Fixed in**

This problem is fully fixed in the source by two changes:

1. **`GBM_Tool.py`**: the module-level `import lightgbm as lgb` / `import xgboost as xgb` were removed; lightgbm/xgboost are now lazy-loaded inside each function or method that uses them (through the `_get_lgb()` / `_get_xgb()` helpers).
2. **`Modeling_Tool/__init__.py`**: the top-level eager `from .Model import (...)` block was removed in favor of lazy loading through `__getattr__` (consistent with the existing `ODPSRunner` lazy-loading pattern).

After the fix, `import Modeling_Tool` or `from Modeling_Tool.Core import *` **no longer triggers** the lightgbm → dask import chain; lightgbm/xgboost are imported only when you actually call Model symbols such as `GradientBoostingModel` or `lgbm_quick_train`.

**Temporary workarounds (until the wheel is updated)**

If you are using a compiled wheel package (`/opt/conda/...`) rather than a source install, the source fix does not take effect until the package is rebuilt. Use any of the following to work around it in the meantime:

=== "Option 1: Uninstall the old dask (recommended)"

    ```bash
    pip uninstall dask -y
    ```

    The dask import in `lightgbm/compat.py` is an optional dependency wrapped in `try/except`; once dask is removed, lightgbm skips it and loads normally.
    This suits inference/scoring scenarios that do not depend on dask.

=== "Option 2: Upgrade dask"

    ```bash
    pip install "dask[array]>=2022.01" --upgrade
    ```

    dask fixed the `np.float` usage after 2021.11. Restart the kernel after upgrading.
    Note: upgrading dask can bring in many dependency changes, so test it in an isolated environment.

=== "Option 3: Import on demand, skipping the Model submodule"

    ```python
    # In config.py, import only modules that do not trigger lightgbm
    from Modeling_Tool.Core import *
    from Modeling_Tool.Sample import *
    from Modeling_Tool.Eval import *
    from Modeling_Tool.WOE import *
    from Modeling_Tool.Feature import *
    # Do not import Modeling_Tool.Model; inference scenarios usually do not need training features
    ```

**Affected version combinations**

| Condition | Notes |
|---|---|
| NumPy 1.20–1.23 + dask < 2021.11 | Triggers `AttributeError: np.float` (only a warning level in itself, but the old dask code turns it into a crash) |
| NumPy ≥ 1.24 + dask < 2021.11 | Same, but more severe (`np.float` has been removed entirely) |
| NumPy ≥ 1.24 + dask ≥ 2022.01 | Works |
| Source install of the latest SuperModelingFactory | Fixed; not affected |

### After upgrading to 0.8.2, why does my notebook show many more warnings?

**Cause**

In 0.8.1 and earlier, `Modeling_Tool.WOE.WOE_Monotone_Binner` (loaded as soon as you `import Modeling_Tool`) and `Modeling_Tool.Model.Backward_Tool` executed `warnings.filterwarnings("ignore")` at import time. That suppresses all warnings in the **entire Python process**: SMF's own guard warnings (such as special-bin governance downgrades and declared special values that never occur in the fit sample), warnings from your own code, and deprecation notices from third-party libraries all disappeared. 0.8.2 removed both global settings, and warnings are back to Python's default behavior. Computed results are identical to before; these messages are simply visible again.

Likewise, in 0.8.1 and earlier, `import Modeling_Tool` also executed `pd.set_option('future.no_silent_downcasting', True)` (one occurrence each in `Core/Binning_Tool.py`, `Core/kDataFrame.py`, `Core/Slope_Tool.py`, and `Core/ODPS_Tool.py`). This is also a **process-level** switch: once SMF was imported, unrelated `replace()` / `fillna()` calls in your own code changed their downcasting behavior and began emitting downcasting warnings. 0.8.2 removed all four occurrences; SMF itself does not depend on the switch (the full regression shows no downcasting warnings without it). If your code really needs it, set it explicitly in your own script.

The same batch also removed `pd.options.mode.chained_assignment = None` (previously set at import time in `Core/kDataFrame.py`, `Core/Binning_Tool.py`, `Core/ODPS_Tool.py`, `Core/Slope_Tool.py`, and `ExcelMaster/Template.py`, and set once more when `Report/Report_Tool.plot_woe` was called). What it turned off is pandas' `SettingWithCopyWarning` — **the alarm for real bugs of the "I modified a slice and the change didn't take effect" kind** — and because it was process-level, your own code lost it too.

It had been turned off because SMF itself triggered the warning in 9 places across 6 modules. 0.8.2 changed all 9 to operate on SMF's own frames, so the warning can stay on without SMF ever tripping it. **Resulting behavior changes**:

| Location | Before | 0.8.2 |
|---|---|---|
| `run_binning` / `super_binning` | Could write the `bin_num` / `bin_range` columns back into the DataFrame you passed in | Only in the return value; your table is untouched |
| `get_gains_table` (when `model` is passed) | Left a temporary `_mdl_scr` column on your table | No longer leaves it |
| `plot_woe` / `plot_woe_group` (**public API**) | Lowercased your column names and replaced inf in `woe`/`iv` in place | Your table is left as-is; the plots are unchanged |
| `scoring` / `select_sample_seed` | Only triggered the warning; the writes already landed on a copy | No change |

If your code depends on the old side effects in the first three rows (for example, looking up columns by lowercase name after calling `plot_woe`), do it explicitly yourself: `df.columns = [c.lower() for c in df.columns]`.

0.8.2 also cleaned up the deprecation and numerical notices triggered by SMF's own calls (seaborn `distplot` / `bw`, `log(0)` in WOE/IV, pandas `concat` / `groupby` FutureWarnings, sklearn feature names, xlsxwriter single-cell merges, pyodps `Schema`, shap global random state), and fixed the problems this exposed (see the 0.8.2 changelog). The main warnings you may still see are:

| Warning | Source | Notes |
|---|---|---|
| `declared special value(s) never occur in the fit sample`, `bins have zero-mass class`, `keep_all_warn`, etc. | SMF guard warnings | Your data or configuration needs attention; see the [WOE guide](guides/woe.md) and the [Feature guide](guides/feature.md) |
| sklearn `UndefinedMetricWarning` | Single-class datasets (such as an OOT sample with no bad samples) | Metrics such as AUC / KS are recorded as NaN for that dataset |
| sklearn notice that `sample_weight` applies only to the calibrator during calibration | Weighted calibration | A weighting-semantics notice; worth reading |
| xgboost `Parameters: { ... } are not used` | Model parameters | Parameters passed to the native training interface (such as `n_estimators`) took no effect |

**Silence only the warnings you don't want to see**

```python
import warnings

warnings.filterwarnings("ignore", message=r".*zero-mass class")   # by message regex (matched from the start of the message)
warnings.filterwarnings("ignore", category=FutureWarning)          # by category
```

Avoid the bare `warnings.filterwarnings("ignore")` again: it also silences SMF's guard warnings.

---

### Evaluation fails with `No module named 'IPython'` outside a notebook

`PerformanceEvaluator.evaluate`, `get_perf_summary`, and some other evaluation functions print their result table with
`IPython.display.display` when `display=True`, which is the default. IPython is not a declared dependency of SMF, so a plain
Python environment without it raises `ModuleNotFoundError`. Either pass `display=False` (the table is still returned), or
install it with `pip install ipython`.

---

## ODPS Access Key Configuration

### How do I manage ODPS credentials and package imports uniformly with `config.py`?

**Pain points**

When using `ODPSRunner` in a Jupyter notebook or script, two problems are common:

1. Every file repeats `os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = "..."` by hand, and an AccessKey can easily be committed to Git by mistake;
2. Every file repeats a long list of `from Modeling_Tool.XXX import *` lines, which is noisy and easy to get wrong by missing a submodule.

The recommended approach is to manage the AccessKey centrally in the **system-level shared path** `/opt/workspace/.env`, and keep a single `config.py` in the project root that loads it explicitly. Multiple projects can then share one AK instead of configuring each repository. Every notebook needs only one line at the top, `from config import *`, which both loads the credentials and imports the package.

---

!!! note "What `ODPSRunner` reads"

    `ODPSRunner()` takes no arguments. It reads `ALIBABA_CLOUD_ACCESS_KEY_ID` and `ALIBABA_CLOUD_ACCESS_KEY_SECRET` (both
    required: a missing one raises `KeyError`), plus `ODPS_PROJECT` and `ODPS_ENDPOINT`. If the last two are not set, it
    falls back to built-in defaults (project `mex_anls` and a Singapore VPC endpoint) that belong to the author's
    environment, so always set both. `pyodps` must be installed (`pip install 'supermodelingfactory[odps]'`).

---

**Step 1: Install python-dotenv**

The `Modeling_Tool` main package does not depend on `python-dotenv`, so install it separately once:

```bash
pip install python-dotenv
```

---

**Step 2: Create the system-level shared `.env`**

```bash
# Create the directory and hand it to the current user
sudo mkdir -p /opt/workspace
sudo chown $USER:$USER /opt/workspace

# Create the file and lock down its permissions (readable/writable only by the current user)
touch /opt/workspace/.env
chmod 600 /opt/workspace/.env
```

Then write the credentials with an editor:

```bash
# /opt/workspace/.env  —— one copy shared by all projects
ALIBABA_CLOUD_ACCESS_KEY_ID=LTAI5tXXXXXXXXXXXXXXXXXX
ALIBABA_CLOUD_ACCESS_KEY_SECRET=YYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
ODPS_PROJECT=mex_anls
ODPS_ENDPOINT=http://service.cn-shanghai.maxcompute.aliyun.com/api
```

!!! warning "Security reminder"
    - `/opt/workspace/.env` is outside the repository and cannot be tracked by Git — but you **must** `chmod 600` it to stop other users on the same machine from reading it;
    - Do not share the `.env` through Slack, email, or screenshots — use a secrets-management service instead (Vault, Alibaba Cloud KMS, etc.);
    - On a multi-user server, consider `~/.config/smf/.env` instead to avoid cross-user leaks.

---

**Step 3: `.gitignore` in the project root (a precaution)**

Although `.env` is not in the repository, it is still a good idea to add the following lines, so that if you ever copy `.env` into the project root by accident, it is not committed:

```gitignore
# Credential files (precaution)
.env
.env.local
.env.*.local
```

---

**Step 4: Create `config.py` in the project root**

```python
# ═══════════════════════════════════════════════════════════════════════════
# config.py — shared project imports + environment variable loading
# ═══════════════════════════════════════════════════════════════════════════
# Usage: write one line at the top of any notebook / script:
#
#     from config import *
#
# which does both of the following:
#   1. Loads the ODPS / Aliyun credentials from the system-level shared path
#      /opt/workspace/.env into os.environ
#   2. Imports all Modeling_Tool submodules into the current namespace
#
# After editing .env, restart the Jupyter kernel for the change to take effect.
# /opt/workspace/.env must be chmod 600 so other users on the machine cannot read it.
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
from pathlib import Path

# ── Load the system-level shared .env ──
from dotenv import load_dotenv

ENV_PATH = Path("/opt/workspace/.env")

if ENV_PATH.is_file():
    # override=False: variables already in the environment (e.g. injected by K8s) take priority over .env
    load_dotenv(ENV_PATH, override=False)
else:
    print(f"[config] WARNING: {ENV_PATH} not found; ODPS credentials may be missing.",
          file=sys.stderr)

# ── Import all Modeling_Tool submodules ──
#   Core    — basic utilities: DateTimeUtils, load_model, save_model, calc_iv, calc_woe ...
#   Sample  — sample splitting: SampleSplitter, StratifiedSampler, SampleBalancer
#   Eval    — model evaluation: PerformanceEvaluator, GainsTableCalculator
#   Feature — feature analysis: proc_means_by_grp, PSICalculator, CorrelationFilter
#   WOE     — WOE binning: WOE_Master, MonotoneWOEBinner, woe_transform
#   Model   — model training: LRMaster, GradientBoostingModel, BackwardVariableEliminator
import Modeling_Tool as smf  # namespace backup: smf.Model.LRMaster stays available if an import * collides

from Modeling_Tool.Core import *      # noqa: F401,F403
from Modeling_Tool.Sample import *    # noqa: F401,F403
from Modeling_Tool.Eval import *      # noqa: F401,F403
from Modeling_Tool.Feature import *   # noqa: F401,F403
from Modeling_Tool.WOE import *       # noqa: F401,F403
from Modeling_Tool.Model import *     # noqa: F401,F403
```

---

**Step 5: Use it in a notebook**

```python
# check: skip   (needs your own config.py and ODPS credentials)
# First cell of the notebook
from config import *

# At this point:
#   1. os.environ already holds ALIBABA_CLOUD_ACCESS_KEY_ID / _SECRET / ODPS_PROJECT / ODPS_ENDPOINT
#   2. All SMF classes such as LRMaster / WOE_Master / PerformanceEvaluator are ready to use

odps = ODPSRunner()                            # no arguments; credentials are read from os.environ
df   = odps.run_sql("SELECT * FROM ... LIMIT 100")      # returns a DataFrame (to_df=True by default)

woe  = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")   # use directly, no extra import needed
lr   = LRMaster(params={"C": 1.0})
```

---

**Key design notes**

=== "Why `/opt/workspace/.env`"

    Keeping the file at `/opt/workspace/.env` and loading it by **absolute path** solves three problems at once:

    1. **Multiple projects share one AK** — no need to paste a `.env` into every repository, and rotating credentials means changing one place;
    2. **It cannot be committed by accident with a repo push** — it is not under a Git working tree at all;
    3. **Easy to debug** — the path is hard-coded, so when something goes wrong you can see at a glance which credentials were loaded; Jupyter startup does not rely on the unreliable `__file__` or CWD.

=== "Why `override=False`"

    `override=False` (the default) means that variables already present in `os.environ` are **not overwritten by .env**.
    In containerized deployments (K8s, Docker, CI/CD), this ensures environment variables injected by operations always take priority over the local `.env` — otherwise local credentials could overwrite them by accident at release time.

=== "Why keep `import Modeling_Tool as smf`"

    Six `from ... import *` statements carry a naming-collision risk (a submodule imported later silently overrides same-named symbols imported earlier).
    Keeping an explicit `smf` namespace as a fallback lets you write `smf.Model.LRMaster` to pin down the exact source when a collision happens.

=== "Why dotenv is not built into the SMF main package"

    `python-dotenv` is a project-level engineering convention, not a modeling-tool responsibility. The SMF main package only does "use these keys if they are in `os.environ`"; how they get into `os.environ` (dotenv / K8s Secret / startup script / manual export) is entirely up to the project.

---

**Common pitfalls**

| Symptom | Cause | Fix |
|---|---|---|
| After `from config import *`, `os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]` is still empty | `/opt/workspace/.env` does not exist or the path is wrong | Run `ls -la /opt/workspace/.env` to confirm the file exists; check that `ENV_PATH` in `config.py` is correct |
| `PermissionError: [Errno 13]` when reading `.env` | After `chmod 600`, the current user is not the owner | Check the owner with `ls -la /opt/workspace/.env`; run `sudo chown $USER:$USER /opt/workspace/.env` |
| Credentials did not update after editing `.env` | The Jupyter kernel has cached `os.environ` | Restart the kernel (menu: Kernel → Restart) |
| `ImportError: No module named 'dotenv'` | `python-dotenv` is not installed | `pip install python-dotenv` |
| The correct credentials were overwritten by `.env` after deploying to K8s | `override=True` was used | Switch back to `override=False` (the recommended default) |
| One notebook calls several data sources and the wrong AK is used | `os.environ` is process-level global state | For multi-account scenarios in one process, pass explicit arguments such as `ODPS(access_id=..., secret_access_key=...)` |

---

## Sample Weights

### Which keyword passes sample weights: `weight_col` or `sample_weight`?

Typical scenarios: correcting sampling bias (giving original samples higher weight after oversampling), weighting by loan balance/amount, time-decay weighting, and so on.

| Argument | Where | Notes |
|---|---|---|
| `weight_col` | `LRMaster` (`fit`, `stepwise_selection`, `get_aic`, `get_bic`, `calibrate_model`), `PerformanceEvaluator`, `GainsTableCalculator`, `Model_Evaluation_Tool`, `cross_risk`, `BackwardVariableEliminator` | Name of a column of the DataFrame you pass |
| `sample_weight` | `GradientBoostingModel.fit` and the single-backend classes, and the array-based evaluation functions (`calc_roc`, `calc_pr`, `calc_equid_*`, `calc_lift_apt`, the `'sample_weight'` key of `evaluate_performance` datasets) | A 1-D array with one weight per row |

Weights must be finite, non-negative, one per row, and sum to more than 0. APIs that accept both forms (for example
`cross_risk`) raise `ValueError` if you pass both; `wgt` and `wgt_col` are accepted as aliases there. `LRMaster.fit`
takes only `weight_col`.

If you trained with weights but forget to pass them at evaluation time, metrics are computed with equal weights (1 per row), which is inconsistent with the training objective. Pass the same weights consistently through the training and evaluation chain.

For the argument each API takes and the weighted metric semantics (`N` vs `N_RAW`, weighted AUC/KS/Lift), see [Model Training: Sample Weights](guides/model.md#sample-weights) and [Model Evaluation: Sample-Weighted Evaluation](guides/eval.md#sample-weighted-evaluation).

---

## Known Issues

There are no open defects at the moment. The issues that the documentation audit and the end-to-end audit of the two pipelines found have been fixed; the [Unreleased notes](changelog/unreleased.md) list them one by one, with the results that change.

A few behaviors are documented instead of changed, because changing them would move numbers that existing models rely on. They are listed in [Known Issues](changelog/unreleased.md#6-known-issues) with the parameter each one concerns: the totals used by special-value bins, what the warm-start prior does and does not reach, `perf_min_bin_prop` as a target, files that stay in a reused `output_dir`, and the per-batch selection of the CSV batch mode.

---

*If you have other questions, feel free to submit them in [GitHub Issues](https://github.com/Kyle-J-Sun/SuperModelingFactory/issues).*
