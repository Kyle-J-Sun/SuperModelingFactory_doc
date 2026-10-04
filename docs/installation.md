# Installation

SuperModelingFactory (SMF) is a pure-Python package: it ships as one universal wheel plus a source distribution, so
installing it needs no compiler. The heavy lifting is done by its dependencies (NumPy, pandas, scikit-learn, LightGBM,
XGBoost, CatBoost), which pip installs for you.

## Install

```bash
pip install supermodelingfactory
```

The command installs three importable packages: `Modeling_Tool` (modeling engine), `ExcelMaster` (Excel writer), and
`Report` (report templates).

macOS users also need the OpenMP runtime, which LightGBM requires:

```bash
brew install libomp
```

## Requirements

| Requirement | Supported |
|---|---|
| Python | 3.10, 3.11, 3.12, 3.13 |
| NumPy | 1.23 or later, including 2.x |
| pandas | 1.5 or later, below 3.0 |
| Operating system | Any platform on which the dependencies install. The automated tests run on Linux |

The other runtime dependencies are installed automatically: SciPy, scikit-learn, joblib, python-dateutil, LightGBM,
XGBoost, CatBoost, matplotlib, seaborn, XlsxWriter, openpyxl, Pillow, and tqdm.

## Optional Dependencies

Features that need extra packages are kept behind extras:

| Extra | Install | Enables |
|---|---|---|
| `odps` | `pip install 'supermodelingfactory[odps]'` | Alibaba Cloud MaxCompute access: `ODPSRunner`, `ParallelODPSManager`, `proc_means_odps` |
| `explain` | `pip install 'supermodelingfactory[explain]'` | The SHAP, Owen value, and LIME methods of `ModelExplainer` (PDP, ICE, and ALE need no extra) |
| `stats` | `pip install 'supermodelingfactory[stats]'` | `statsmodels` for variance inflation factors: `FeatureSelectionAnalyzer.compute_vif`, `CorrelationFilter.calculate_vif`, and `FeatureScreenConfig(vif_enabled=True)` |
| `imblearn` | `pip install 'supermodelingfactory[imblearn]'` | imbalanced-learn samplers: `StratifiedSampler.balance(method='smote')`, and `SampleBalancer` with `method='nearmiss'`, `'tomek'`, or `'enn'` |
| `optuna` | `pip install 'supermodelingfactory[optuna]'` | Optuna search: `GradientBoostingModel.param_search(engine='optuna')` and the Optuna stage of `CreditModelPipeline` |
| `mic` | `pip install 'supermodelingfactory[mic]'` | MIC correlation in `build_coalition_structure(corr_method='mic')`. The extra installs `minepy` on Python below 3.11 only |

Combine extras with commas: `pip install 'supermodelingfactory[explain,stats,optuna]'`.

!!! note "Notebook display"
    `PerformanceEvaluator.evaluate()` and `get_perf_summary()` print their tables through `IPython.display` by default.
    In a plain script, pass `display=False`, or run `pip install ipython`.

## Verify the Installation

```bash
python -c "import Modeling_Tool; print(Modeling_Tool.__version__)"
```

The command prints `0.8.2` (or the release you installed). To check that the three packages import:

```bash
python -c "import Modeling_Tool, ExcelMaster, Report; print('OK')"
```

## Upgrade

```bash
pip install --upgrade supermodelingfactory
```

## Work from a Source Checkout

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory.git
cd SuperModelingFactory
pip install -e .
```

## Test Environment for Maintainers

The regression suite lives in the private `SuperModelingFactory_pytest` repository. Besides SMF's own dependencies, it
needs packages that are not runtime dependencies of SMF: `pyodps`, `shap`, `lime`, `statsmodels`, and `PyYAML`. They are
listed in the suite's `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

Without them, the tests that cover the optional features are skipped, which hides regressions. The release baseline is
0 skipped and 0 failed.

## Preview the Documentation Site

The API pages are generated from the SMF source tree, so check out the two repositories **side by side** with these exact
folder names:

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory.git
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_doc.git
cd SuperModelingFactory_doc
pip install -r requirements-docs.txt
mkdocs serve    # open http://127.0.0.1:8000 in your browser
```

## Troubleshooting

??? question "`No matching distribution found for supermodelingfactory`"

    pip prints `Requires-Python >=3.10` for every release when the interpreter is older than 3.10. Install SMF with
    Python 3.10 or later:

    ```bash
    python --version
    ```

??? question "`AttributeError: _ARRAY_API not found` when importing `Modeling_Tool`"

    A compiled dependency (usually matplotlib) was built for NumPy 1.x, but NumPy 2.x is installed. Upgrade the package
    named in the traceback, or pin NumPy below 2. See the [FAQ](faq.md#why-does-import-modeling_tool-fail-with-_array_api-not-found).

??? question "`ModuleNotFoundError: No module named 'odps'`"

    The ODPS extra is not installed. `ODPSRunner` needs it, and so do the star imports `from Modeling_Tool.Core import *`
    and `from Modeling_Tool import *`:

    ```bash
    pip install 'supermodelingfactory[odps]'
    ```

??? question "`ModuleNotFoundError: No module named 'IPython'` from an evaluation call"

    The call tried to display a table. Pass `display=False` (or `disp=False` for the `Model_Evaluation_Tool` methods), or
    install IPython. See the [FAQ](faq.md#evaluation-fails-with-no-module-named-ipython-outside-a-notebook).

??? question "`OSError: Library not loaded: @rpath/libomp.dylib` (macOS)"

    The OpenMP runtime is missing. Run:

    ```bash
    brew install libomp
    ```

??? question "`Bad CPU type in executable` (Apple Silicon)"

    The current Python interpreter is the Intel build. Switch to an arm64 Python installed via Homebrew or conda, and
    verify it with:

    ```bash
    python -c "import platform; print(platform.machine())"  # expected output: arm64
    ```

??? question "Non-Latin characters in plots appear as empty boxes"

    matplotlib's default font (DejaVu Sans) has no CJK glyphs, so labels in such scripts print `Glyph ... missing from font(s)
    DejaVu Sans` and are drawn as boxes.
    The WOE plots already use the KaiTi font that SMF bundles for their titles and tables; for every other figure, register
    a CJK font. The snippet below uses the bundled font, so no system font is needed:

    ```python
    import os

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    import Modeling_Tool

    font_path = os.path.join(os.path.dirname(Modeling_Tool.__file__), "ref_font", "KaiTi.ttf")
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    ```

    Run it once per session, before drawing. You can also install a system CJK font (for example
    `sudo apt install fonts-noto-cjk` on Debian or Ubuntu) and set `plt.rcParams["font.family"]` to its name.
