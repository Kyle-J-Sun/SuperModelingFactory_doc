# Installation

## Quick Install

```bash
pip install supermodelingfactory
```

macOS users also need to install the OpenMP runtime (required by lightgbm):

```bash
brew install libomp
```

## Supported Environments

| Operating system | Architecture | Python |
|---|---|---|
| macOS 11+ | arm64 (Apple Silicon) | 3.10 / 3.11 / 3.12 / 3.13 |
| Linux | x86_64 (manylinux_2_28) | 3.10 / 3.11 / 3.12 / 3.13 |
| Windows | x86_64 | 3.10 / 3.11 / 3.12 / 3.13 |

!!! note "Other platforms"
    SMF ships plain Python source, so no compiler is needed. On platforms without a prebuilt wheel (for example an Intel Mac),
    `pip` installs from the source distribution.

## Optional Dependencies

Features that need extra packages are kept behind extras:

| Extra | Install | Enables |
|---|---|---|
| `odps` | `pip install 'supermodelingfactory[odps]'` | Alibaba Cloud MaxCompute: `ODPSRunner`, `proc_means_odps`, `ParallelODPSManager` |
| `explain` | `pip install 'supermodelingfactory[explain]'` | `ModelExplainer`: SHAP, Owen value, LIME |
| `stats` | `pip install 'supermodelingfactory[stats]'` | `statsmodels`-based VIF gates and LR diagnostics |
| `imblearn` | `pip install 'supermodelingfactory[imblearn]'` | SMOTE and imbalanced-learn samplers |
| `optuna` | `pip install 'supermodelingfactory[optuna]'` | Optuna search in `GradientBoostingModel.param_search` and the pipelines |
| `mic` | `pip install 'supermodelingfactory[mic]'` | MIC correlation in `build_coalition_structure` (Python < 3.11 only) |

Combine extras with commas: `pip install 'supermodelingfactory[explain,stats,optuna]'`.

!!! note "Notebook display"
    `PerformanceEvaluator.evaluate()` prints its table with `IPython.display` by default. In a plain script pass
    `display=False`, or `pip install ipython`.

## Developer / Test Environment Dependencies

If you plan to run the full [`SuperModelingFactory_pytest`](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest) suite locally, install its `requirements-dev.txt`:

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest.git
cd SuperModelingFactory_pytest
pip install -r requirements-dev.txt
```

That file declares **packages that are not runtime dependencies of the main package but are required by the test suite** (`pyodps`, `shap`, `lime`, `statsmodels`, `PyYAML`). Without them, dozens of tests are silently skipped, which masks real regressions. Target baseline: **0 skipped / 0 failed**.

## Upgrade

```bash
pip install --upgrade supermodelingfactory
```

## Verify the Installation

```bash
python -c "
from Modeling_Tool import WOE_Master, LRMaster, PSICalculator
import Modeling_Tool
print('SMF version:', Modeling_Tool.__version__)
print('OK')
"
```

## Previewing the Documentation Site Locally

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_doc.git
cd SuperModelingFactory_doc
pip install -r requirements-docs.txt
mkdocs serve    # open http://127.0.0.1:8000 in your browser
```

## FAQ

??? question "`OSError: Library not loaded: @rpath/libomp.dylib` (macOS)"

    The OpenMP runtime is missing. Run:

    ```bash
    brew install libomp
    ```

??? question "`Bad CPU type in executable` (Apple Silicon)"

    The current Python interpreter is the Intel build. Switch to an arm64 Python installed via Homebrew or conda, and verify it with:

    ```bash
    python -c "import platform; print(platform.machine())"  # expected output: arm64
    ```

??? question "`ModuleNotFoundError: No module named 'odps'`"

    Install the ODPS extra dependencies:

    ```bash
    pip install 'supermodelingfactory[odps]'
    ```

??? question "`ImportError` on an unsupported Python version or platform"

    Check your environment against the supported list above and reinstall:

    ```bash
    pip debug --verbose
    ```

??? question "Chinese characters in plots appear as empty boxes"

    Install a CJK font, then refresh the font cache:

    ```bash
    # Linux
    sudo apt install fonts-noto-cjk fonts-wqy-zenhei
    fc-cache -fv
    ```
