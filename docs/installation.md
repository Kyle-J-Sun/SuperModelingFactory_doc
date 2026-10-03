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

!!! warning "Intel Mac users"
    PyPI currently provides prebuilt wheels only for Apple Silicon (arm64). On an Intel Mac, installation falls back to the sdist and requires a local Cython build environment:
    ```bash
    pip install cython
    pip install supermodelingfactory
    ```

## Optional Dependencies

Alibaba Cloud MaxCompute / ODPS integration requires an extra install:

```bash
pip install 'supermodelingfactory[odps]'
```

## Developer / Test Environment Dependencies

If you plan to run the full [`SuperModelingFactory_pytest`](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest) suite locally, install its `requirements-dev.txt`:

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest.git
cd SuperModelingFactory_pytest
pip install -r requirements-dev.txt
```

That file declares three **packages that are not runtime dependencies of the main package but are required by the test suite** (`pyodps`, `shap`, `lime`). Without them, about 74 tests are silently skipped, which masks real regressions. Target baseline: **0 skipped / 0 failed**.

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

??? question "`ImportError` from a closed-source module"

    The current Python version or operating system is not in the supported list. Confirm your environment and reinstall:

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
