# 安装

## 快速安装

```bash
pip install supermodelingfactory
```

macOS 用户需要额外安装 OpenMP 运行时（lightgbm 依赖）：

```bash
brew install libomp
```

## 支持的环境

| 操作系统 | 架构 | Python |
|---|---|---|
| macOS 11+ | arm64（Apple Silicon） | 3.10 / 3.11 / 3.12 / 3.13 |
| Linux | x86_64（manylinux_2_28） | 3.10 / 3.11 / 3.12 / 3.13 |
| Windows | x86_64 | 3.10 / 3.11 / 3.12 / 3.13 |

!!! warning "Intel Mac 用户"
    PyPI 目前只提供 Apple Silicon (arm64) 的预编译 wheel。Intel Mac 安装时会回退到 sdist，需要本地 Cython 编译环境：
    ```bash
    pip install cython
    pip install supermodelingfactory
    ```

## 可选依赖

阿里云 MaxCompute / ODPS 集成需额外安装：

```bash
pip install 'supermodelingfactory[odps]'
```

## 开发者 / 测试环境依赖

若你打算在本地运行 [`SuperModelingFactory_pytest`](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest) 全量套件，需安装其 `requirements-dev.txt`：

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest.git
cd SuperModelingFactory_pytest
pip install -r requirements-dev.txt
```

该文件声明了三个 **不属于主包运行时依赖但测试套件依赖的包**（`pyodps`、`shap`、`lime`）。不安装它们会导致约 74 个测试静默 skip，掩盖真实回归。目标基线：**0 skipped / 0 failed**。

## 升级

```bash
pip install --upgrade supermodelingfactory
```

## 验证安装

```bash
python -c "
from Modeling_Tool import WOE_Master, LRMaster, PSICalculator
import Modeling_Tool
print('SMF version:', Modeling_Tool.__version__)
print('OK')
"
```

## 文档站点本地预览

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_doc.git
cd SuperModelingFactory_doc
pip install -r requirements-docs.txt
mkdocs serve    # 浏览器打开 http://127.0.0.1:8000
```

## 常见问题

??? question "`OSError: Library not loaded: @rpath/libomp.dylib`（macOS）"

    缺少 OpenMP 运行时，执行：

    ```bash
    brew install libomp
    ```

??? question "`Bad CPU type in executable`（Apple Silicon）"

    当前 Python 解释器是 Intel 版本。请改用 Homebrew 或 conda 安装的 arm64 Python，验证方法：

    ```bash
    python -c "import platform; print(platform.machine())"  # 期望输出 arm64
    ```

??? question "`ModuleNotFoundError: No module named 'odps'`"

    安装 ODPS 额外依赖：

    ```bash
    pip install 'supermodelingfactory[odps]'
    ```

??? question "`ImportError` from a closed-source module"

    当前 Python 版本或操作系统不在支持列表中。请确认环境后重新安装：

    ```bash
    pip debug --verbose
    ```

??? question "中文字体显示为方框"

    安装中文字体后刷新字体缓存：

    ```bash
    # Linux
    sudo apt install fonts-noto-cjk fonts-wqy-zenhei
    fc-cache -fv
    ```
