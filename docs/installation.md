# 安装

## 环境要求

- **Python** >= 3.9
- **操作系统**：Linux / macOS / Windows 均可（ODPS 相关功能需要 Linux）
- **磁盘**：约 200 MB（含可选依赖）

## 核心依赖

运行任何子项目都需要这些依赖：

```bash
pip install pandas>=1.5.0 numpy>=1.23.0 scipy>=1.9.0
```

## 按需安装

=== "Modeling_Tool 建模引擎"

    ```bash
    pip install scikit-learn>=1.1.0 joblib>=1.2.0 python-dateutil>=2.8.0
    pip install lightgbm>=2.3.0 xgboost>=1.7.0
    pip install matplotlib>=3.5.0 seaborn>=0.12.0
    pip install statsmodels>=0.13.0 tqdm>=4.64.0
    ```

=== "ExcelMaster / Report 报告引擎"

    ```bash
    pip install xlsxwriter>=3.0.0 openpyxl>=3.0.0 Pillow>=9.0.0
    ```

=== "可选：Optuna 超参搜索"

    ```bash
    pip install optuna>=4.0.0
    ```

    WOE 单调分箱器（`MonotoneWOEBinner`）使用 Optuna 调参与合并。

=== "可选：SMOTE 不平衡采样"

    ```bash
    pip install imbalanced-learn>=0.10.0
    ```

    `SampleBalancer` 的 SMOTE 模式依赖此包。

=== "可选：阿里云 MaxCompute"

    ```bash
    pip install pyodps>=0.11.0
    ```

    `ODPSRunner` 与 `Check_DuckDB_Compatibility` 依赖此包。

=== "可选：H2O 平台"

    ```bash
    pip install h2o>=3.40.0
    ```

    `Core/utils.py` 中预留了 H2O 辅助函数接口，按需启用。

## 一键安装

如果只是本地试用：

```bash
git clone <repo-url>
cd SuperModelingFactory
pip install -r requirements.txt       # 一次性安装所有核心 + 可选依赖
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

!!! info "`PYTHONPATH` 为何重要"

    SuperModelingFactory 各子包之间通过**绝对包路径**互相 import
    （如 `from Modeling_Tool.Core.utils import ...`）。
    必须把仓库根目录加入 `PYTHONPATH`，否则子包找不到同级的 `Modeling_Tool`。

    建议在项目根目录创建 `.envrc`（direnv）或 `.venv/bin/activate` 脚本中
    预设该环境变量。

## 文档站点本地预览

```bash
cd ../SuperModelingFactory-docs
pip install -r requirements-docs.txt
mkdocs serve    # 浏览器打开 http://127.0.0.1:8000
```

## 验证安装

```python
import pandas as pd
from Modeling_Tool import (
    SampleSplitter, WOE_Master, PSICalculator,
    GradientBoostingModel, LRMaster, PerformanceEvaluator,
)

# 全部能成功 import 即表示依赖齐全
print("OK")
```

## 常见问题

??? question "`ImportError: No module named Modeling_Tool.Core`"

    把仓库根目录加入 `PYTHONPATH`：

    ```bash
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    ```

??? question "`AttributeError: module 'pandas' has no attribute 'set_option'`"

    `pandas` 版本过低。请升级至 `>= 1.5.0`：

    ```bash
    pip install -U "pandas>=1.5.0"
    ```

??? question "中文字体显示为方框"

    安装 matplotlib 后端 + 中文字体：

    ```bash
    # Linux
    sudo apt install fonts-noto-cjk fonts-wqy-zenhei

    # 或从项目自带的 ref_font/ 目录手动安装
    cp ../SuperModelingFactory/Modeling_Tool/ref_font/*.ttf ~/.local/share/fonts/
    fc-cache -fv
    ```

??? question "`Multi_class='deprecated'` 报错"

    sklearn 1.5+ 的 LogisticRegression 在 `get_params()` 时会把历史参数
    返回为 `'deprecated'`，再次 fit 会失败。`lr_model()` 已做兼容处理，
    若仍报错请把 `multi_class` 显式设为 `'auto'` 或降级到 1.4.x。
