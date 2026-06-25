# 常见问题 FAQ

本页收录使用 SuperModelingFactory 过程中的高频问题与解决方案。

---

## 环境与依赖

### Q1: 导入包时报 `_ARRAY_API not found` / `NameError: name 'exit' is not defined`

**现象**

在 K8s（或其他容器化）环境中执行 `from Modeling_Tool.Core import *` 或 `from config import *` 时，出现以下错误链：

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.x as it may crash. ...
AttributeError: _ARRAY_API not found
...
NameError: name 'exit' is not defined
```

**根本原因**

环境中安装了 NumPy 2.x（如 2.2.6），但 `matplotlib`、`lightgbm` 等依赖是用 NumPy 1.x ABI 编译的。
`Modeling_Tool` 的 `.so`/`.pyd` 闭源扩展模块同样基于 NumPy 1.x 编译。
两者 ABI 不兼容，导致整条 import 链崩溃：

```
NumPy 2.x 安装
  └─► matplotlib (NumPy 1.x 编译) → _ARRAY_API not found
        └─► lightgbm/compat.py 导入 matplotlib 失败
              └─► GBM_Tool.py 导入 lightgbm 失败
                    └─► Backward_Tool .so 初始化中断
                          └─► NameError: 'exit' is not defined
```

**解决方案**

=== "方案一：降级 NumPy（推荐）"

    ```bash
    pip install "numpy<2" --force-reinstall
    ```

    执行后**重启 Jupyter Kernel**。这是最简单、最稳妥的方法，与 `Modeling_Tool` 所有模块完全兼容。

=== "方案二：升级依赖到支持 NumPy 2.x 的版本"

    若环境策略不允许降级 NumPy，则需升级 `matplotlib` 和 `lightgbm`：

    ```bash
    pip install "matplotlib>=3.9" "lightgbm>=4.3" --upgrade
    ```

    > ⚠️ 注意：`Modeling_Tool` 的 `.so` 扩展本身也需要使用 NumPy 2.x 重新编译的版本才能完整支持，请联系维护者获取对应构建。

=== "方案三：按需导入，绕过 Model 模块"

    若当前任务（如推理工作流）不需要 GBM/LR 训练功能，可在 `config.py` 中只导入必要模块：

    ```python
    # 只导入 Core / Sample / Eval，跳过触发 lightgbm 的 Model 模块
    from Modeling_Tool.Core import *
    from Modeling_Tool.Sample import *
    from Modeling_Tool.Eval import *
    # 不执行 from Modeling_Tool import *  或  from Modeling_Tool.Model import *
    ```

**受影响的版本**

| 包 | 不兼容版本 | 修复版本 |
|---|---|---|
| NumPy | ≥ 2.0 | < 2.0（降级） |
| matplotlib | < 3.9 | ≥ 3.9 |
| lightgbm | < 4.3 | ≥ 4.3 |

---

### Q2: 导入包时报 `AttributeError: module 'numpy' has no attribute 'float'`

**现象**

已将 NumPy 降级至 1.23.x，但执行 `from Modeling_Tool.Core import *`（或任何触发顶层 `Modeling_Tool` 的导入）时，仍报：

```
File .../dask/array/numpy_compat.py, line 13
    if (not np.allclose(np.divide(.4, 1, casting="unsafe"),
                        np.divide(.4, 1, casting="unsafe", dtype=np.float)) ...
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`.
The aliases was originally deprecated in NumPy 1.20;
```

**根本原因**

`GBM_Tool.py` 原先在模块顶层执行 `import lightgbm as lgb`，而 `lightgbm/compat.py` 会尝试可选导入 `dask.array`。
环境中的 **旧版 dask（早于 2022 年，约 `<2021.11`）** 在 `dask/array/numpy_compat.py` 中使用了 `np.float`，
该别名在 NumPy 1.20 被废弃、在 NumPy 1.24+ 彻底移除。
即使使用 NumPy 1.23.x，这一写法也会触发 `AttributeError`，导致所有依赖 `Modeling_Tool` 的 import 失败。

错误链：

```
from Modeling_Tool.Core import *
  └─► Modeling_Tool/__init__.py (旧版) → from .Model import ...
        └─► GBM_Tool.py → import lightgbm as lgb  (模块级)
              └─► lightgbm/compat.py → from dask.array import ...
                    └─► dask/array/numpy_compat.py → np.float
                          └─► AttributeError: module 'numpy' has no attribute 'float'
```

**已修复版本**

此问题已在源码中通过以下两处修改彻底修复（参见 commit `b0038ac` / `f92505b`）：

1. **`GBM_Tool.py`**：移除模块级 `import lightgbm as lgb` / `import xgboost as xgb`，改为在每个用到 lightgbm/xgboost 的函数/方法体内部懒加载（通过 `_get_lgb()` / `_get_xgb()` 辅助函数）。
2. **`Modeling_Tool/__init__.py`**：移除顶层的 `from .Model import (...)` eager 导入块，改用 `__getattr__` 延迟加载（与现有的 `ODPSRunner` 懒加载模式一致）。

修复后，`import Modeling_Tool` 或 `from Modeling_Tool.Core import *` **不再触发** lightgbm → dask 的导入链，只有真正调用 `GradientBoostingModel`、`lgbm_quick_train` 等 Model 符号时才会 import lightgbm/xgboost。

**临时绕过方案（等待 wheel 更新）**

如果你使用的是已编译的 wheel 包（`/opt/conda/...`）而非源码安装，源码修复在重新打包前不会生效。可使用以下任一方案临时绕过：

=== "方案一：卸载旧版 dask（推荐）"

    ```bash
    pip uninstall dask -y
    ```

    `lightgbm/compat.py` 对 dask 的导入是 `try/except` 包裹的可选依赖，卸掉 dask 后 lightgbm 会跳过它，正常加载。
    适用于不依赖 dask 的推理/评分场景。

=== "方案二：升级 dask"

    ```bash
    pip install "dask[array]>=2022.01" --upgrade
    ```

    dask 在 2021.11 之后修复了 `np.float` 用法。升级后重启 kernel。
    注意：dask 升级可能带入较多依赖变更，建议在独立环境中测试。

=== "方案三：按需导入，跳过 Model 子模块"

    ```python
    # config.py 中只导入不触发 lightgbm 的模块
    from Modeling_Tool.Core import *
    from Modeling_Tool.Sample import *
    from Modeling_Tool.Eval import *
    from Modeling_Tool.WOE import *
    from Modeling_Tool.Feature import *
    # 不导入 Modeling_Tool.Model，推理场景通常不需要训练功能
    ```

**受影响的版本组合**

| 条件 | 说明 |
|---|---|
| NumPy 1.20–1.23 + dask < 2021.11 | 触发 `AttributeError: np.float`（警告级，但旧 dask 写法使其崩溃） |
| NumPy ≥ 1.24 + dask < 2021.11 | 同上，但更严重（`np.float` 已彻底移除） |
| NumPy ≥ 1.24 + dask ≥ 2022.01 | 正常 |
| 源码安装最新版 SuperModelingFactory | 已修复，不受影响 |

---

*如有其他问题，欢迎在 [GitHub Issues](https://github.com/Kyle-J-Sun/SuperModelingFactory/issues) 中提交。*
