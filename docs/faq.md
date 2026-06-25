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

*如有其他问题，欢迎在 [GitHub Issues](https://github.com/Kyle-J-Sun/SuperModelingFactory/issues) 中提交。*
