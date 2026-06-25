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

## ODPS 访问密钥配置

### Q3: 如何用 `config.py` 统一管理 ODPS 凭据和包导入

**痛点**

在 Jupyter notebook 或脚本里使用 `ODPSRunner` 时，常见两个问题：

1. 每个文件都要重复手写 `os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = "..."`，AccessKey 容易被误提交到 Git；
2. 每个文件都要重复一长串 `from Modeling_Tool.XXX import *`，noisy 且容易漏掉子模块。

推荐的做法是在**系统级共享路径** `/opt/workspace/.env` 集中管理 AccessKey，项目根目录只放一个 `config.py` 显式加载它。这样多个项目可以共用一份 AK，不用每个仓库都重复配置。所有 notebook 顶端只写一行 `from config import *`，即同时完成凭据加载和包导入。

---

**步骤 1：安装 python-dotenv**

`Modeling_Tool` 主包不依赖 `python-dotenv`，需要单独装一次：

```bash
pip install python-dotenv
```

---

**步骤 2：创建系统级共享 `.env`**

```bash
# 创建目录并交给当前用户
sudo mkdir -p /opt/workspace
sudo chown $USER:$USER /opt/workspace

# 创建文件并锁死权限（只有当前用户可读写）
touch /opt/workspace/.env
chmod 600 /opt/workspace/.env
```

然后用编辑器写入凭据：

```bash
# /opt/workspace/.env  —— 所有项目共享一份
ALIBABA_CLOUD_ACCESS_KEY_ID=LTAI5tXXXXXXXXXXXXXXXXXX
ALIBABA_CLOUD_ACCESS_KEY_SECRET=YYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
ODPS_PROJECT=mex_anls
ODPS_ENDPOINT=http://service.cn-shanghai.maxcompute.aliyun.com/api
```

!!! warning "安全提醒"
    - `/opt/workspace/.env` 在仓库之外，不可能被 Git 跟踪 —— 但**必须** `chmod 600` 防止同机器其他用户读取；
    - 不要把 `.env` 通过 Slack/邮件/截图分享 —— 改用密钥管理服务（Vault、阿里云 KMS 等）；
    - 如果是多用户服务器，考虑改放 `~/.config/smf/.env`，避免跨用户泄露。

---

**步骤 3：项目根目录的 `.gitignore`（防范性措施）**

虽然 `.env` 不在仓库里，但仍建议加上以下行，防止未来某天手滑把 `.env` 复制到项目根后误提交：

```gitignore
# 凭据文件（防范性）
.env
.env.local
.env.*.local
```

---

**步骤 4：在项目根目录新建 `config.py`**

```python
# ═══════════════════════════════════════════════════════════════════════════
# config.py — 项目通用导入 + 环境变量加载
# ═══════════════════════════════════════════════════════════════════════════
# 用法：在任意 notebook / 脚本顶端写一行：
#
#     from config import *
#
# 即同时完成：
#   1. 从系统级共享路径 /opt/workspace/.env 加载 ODPS / Aliyun 凭据到 os.environ
#   2. 导入 Modeling_Tool 全部子模块到当前命名空间
#
# 修改 .env 后需重启 Jupyter kernel 才能生效。
# /opt/workspace/.env 必须 chmod 600，防止同机器其他用户读取。
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
from pathlib import Path

# ── 加载系统级共享 .env ──
from dotenv import load_dotenv

ENV_PATH = Path("/opt/workspace/.env")

if ENV_PATH.is_file():
    # override=False: 已在环境中的变量（如 K8s 注入）优先于 .env
    load_dotenv(ENV_PATH, override=False)
else:
    print(f"[config] WARNING: {ENV_PATH} not found; ODPS credentials may be missing.",
          file=sys.stderr)

# ── 导入 Modeling_Tool 全部子模块 ──
#   Core    — 基础工具: DateTimeUtils, load_model, save_model, calc_iv, calc_woe ...
#   Sample  — 样本拆分: SampleSplitter, StratifiedSampler, SampleBalancer
#   Eval    — 模型评估: PerformanceEvaluator, GainsTableCalculator
#   Feature — 特征分析: proc_means_by_grp, PSICalculator, CorrelationFilter
#   WOE     — WOE 分箱: WOE_Master, MonotoneWOEBinner, woe_transform
#   Model   — 模型训练: LRMaster, GradientBoostingModel, BackwardVariableEliminator
import Modeling_Tool as smf  # 命名空间备份，避免 import * 冲突时仍可用 smf.Model.LRMaster

from Modeling_Tool.Core import *      # noqa: F401,F403
from Modeling_Tool.Sample import *    # noqa: F401,F403
from Modeling_Tool.Eval import *      # noqa: F401,F403
from Modeling_Tool.Feature import *   # noqa: F401,F403
from Modeling_Tool.WOE import *       # noqa: F401,F403
from Modeling_Tool.Model import *     # noqa: F401,F403
```

---

**步骤 5：在 notebook 里使用**

```python
# notebook 第一个 cell
from config import *

# 此时已经完成：
#   1. os.environ 中已加载 ALIBABA_CLOUD_ACCESS_KEY_ID / _SECRET / ODPS_PROJECT / ODPS_ENDPOINT
#   2. LRMaster / WOE_Master / PerformanceEvaluator 等全部 SMF 类可直接使用

odps = ODPSRunner()                            # 无需再传 access_key，自动从 os.environ 读
df   = odps.read_sql("SELECT * FROM ... LIMIT 100")

woe  = WOE_Master(...)                          # 直接使用，无需 from Modeling_Tool.WOE import *
lr   = LRMaster(params={"C": 1.0})
```

---

**关键设计说明**

=== "为什么放在 `/opt/workspace/.env`"

    集中放在 `/opt/workspace/.env` 用**绝对路径**加载，一下子解决三个问题：

    1. **多项目共用一份 AK** —— 不用每个仓库都贴一份 `.env`，轮转凭据时只需改一处；
    2. **不可能随仓库 push 被误提交** —— 根本不在 Git 工作区下；
    3. **调试友好** —— 路径是硬编码的，出问题时一眼能看出加载的是哪份凭据；Jupyter 启动时不依赖不可靠的 `__file__` 或 CWD。

=== "为什么用 `override=False`"

    `override=False`（默认值）表示：已经存在于 `os.environ` 的变量**不会被 .env 覆盖**。
    这是为了在容器化部署（K8s、Docker、CI/CD）时，运维注入的环境变量始终优先于本地 `.env` —— 否则上线时容易被本地凭据意外覆盖。

=== "为什么保留 `import Modeling_Tool as smf`"

    六次 `from ... import *` 存在命名冲突风险（后导入的子模块会静默覆盖前面的同名符号）。
    保留一份显式命名空间 `smf` 作为 fallback：当遇到命名冲突时，可直接写 `smf.Model.LRMaster` 来精确指定来源。

=== "为什么不在 SMF 主包里集成 dotenv"

    `python-dotenv` 是项目级工程约定，不是建模工具职责。SMF 主包只负责"如果 `os.environ` 里有这些 key 就用它们"，至于这些 key 怎么进入 `os.environ`（dotenv / K8s Secret / 启动脚本 / 手动 export）完全由项目自己决定。

---

**常见踩坑**

| 现象 | 原因 | 解决 |
|---|---|---|
| `from config import *` 后 `os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]` 仍为空 | `/opt/workspace/.env` 不存在或路径拼错 | `ls -la /opt/workspace/.env` 确认文件存在；检查 `config.py` 中 `ENV_PATH` 路径是否正确 |
| `PermissionError: [Errno 13]` 读不了 `.env` | `chmod 600` 后当前用户不是 owner | `ls -la /opt/workspace/.env` 检查拥有者；`sudo chown $USER:$USER /opt/workspace/.env` |
| 修改 `.env` 后凭据没更新 | Jupyter kernel 已缓存 `os.environ` | 重启 kernel（菜单 Kernel → Restart） |
| `ImportError: No module named 'dotenv'` | 没装 `python-dotenv` | `pip install python-dotenv` |
| 部署到 K8s 后被 `.env` 覆盖了正确的凭据 | 使用了 `override=True` | 改回 `override=False`（推荐默认值） |
| 一个 notebook 同时调多个数据源，AK 用错了 | `os.environ` 是进程级全局状态 | 同进程内多账号场景请直接用 `ODPS(access_id=..., secret_access_key=...)` 显式参数 |

---

*如有其他问题，欢迎在 [GitHub Issues](https://github.com/Kyle-J-Sun/SuperModelingFactory/issues) 中提交。*
