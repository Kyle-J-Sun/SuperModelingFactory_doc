# 架构

## 顶层结构

```
SuperModelingFactory/
├── Modeling_Tool/          # 核心建模引擎（≈ 21k 行 Python）
│   ├── Core/               # 基础设施：分箱、ODPS、工具、加密、JSON
│   ├── WOE/                # WOE 编码：分箱、变换、绘图、单调合并
│   ├── Feature/            # 特征分析：PSI、IV、相关性、分布
│   ├── Model/              # 模型训练：LR / LightGBM / XGBoost / 后向消元
│   ├── Eval/               # 模型评估：Gains、ROC、KS、对比、流水线
│   ├── Sample/             # 样本管理：切分、分层、拒绝推断、分布适配
│   └── UAT/                # 线上线下一致性校验
│
├── ExcelMaster/            # Excel 报告引擎（≈ 2.1k 行）
│   ├── ExcelFormatTool.py  # 50+ 预设单元格格式
│   ├── ExcelMaster.py      # 核心引擎（光标流式、图表、条件格式）
│   ├── Template.py         # PVA / Bivar / GridSearch 等模板
│   └── Utility.py          # 路径 / 颜色 / PSI 处理
│
└── Report/                 # 报告模板（≈ 380 行）
    └── Report_Tool.py      # 单/多模型报告 + WOE 批量图
```

## 模块依赖图

```
                ┌─────────┐
                │  Core   │  (基础设施，无跨包依赖)
                └────┬────┘
       ┌─────────┬───┼───────┬─────────┬──────────┐
       ▼         ▼   ▼       ▼         ▼          ▼
     WOE      Model  Eval  Feature  Sample    UAT
       │         │              │        │        │
       └─────────┴──────────────┴────────┴────────┘
           (均单向依赖 Core，模块间延迟 import)

                        │
                        ▼
                  ExcelMaster (独立引擎)
                        ▲
                        │
                      Report (消费 ExcelMaster)
```

### 依赖原则

1. **Core 是叶子节点** —— Core 不导入任何子包，子包之间通过**函数体内 import**（延迟导入）打破循环。
2. **顶层 `Modeling_Tool/__init__.py` 精选统一 API** —— 用户只需 `from Modeling_Tool import ...`，无需关心子包结构。
3. **ExcelMaster 完全独立** —— 只依赖 `xlsxwriter` + `pandas`，可单独抽离使用。
4. **Report 消费 ExcelMaster** —— 不做实际建模，只编排"已计算好的 CSV / PNG → Excel"。

## 模块职责

### Core — 基础设施

| 文件 | 职责 |
|------|------|
| `Binning_Tool.py` | 等频 / 等距 / 卡方 / 决策树分箱；自动满足最小占比 |
| `ODPS_Tool.py` | 阿里云 MaxCompute 客户端（执行 SQL 并行转 DataFrame） |
| `utils.py` | 杂项：`save_model`、`load_model`、`scoring`、`calc_woe/iv`、Vintage 工具 |
| `kDataFrame.py` | pandas 扩展（`odds_score`, `scale_score`） |
| `Slope_Tool.py` | 线性回归斜率（sklearn / scipy / numpy / manual 4 种实现） |
| `XOR_Encryptor.py` | XOR + Base64 文本加解密（支持整 DataFrame） |
| `Json_Data_Converter.py` | CDC 征信数据 DataFrame ↔ JSON 双向转换 |
| `Check_DuckDB_Compatibility.py` | 扫描 SQL 文件的 DuckDB 兼容性 |
| `test_remove_comments.py` | SQL 注释移除的 pytest 单元测试 |

### WOE — 证据权重编码

| 文件 | 职责 |
|------|------|
| `WOE_Master.py` | 全流程主控：fit → transform → 整体 / 分组 WOE 表 |
| `WOE_Tool.py` | 单变量 / 多变量 WOE 转换、单调性检验 |
| `WOE_Plot_Tool.py` | 单变量 / 双变量绑图、整体 / 分组 WOE 表汇总 |
| `WOE_Monotone_Binner.py` | **贪心单调 WOE 分箱器**（卡方初始化 + 类别聚类 + Optuna + 并行） |
| `WOE_Report_Builder.py` | WOE 批量图 Excel 报告（增强版） |
| `plot_woe_tool.py` | 分组指标提取、PSI 表计算 |

### Feature — 特征分析

| 文件 | 职责 |
|------|------|
| `PSI_Tool.py` | 群体稳定性指数（单变量 / 多变量 / 数据集内） |
| `Feature_Insights.py` | IV 计算、WOE 分箱、自动绑图、相关性过滤 |
| `Distribution_Tool.py` | 分布偏移检测、KDE / 直方图 / 地毯图 |

### Model — 模型训练

| 文件 | 职责 |
|------|------|
| `GBM_Tool.py` | LightGBM / XGBoost 统一接口 + 校准 + 变量重要性 |
| `LRM_Tool.py` | 逻辑回归 + statsmodels 摘要 + AIC/BIC + 逐步选择 |
| `Backward_Tool.py` | 后向变量消元（基于累计重要性阈值） |

### Eval — 模型评估

| 文件 | 职责 |
|------|------|
| `Model_Eval_Tool.py` | Gains 表、性能汇总、交叉风险 |
| `Evaluation_Tool.py` | 链式 `EvaluationPipeline`（`.group_by().subset_by().apply()`） |
| `evaluate_model.py` | 单 / 多模型 ROC、KS、PR、KDE、PCT、Gain 绘图 |

### Sample — 样本管理

| 文件 | 职责 |
|------|------|
| `Sample_Split.py` | 样本切分、分层采样、SMOTE 均衡、最优种子搜索 |
| `Reject_Infer.py` | 拒绝推断（Hard-Cut / Fuzzy / Parceling / SimpleAugment） |
| `Distribution_Adaptation.py` | 密度比 / KL / 协变量偏移三种样本加权 |

### UAT — 一致性校验

| 文件 | 职责 |
|------|------|
| `UAT_Consistency_Checker.py` | 线上线下分数 / 数据一致性校验，输出容差对比 Excel |

## 命名规范

- **公开 API**：通过 `__init__.py` 导出，使用方只需 `from Modeling_Tool import ...`
- **类名**：PascalCase（如 `WOE_Master`, `PerformanceEvaluator`）
- **函数名**：snake_case（如 `woe_transform`, `calculate_slope_sklearn`）
- **私有成员**：以 `_` 开头（如 `_patch_wide_schema_download`），不保证稳定接口
- **常量**：全大写（部分历史代码使用）

## 单文件规模 Top 10

| 排名 | 文件 | 行数 |
|------|------|------|
| 1 | `WOE/WOE_Monotone_Binner.py` | 3324 |
| 2 | `Core/utils.py` | 2645 |
| 3 | `Eval/Model_Eval_Tool.py` | 2341 |
| 4 | `Eval/evaluate_model.py` | 1945 |
| 5 | `Model/Backward_Tool.py` | 1485 |
| 6 | `Eval/Evaluation_Tool.py` | 1451 |
| 7 | `Model/LRM_Tool.py` | 1529 |
| 8 | `Model/GBM_Tool.py` | 1348 |
| 9 | `Feature/PSI_Tool.py` | 1175 |
| 10 | `WOE/WOE_Tool.py` | 1094 |

合计 **≈ 31,300 行 Python**。
