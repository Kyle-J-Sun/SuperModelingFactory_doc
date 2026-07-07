# SuperModelingFactory

> **面向信用评分卡开发的端到端 Python 建模工具链**

SuperModelingFactory 整合了信贷风控建模全流程所需的三大能力：

| 子项目 | 功能定位 | 核心能力 |
|--------|---------|---------|
| **[Modeling_Tool](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/Modeling_Tool)** | 建模引擎 | 数据分箱、WOE 编码、特征分析、模型训练与评估、样本管理 |
| **[ExcelMaster](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/ExcelMaster)** | 报告引擎 | 程序化 Excel 工作簿生成，支持图表、条件格式、光标流式写入 |
| **[Report](https://github.com/Kyle-J-Sun/SuperModelingFactory/tree/main/Report)** | 报告模板 | 模型性能报告、WOE 图批量导出、多模型对比报告 |

---

## 它能帮你做什么

!!! tip "典型场景"

    - 从一张行为评分样本出发，**5 分钟产出评分卡训练样本**
    - 用 **WOE / IV / PSI** 做特征筛选与稳定性监控
    - 训练 **逻辑回归 / LightGBM / XGBoost / CatBoost** 模型并自动出 Gains / ROC / KS 报告
    - 支持**样本权重**训练与评估（`weight_col` / `sample_weight`，按余额/过采样校正等场景）
    - 处理**拒绝推断**与**分布偏移**问题
    - 通过 **ExcelMaster** 一键导出格式化的中文建模报告
    - 用 **UAT 模块**做线上线下分数一致性校验

---

## 快速一览

=== "样本切分"

    ```python
    from Modeling_Tool import SampleSplitter
    splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
    train_df, test_df = splitter.split_df(data, target='bad_flag')
    ```

=== "WOE 编码"

    ```python
    from Modeling_Tool import WOE_Master
    woe = WOE_Master(train_data=train_df, varlist=features, dep='bad_flag')
    woe.fit(nbins=10, equal_freq=True)
    train_woe = woe.transform(train_df)
    test_woe  = woe.transform(test_df)
    ```

=== "模型训练"

    ```python
    from Modeling_Tool import GradientBoostingModel
    model = GradientBoostingModel('lgb', {'n_estimators': 200, 'learning_rate': 0.05})
    model.fit(train_woe[features], train_woe['bad_flag'],
              test_woe[features],  test_woe['bad_flag'])
    ```

=== "Excel 报告"

    ```python
    from ExcelMaster.ExcelMaster import ExcelMaster
    em = ExcelMaster('model_report.xlsx', verbose=False)
    ws = em.add_worksheet('Performance')
    em.write_dataframe(ws, perf, title='模型性能', titleformat='BLUE_H2')
    em.insert_image(ws, 'roc_curve.png', figScale=(600, 400))
    em.close_workbook()
    ```

---

## 文档导航

<div class="grid cards" markdown>

- :material-rocket-launch: **[快速上手](quickstart.md)**

    5 分钟跑通你的第一个评分卡训练流水线。

- :material-package-variant: **[安装](installation.md)**

    核心依赖、可选依赖、MaxCompute 接入。

- :material-graph: **[架构](architecture.md)**

    模块依赖图、设计原则、命名规范。

- :material-pipe: **[端到端流水线](pipeline.md)**

    从样本切分到 Excel 报告的完整建模流程。

- :material-book-open-variant: **[用户指南](guides/index.md)**

    按场景分册：样本 / WOE / 特征 / 模型 / 评估 / UAT / 报告。

- :material-api: **[API 参考](api/index.md)**

    所有公开类、方法、函数的详细说明。

</div>

---

## 适用对象

- **信贷风控建模师**：开发 A 卡、B 卡、C 卡、反欺诈模型
- **模型验证 / 审计**：UAT 一致性、PSI 监控、变量解释性
- **数据科学家**：复用分箱 / WOE / 后向消元 / 拒绝推断等模块
- **建模平台开发**：把 SuperModelingFactory 作为底层库进行二次封装

---

## 版本

- **Version**: 0.5.2
- **Author**: Jingkai Sun
- **License**: [Business Source License 1.1](https://github.com/Kyle-J-Sun/SuperModelingFactory/blob/main/LICENSE)（2030-06-24 后转 Apache 2.0，商业使用须联系作者授权）

---

## 下一步

👉 [快速上手](quickstart.md) → [安装](installation.md) → [架构](architecture.md)
