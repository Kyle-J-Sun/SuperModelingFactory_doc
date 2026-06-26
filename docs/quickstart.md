# 快速上手

5 分钟跑通一条完整的评分卡训练流水线。本节使用 **合成数据** 演示，不依赖任何外部数据源。

## 0. 准备

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# 假设当前在 SuperModelingFactory 仓库根目录
```

## 1. 造一份合成样本

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

data = pd.DataFrame({
    "user_id":    np.arange(n),
    "age":        rng.normal(35, 8, n).clip(18, 70),
    "income":     rng.lognormal(10, 0.4, n),
    "score_b":    rng.normal(600, 60, n),
    "city_grade": rng.choice(["A", "B", "C", "D"], n),
    "n_overdue":  rng.poisson(0.3, n),
})

# 合成一个与特征相关的坏样本率
logit = -6 + 0.02 * (data["score_b"] - 600) + 0.001 * (data["income"] - data["income"].mean())
prob = 1 / (1 + np.exp(-logit))
data["bad_flag"] = rng.binomial(1, prob)

print(data.head())
print("坏样本率：", data["bad_flag"].mean())
```

## 2. 样本切分

```python
from Modeling_Tool import SampleSplitter

features = ["age", "income", "score_b", "city_grade", "n_overdue"]
splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
train_df, test_df = splitter.split_df(data, target="bad_flag")
print(f"train={len(train_df)}  test={len(test_df)}")
```

## 3. WOE 编码

```python
from Modeling_Tool import WOE_Master

woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)

train_woe = woe.transform(train_df)
test_woe  = woe.transform(test_df)

print(train_woe[[f"{f}_woe" for f in features]].head())
```

## 4. 训练逻辑回归

```python
from Modeling_Tool import LRMaster

woe_features = [f"{f}_woe" for f in features]

lr = LRMaster(params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"})
# fit 接收 (data, varlist, tgt_name)
lr.fit(train_woe, woe_features, "bad_flag")

coef = lr.get_statsmodel_summary()
print(coef)
```

## 5. 训练 LightGBM

```python
from Modeling_Tool import GradientBoostingModel

gbm = GradientBoostingModel(
    "lgb",
    params={
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "early_stopping_rounds": 20,
        "eval_metric": "auc",
    },
)
gbm.fit(
    train_woe[woe_features], train_woe["bad_flag"],
    test_woe[woe_features],  test_woe["bad_flag"],
)
```

## 6. 模型评估

```python
from Modeling_Tool import PerformanceEvaluator

evaluator = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
)
evaluator.add_dataset("train", train_woe).add_dataset("test", test_woe)
perf = evaluator.evaluate()

print(perf[["index", "KS", "AUC", "Top10%_TargetRate"]])
```

## 7. 生成 Excel 报告

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("Performance")

em.merge_col(ws, ncols=5, text="LightGBM 模型性能汇总")
em.write_dataframe(
    ws, perf,
    title="性能指标",
    titleformat="BLUE_H2",
    headerformat="ORANGE_H4",
    valueformat="NUM%.4",
)
em.close_workbook()

print("已生成 model_report.xlsx")
```

## 完整脚本

将以上片段合并：

```python
import numpy as np
import pandas as pd
from Modeling_Tool import (
    SampleSplitter, WOE_Master, LRMaster,
    GradientBoostingModel, PerformanceEvaluator,
)
from ExcelMaster.ExcelMaster import ExcelMaster

# 1) 数据
rng = np.random.default_rng(42)
n = 5000
data = pd.DataFrame({
    "user_id":    np.arange(n),
    "age":        rng.normal(35, 8, n).clip(18, 70),
    "income":     rng.lognormal(10, 0.4, n),
    "score_b":    rng.normal(600, 60, n),
    "city_grade": rng.choice(["A", "B", "C", "D"], n),
    "n_overdue":  rng.poisson(0.3, n),
})
logit = -6 + 0.02 * (data["score_b"] - 600) + 0.001 * (data["income"] - data["income"].mean())
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

features = ["age", "income", "score_b", "city_grade", "n_overdue"]

# 2) 切分
train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True) \
                     .split_df(data, target="bad_flag")

# 3) WOE
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe, test_woe = woe.transform(train_df), woe.transform(test_df)
woe_features = [f"{f}_woe" for f in features]

# 4) LightGBM
gbm = GradientBoostingModel("lgb", {"n_estimators": 200, "learning_rate": 0.05})
gbm.fit(train_woe[woe_features], train_woe["bad_flag"],
        test_woe[woe_features],  test_woe["bad_flag"])

# 5) 评估
perf = PerformanceEvaluator(
    tgt_name="bad_flag",
    model=gbm._model.model,
    feature_cols=woe_features,
).add_dataset("train", train_woe).add_dataset("test", test_woe).evaluate()

# 6) 报告
em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("Performance")
em.write_dataframe(ws, perf, title="模型性能",
                   titleformat="BLUE_H2",
                   headerformat="ORANGE_H4",
                   valueformat="NUM%.4")
em.close_workbook()
```

## 下一步

- 想了解每一步的更多选项，请阅读 [端到端流水线](pipeline.md)
- 想看所有公开 API 的详细说明，请跳转到 [API 参考](api/index.md)
- 想直接生成可直接部署的脚本，请参考各 [用户指南](guides/index.md) 中的代码片段
