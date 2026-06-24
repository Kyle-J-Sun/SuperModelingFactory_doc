# SuperModelingFactory Documentation

本目录是 [SuperModelingFactory](../SuperModelingFactory/) 工具包的 **MkDocs 文档站点源文件**，与工具包仓库平级存放。

## 本地预览

```bash
# 1) 安装文档依赖
pip install -r requirements-docs.txt

# 2) 安装 SuperModelingFactory 运行时依赖（保证 mkdocstrings 能解析符号）
pip install -r ../SuperModelingFactory/requirements.txt

# 3) 启动本地预览（默认 http://127.0.0.1:8000）
mkdocs serve

# 4) 构建静态站点至 ./site/
mkdocs build --strict
```

## 目录结构

```
SuperModelingFactory-docs/
├── mkdocs.yml                # MkDocs 主配置（主题 / 导航 / mkdocstrings）
├── requirements-docs.txt     # 文档站点依赖
├── docs/                     # 文档源 Markdown
│   ├── index.md              # 首页
│   ├── installation.md       # 安装说明
│   ├── quickstart.md         # 5 分钟快速上手
│   ├── architecture.md       # 依赖图与模块职责
│   ├── pipeline.md           # 端到端建模流水线
│   ├── guides/               # 用户指南
│   └── api/                  # API 参考（mkdocstrings 自动生成）
└── overrides/                # 主题自定义（预留）
```

## 内容组织

| 章节 | 面向读者 | 内容 |
|------|---------|------|
| **首页 / 安装 / 快速上手** | 新用户 | 项目定位、依赖清单、5 分钟示例 |
| **架构 / 流水线** | 中高级用户 | 模块依赖图、典型端到端流水线 |
| **用户指南** | 业务建模师 | 按场景分册（样本/WOE/特征/模型/评估/UAT/报告） |
| **API 参考** | 工具开发 / 高级用户 | 由源码 docstring 自动生成 |

## 写作约定

- 所有示例代码均使用相对路径或标准 import，复制即可运行
- API 标题层级：`<模块>.<类>`、`方法签名`、`参数/返回/示例`
- 中文术语保持原 README 表述（"分箱"、"绑定"、"绑图"、"打分"等）
- 重大变更须同步更新 `pipeline.md` 与 `quickstart.md`

## 构建产物

执行 `mkdocs build` 后会在 `site/` 目录生成完整静态站点，可部署到任意 HTTP 服务器。
