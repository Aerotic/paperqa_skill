# PaperQA2 论文分析 Skill (paperqa_skill)

> 可复用的论文深度分析流水线 — DeepSeek (文本 Q&A) + Qwen3-VL-Plus (多模态图片描述) + st-all-mpnet-base-v2 (本地嵌入)

## 目录

1. [架构概览](#1-架构概览)
2. [环境要求](#2-环境要求)
3. [安装](#3-安装)
4. [快速开始](#4-快速开始)
5. [Python API](#5-python-api)
6. [命令行使用](#6-命令行使用)
7. [分析策略](#7-分析策略)
8. [HTML 报告生成](#8-html-报告生成)
9. [脚本索引](#9-脚本索引)

---

## 1. 架构概览

```
┌──────────────────────────────────────────────────────┐
│                    paperqa_skill                      │
├──────────────────────────────────────────────────────┤
│  Step 1: PDF → 文本分块 + 图片提取 (pypdf)            │
│  Step 2: 本地嵌入向量化 (st-all-mpnet-base-v2)        │
│  Step 3: 多模态图片增强 (Qwen3-VL-Plus @ Bailian)     │
│  Step 4: 文本 Q&A (DeepSeek Chat)                    │
│  Step 5: HTML 图文报告生成                            │
└──────────────────────────────────────────────────────┘
```

### 双 LLM 架构

| 组件 | 模型 | 用途 | API |
|------|------|------|-----|
| 文本 LLM | DeepSeek Chat | 全部文本 Q&A, 上下文摘要 | `api.deepseek.com` |
| Summary LLM | DeepSeek Chat | 上下文摘要文本 | `api.deepseek.com` |
| 多模态 LLM | Qwen3-VL-Plus | 图片/图表描述、多模态问答 | `dashscope-intl.aliyuncs.com` |
| 嵌入模型 | st-all-mpnet-base-v2 | 本地语义搜索 | 本地 (sentence-transformers) |

---

## 2. 环境要求

- **OS:** Windows (已验证), Linux/macOS 理论上可运行
- **Python:** ≥ 3.10 (实测 Python 3.14)
- **依赖:**
  - `paper-qa>=2026.3.18`
  - `paper-qa[local]` (安装本地嵌入依赖)
  - 安装时会自动拉取: `pypdf`, `litellm`, `lmi`, `sentence-transformers` 等

- **API Keys:**
  - DeepSeek: `https://api.deepseek.com` (文本模型)
  - Bailian (阿里百炼): `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (Qwen3-VL-Plus)

---

## 3. 安装

```bash
# 1. 安装 paper-qa
pip install paper-qa
pip install paper-qa[local]

# 2. 确认安装
python -c "import paperqa; print(paperqa.__version__)"
```

### 环境变量配置 (推荐)

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-deepseek-key"
$env:BAILIAN_API_KEY = "sk-your-bailian-key"
$env:PAPER_DIR = "C:\path\to\papers"
```

---

## 4. 快速开始

### 4.0 输入方式

所有函数同时支持 **PDF URL** 和 **本地路径**。传入 URL 时自动下载。

```python
from paperqa_skill import resolve_pdf_source

# URL → 自动下载 → 返回本地路径
local = resolve_pdf_source("https://arxiv.org/pdf/2401.12345.pdf")
# 支持: 直接 PDF 链接, arXiv, ScienceDirect, DOI, handle.net 等

# 本地路径 → 直接返回
local = resolve_pdf_source(r"C:\papers\paper.pdf")
```

### 4.1 分析一篇论文 (Python)

```python
import asyncio
from paperqa_skill import analyze_paper

# 传入 PDF URL (自动下载) 或本地路径
results = asyncio.run(analyze_paper(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",
    queries=[
        "What is the key problem addressed? Describe the methodology and key results.",
        "Explain the experimental setup, benchmarks, and evaluation metrics in detail.",
    ],
    multimodal=True,
))
```

### 4.2 完整流水线 (含 HTML 报告)

```python
import asyncio
from paperqa_skill import full_pipeline

html = asyncio.run(full_pipeline(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",  # URL 或本地路径
    paper_title="My Paper Title",
    multimodal=True,
))
print(f"Report generated: {html}")
```

### 4.3 命令行

```bash
# 传入 URL 或本地路径
python -m paperqa_skill https://arxiv.org/pdf/2401.12345.pdf --title "论文标题"

# 完整流水线 + HTML 报告
python -m paperqa_skill paper.pdf --full --title "论文标题" --output ./reports
python -m paperqa_skill https://doi.org/10.1016/j.sysarc.2025.103580 --full -t "Android UI"

# 禁用多模态 (仅文本)
python -m paperqa_skill paper.pdf --full --no-multimodal
```

---

## 5. Python API

### `make_settings()`

创建 PaperQA2 Settings 实例。

```python
from paperqa_skill import make_settings

settings = make_settings(
    paper_directory="./papers",   # PDF 目录
    multimodal=True,              # 启用多模态
    agent_type="fake",            # "fake" = 禁用多步 agent 循环
)
```

### `resolve_pdf_source()`

解析 PDF 输入源: URL → 自动下载, 本地路径 → 直接返回。

```python
from paperqa_skill import resolve_pdf_source

# URL 自动下载
pdf_path = resolve_pdf_source(
    "https://arxiv.org/pdf/2401.12345.pdf",
    output_dir="./downloads",  # 下载目录 (可选)
    filename="my_paper",       # 自定义文件名 (可选)
)
# 返回: "C:/.../downloads/my_paper.pdf"
```

### `download_pdf()`

底层 PDF 下载函数，支持多种来源。

```python
from paperqa_skill import download_pdf

pdf_path = await download_pdf(
    url="https://arxiv.org/pdf/2401.12345.pdf",
    output_dir="./downloads",
    filename="my_paper",
)
```

### `analyze_paper()`

对单篇 PDF 执行 N 个独立查询。自动处理 URL → 本地下载。

```python
results = await analyze_paper(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",  # URL 或本地路径
    queries=["query 1", "query 2"],    # 问题列表
    output_dir="./output",              # 输出目录
    multimodal=True,
)
# returns {"query_key": "output_file_path", ...}
```

### `full_pipeline()`

完整流水线: 多角度查询 + HTML 图文报告生成。

```python
html_path = await full_pipeline(
    pdf_source="https://arxiv.org/pdf/2401.12345.pdf",  # URL 或本地路径
    paper_title="My Paper",
    queries=None,         # None = 使用内置标准问题
    output_dir="./output",
    multimodal=True,
)
# returns "/path/to/report.html"
```

---

## 6. 命令行使用

```bash
python -m paperqa_skill [-h] [--title TITLE] [--output DIR] [--no-multimodal] [--full] source

位置参数:
  source                PDF URL 或本地文件路径

可选参数:
  -h, --help            显示帮助
  -t, --title TITLE     论文标题
  -o, --output DIR      输出目录
  --no-multimodal       禁用多模态图片增强
  -f, --full            运行完整流水线 (含 HTML 报告)
```

---

## 7. 分析策略

针对每篇论文推荐以下多角度查询策略：

### 标准 3-Query 策略

| # | 角度 | 示例 Prompt |
|---|------|------------|
| 1 | 总体概览 | "Provide a comprehensive overview of the paper. Describe the key problem, proposed approach, methodology, algorithms, experiments, and results." |
| 2 | 技术深入 | "Explain the core technical contribution in detail. Describe any novel algorithms, architectures, or theoretical foundations." |
| 3 | 实验与对比 | "Describe all experiments: setup, datasets, baselines, metrics, and key quantitative results. How does this compare to prior work?" |

### 增强策略 (多模态启用时)

添加针对图表的问题:
- "Include details from any diagrams, figures, or architectural illustrations."
- "Describe what Figure 1-5 show and their significance to the paper."

---

## 8. HTML 报告生成

`full_pipeline()` 自动生成包含以下特征的 HTML 图文报告:

- 深色渐变标题头
- 核心指标卡片网格
- SVG 架构图 / 流程图占位
- 前后对比表 (Baseline vs Optimized)
- 响应式 CSS (移动端适配)
- 标准引用信息区块

### 自定义 HTML 模板

使用 `templates/report_template.html` 中的 `{{PLACEHOLDER}}` 变量:

| 变量 | 说明 |
|------|------|
| `{{TITLE}}` | 论文标题 |
| `{{SUBTITLE}}` | 副标题 |
| `{{BODY}}` | 分析内容 (HTML) |
| `{{DATE}}` | 生成日期 |
| `{{CITATION}}` | 引用信息 |
| `{{DOI}}` | DOI 链接 |

---

## 9. 脚本索引

`pqa_test/` 目录下积累的完整脚本:

| 脚本 | 用途 |
|------|------|
| `query_methods.py` | 基础 PaperQA2 查询 (纯文本) |
| `query_ghosting_model.py` | 技术深入查询 (纯文本) |
| `query_multimodal.py` | 多模态增强查询 (文本+图片) |
| `query_android_ui.py` | Android 论文概览查询 |
| `query_android_ui_deep.py` | Android 论文技术深入查询 |
| `download_paper.py` | ScienceDirect PDF 下载 |
| `find_pdf_url.py` | PDF URL 提取 v1 |
| `find_pdf2.py` | PDF URL 提取 v2 (HTML 元数据) |
| `find_pdf_oa.py` | Unpaywall/Semantic Scholar 开放获取搜索 |
| `find_pdf_openalex.py` | OpenAlex API OA PDF 查找 |
| `find_pdf_cityu2.py` | CityU Scholars OA PDF 查找 |
| `download_pdf_cityu.py` | CityU PDF 下载 (urllib) |
| `extract_pdf_urls.py` | HTML 页面 PDF 链接提取 |
| `ghostbuster_report.html` | Ghostbuster 图文报告示例 (20KB) |
| `android_ui_report.html` | Android UI 图文报告示例 (20KB) |

---

## PDF 自动下载机制

`resolve_pdf_source()` 和 `download_pdf()` 支持从多种来源获取 PDF:

| 来源类型 | 示例 | 策略 |
|---------|------|------|
| 直接 PDF 链接 | `https://arxiv.org/pdf/2401.12345.pdf` | 直接下载 |
| DOI 页面 | `https://doi.org/10.1016/j.sysarc.2025.103580` | 获取 HTML → 提取 citation_pdf_url |
| ScienceDirect | `https://www.sciencedirect.com/science/article/pii/...` | HTML 扫描 → PDF 链接提取 |
| 机构库 | `https://hdl.handle.net/2031/...` (handle.net) | 跟随重定向 → HTML → PDF 链接 |
| arXiv 抽象页 | `https://arxiv.org/abs/2401.12345` | 页面 HTML 含 PDF 链接 |
| 通用 HTML | 任何含 PDF 链接的页面 | `href` 扫描 + OA 模式匹配 |

> **注意**: ScienceDirect 等商业出版社经常返回 403。如果自动下载失败，工具会给出提示，建议手动下载后传入本地路径。

## 已知问题

- **DeepSeek 成本警告**: `Failed to calculate cost for deepseek-v4-flash` — 无害，仅在 LiteLLM 缺失价格映射时出现
- **多模态上下文摘要错误**: `unknown variant 'image_url', expected 'text'` — DeepSeek 不支持 `image_url` 消息格式，不影响最终答案生成 (图片由 Qwen3-VL-Plus 处理)
- **ScienceDirect 403**: 商业出版社 PDF 通常需通过 OA 仓库 (Unpaywall/Semantic Scholar/机构库) 获取
